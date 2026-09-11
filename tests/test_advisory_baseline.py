from __future__ import annotations

import json
import hashlib
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import verify_dependency_locks as verifier_module
from scripts.advisory_baseline import (
    AUDIT_SURFACES,
    AcceptedAdvisory,
    AdvisoryBaselineError,
    AuditSurfaceError,
    Baseline,
    NonPyPIRecord,
    SURFACE_ORDER,
    compare_baseline,
    derive_lock_audit_surfaces,
    load_baseline,
    main,
    normalize_pip_audit,
    run_audit_surfaces,
)
from scripts.dependency_locks import parse_lock


REPO_ROOT = Path(__file__).resolve().parent.parent
TORCH_CORE_REQUIREMENTS = (
    "torch==2.11.0\n"
    "torchvision==0.26.0\n"
    "torchaudio==2.11.0\n"
)
TORCH_ECOSYSTEM_REQUIREMENTS = (
    "pytorch-lightning==2.6.1\n"
    "torchmetrics==1.9.0\n"
    "torchao==0.18.0\n"
)
PYG_FIND_LINKS = "--find-links https://data.pyg.org/whl/torch-2.11.0+cpu.html"
TORCH_RUNTIME_REQUIREMENTS = (
    "-r torch-ecosystem-requirements.txt\n"
    f"{PYG_FIND_LINKS}\n"
    "pyg-lib==0.8.0\n"
    "torch-scatter==2.1.2\n"
    "torch-sparse==0.6.18\n"
    "torch_geometric==2.8.0.post1\n"
)
TORCH_AUDIT_REQUIREMENTS = (
    "-r torch-core-requirements.txt\n"
    "-r torch-ecosystem-requirements.txt\n"
    "torch_geometric==2.8.0.post1\n"
)
PYG_EXTENSION_AUDIT_REQUIREMENTS = (
    "# Pre-resolved compiled PyG extension supplement for the strict audit.\n"
    "# Runtime source: torch-requirements.txt retains the approved PyG wheel selector.\n"
    "torch-scatter==2.1.2\n"
    "torch-sparse==0.6.18\n"
)


def _write_torch_requirements(repo: Path) -> tuple[Path, Path, Path]:
    (repo / "torch-core-requirements.txt").write_text(TORCH_CORE_REQUIREMENTS, encoding="utf-8")
    (repo / "torch-ecosystem-requirements.txt").write_text(TORCH_ECOSYSTEM_REQUIREMENTS, encoding="utf-8")
    runtime = repo / "torch-requirements.txt"
    audit = repo / "torch-audit-requirements.txt"
    extensions = repo / "pyg-extension-audit-requirements.txt"
    runtime.write_text(TORCH_RUNTIME_REQUIREMENTS, encoding="utf-8")
    audit.write_text(TORCH_AUDIT_REQUIREMENTS, encoding="utf-8")
    extensions.write_text(PYG_EXTENSION_AUDIT_REQUIREMENTS, encoding="utf-8")
    return runtime, audit, extensions


@pytest.fixture(autouse=True)
def _torch_audit_projection(tmp_path: Path) -> None:
    _copy_lock_audit_contract(tmp_path)


def _canonical_document(*, entries: list[dict[str, object]] | None = None) -> dict[str, object]:
    return {
        "schema_version": 1,
        "audited_surfaces": ["combined-runtime", "torch", "documentation", "atlas-contract"],
        "accepted_advisories": entries
        if entries is not None
        else [
            {
                "advisory_id": "CVE-2025-2148",
                "package": "torch",
                "accepted_version": "2.4.1",
                "surfaces": ["combined-runtime", "torch"],
            }
        ],
    }


def _write_policy(path: Path, document: dict[str, object]) -> None:
    path.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def test_load_baseline_accepts_canonical_policy(tmp_path: Path) -> None:
    path = tmp_path / "baseline.json"
    _write_policy(path, _canonical_document())

    baseline = load_baseline(path)

    assert baseline.schema_version == 1
    assert baseline.audited_surfaces == ("combined-runtime", "torch", "documentation", "atlas-contract")
    assert [(item.package, item.advisory_id, item.accepted_version, item.surfaces) for item in baseline.accepted_advisories] == [
        ("torch", "CVE-2025-2148", "2.4.1", ("combined-runtime", "torch"))
    ]


@pytest.mark.parametrize(
    "document",
    [
        {**_canonical_document(), "extra": True},
        _canonical_document(entries=[{**_canonical_document()["accepted_advisories"][0], "extra": True}]),
    ],
)
def test_load_baseline_rejects_unknown_root_and_item_keys(tmp_path: Path, document: dict[str, object]) -> None:
    path = tmp_path / "baseline.json"
    _write_policy(path, document)

    with pytest.raises(AdvisoryBaselineError, match="unknown"):
        load_baseline(path)


@pytest.mark.parametrize(
    "document",
    [
        {
            "audited_surfaces": _canonical_document()["audited_surfaces"],
            "schema_version": 1,
            "accepted_advisories": _canonical_document()["accepted_advisories"],
        },
        _canonical_document(
            entries=[
                {
                    "package": "torch",
                    "advisory_id": "CVE-2025-2148",
                    "accepted_version": "2.4.1",
                    "surfaces": ["combined-runtime", "torch"],
                }
            ]
        ),
    ],
)
def test_load_baseline_rejects_reordered_root_and_item_keys(tmp_path: Path, document: dict[str, object]) -> None:
    path = tmp_path / "baseline.json"
    _write_policy(path, document)

    with pytest.raises(AdvisoryBaselineError, match="key order"):
        load_baseline(path)


def test_load_baseline_rejects_unsupported_schema_version(tmp_path: Path) -> None:
    path = tmp_path / "baseline.json"
    document = _canonical_document()
    document["schema_version"] = 2
    _write_policy(path, document)

    with pytest.raises(AdvisoryBaselineError, match="schema version"):
        load_baseline(path)


@pytest.mark.parametrize(
    "surfaces",
    [
        ["combined-runtime", "torch", "documentation"],
        ["combined-runtime", "torch", "documentation", "atlas-contract", "extra"],
        ["combined-runtime", "torch", "torch", "atlas-contract"],
        ["torch", "combined-runtime", "documentation", "atlas-contract"],
    ],
)
def test_load_baseline_rejects_missing_extra_duplicate_or_reordered_surfaces(tmp_path: Path, surfaces: list[str]) -> None:
    path = tmp_path / "baseline.json"
    document = _canonical_document()
    document["audited_surfaces"] = surfaces
    _write_policy(path, document)

    with pytest.raises(AdvisoryBaselineError, match="audited surfaces"):
        load_baseline(path)


def test_load_baseline_rejects_duplicate_identity_after_pep503_normalization(tmp_path: Path) -> None:
    path = tmp_path / "baseline.json"
    entry = _canonical_document()["accepted_advisories"][0]
    assert isinstance(entry, dict)
    _write_policy(path, _canonical_document(entries=[entry, {**entry, "package": "Torch"}]))

    with pytest.raises(AdvisoryBaselineError, match="duplicate"):
        load_baseline(path)


def test_load_baseline_rejects_lone_noncanonical_package_spelling(tmp_path: Path) -> None:
    path = tmp_path / "baseline.json"
    entry = _canonical_document()["accepted_advisories"][0]
    assert isinstance(entry, dict)
    _write_policy(path, _canonical_document(entries=[{**entry, "package": "Torch"}]))

    with pytest.raises(AdvisoryBaselineError, match="canonical package"):
        load_baseline(path)


@pytest.mark.parametrize(
    "surfaces",
    [["combined-runtime", "unknown"], ["combined-runtime", "combined-runtime"]],
)
def test_load_baseline_rejects_unknown_or_duplicate_item_surfaces(tmp_path: Path, surfaces: list[str]) -> None:
    path = tmp_path / "baseline.json"
    entry = _canonical_document()["accepted_advisories"][0]
    assert isinstance(entry, dict)
    _write_policy(path, _canonical_document(entries=[{**entry, "surfaces": surfaces}]))

    with pytest.raises(AdvisoryBaselineError, match="surfaces"):
        load_baseline(path)


def test_load_baseline_rejects_unsorted_items_and_noncanonical_serialization(tmp_path: Path) -> None:
    path = tmp_path / "baseline.json"
    entries = [
        {
            "advisory_id": "CVE-2025-2149",
            "package": "torch",
            "accepted_version": "2.4.1",
            "surfaces": ["combined-runtime", "torch"],
        },
        {
            "advisory_id": "CVE-2025-2148",
            "package": "torch",
            "accepted_version": "2.4.1",
            "surfaces": ["combined-runtime", "torch"],
        },
    ]
    _write_policy(path, _canonical_document(entries=entries))

    with pytest.raises(AdvisoryBaselineError, match="sorted"):
        load_baseline(path)

    _write_policy(path, _canonical_document())
    path.write_text(path.read_text(encoding="utf-8").replace("\n", "\r\n"), encoding="utf-8", newline="")
    with pytest.raises(AdvisoryBaselineError, match="canonical"):
        load_baseline(path)


def test_real_baseline_contains_current_issue62_reviewed_identities() -> None:
    baseline = load_baseline(REPO_ROOT / "security/accepted-advisories.json")

    assert {(item.package, item.advisory_id, item.accepted_version) for item in baseline.accepted_advisories} == {
        ("nltk", "PYSEC-2026-3740", "3.10.3"),
        ("pytorch-lightning", "PYSEC-2026-3967", "2.6.1"),
        ("setuptools", "PYSEC-2026-3447", "81.0.0"),
        ("torch", "PYSEC-2025-194", "2.11.0"),
    }


def test_real_baseline_contains_exact_reviewed_policy_quadruples() -> None:
    baseline = load_baseline(REPO_ROOT / "security/accepted-advisories.json")

    assert {
        (item.package, item.advisory_id, item.accepted_version, item.surfaces)
        for item in baseline.accepted_advisories
    } == {
        ("nltk", "PYSEC-2026-3740", "3.10.3", ("combined-runtime", "atlas-contract")),
        ("pytorch-lightning", "PYSEC-2026-3967", "2.6.1", ("combined-runtime", "torch")),
        ("setuptools", "PYSEC-2026-3447", "81.0.0", SURFACE_ORDER),
        ("torch", "PYSEC-2025-194", "2.11.0", ("combined-runtime", "torch")),
    }


def _payload(*dependencies: dict[str, object]) -> dict[str, object]:
    return {"dependencies": list(dependencies), "fixes": []}


def _dependency(name: str, version: str, *vulnerability_ids: str) -> dict[str, object]:
    return {
        "name": name,
        "version": version,
        "vulns": [
            {"id": vulnerability_id, "aliases": ["CVE-elsewhere"], "fix_versions": ["9.9.9"]}
            for vulnerability_id in vulnerability_ids
        ],
    }


def _baseline() -> Baseline:
    return Baseline(
        schema_version=1,
        audited_surfaces=("combined-runtime", "torch", "documentation", "atlas-contract"),
        accepted_advisories=(
            AcceptedAdvisory("torch", "CVE-2025-2148", "2.4.1", ("combined-runtime", "torch")),
        ),
    )


def _observations(*surface_payloads: tuple[str, object]):
    return tuple(normalize_pip_audit(surface, payload) for surface, payload in surface_payloads)


def _complete_observations(*, combined: object, torch: object) -> tuple:
    return _observations(
        ("combined-runtime", combined),
        ("torch", torch),
        ("documentation", _payload()),
        ("atlas-contract", _payload()),
    )


def test_normalize_accepts_representative_pip_audit_2_10_clean_and_vulnerable_payloads() -> None:
    clean = normalize_pip_audit("documentation", _payload(_dependency("mkdocs", "9.0")))
    vulnerable = normalize_pip_audit(
        "torch",
        _payload(_dependency("Torch", "2.4.1", "CVE-2025-2148")),
    )

    assert clean.resolved_versions == (("mkdocs", "9.0"),)
    assert clean.advisories == ()
    assert vulnerable.resolved_versions == (("torch", "2.4.1"),)
    assert vulnerable.advisories == (("torch", "2.4.1", "CVE-2025-2148"),)


def test_normalize_collapses_duplicate_raw_records_to_one_identity() -> None:
    observation = normalize_pip_audit(
        "torch",
        _payload(
            _dependency("torch", "2.4.1", "PYSEC-2025-191", "PYSEC-2025-191"),
        ),
    )

    assert observation.advisories == (("torch", "2.4.1", "PYSEC-2025-191"),)


@pytest.mark.parametrize(
    "surface,payload",
    [
        ("unknown", _payload()),
        ("torch", []),
        ("torch", {"dependencies": _payload()["dependencies"]}),
        ("torch", {"dependencies": _payload()["dependencies"], "fixes": [], "extra": True}),
        ("torch", {"dependencies": _payload()["dependencies"], "fixes": [{}]}),
        ("torch", _payload({"name": "torch", "version": "2.4.1"})),
        ("torch", _payload({"name": "torch", "version": "2.4.1", "vulns": [{"id": 1}]})),
        (
            "torch",
            _payload(
                {"name": "torch", "version": "2.4.1", "vulns": [{"id": "CVE-2025-2148", "aliases": "bad"}]}
            ),
        ),
        ("torch", _payload(_dependency("torch", "2.4.1"), _dependency("torch", "2.5.0"))),
    ],
)
def test_normalize_rejects_malformed_payload_and_dependency_records(surface: str, payload: object) -> None:
    with pytest.raises(AdvisoryBaselineError):
        normalize_pip_audit(surface, payload)


@pytest.mark.parametrize(
    "dependency",
    [
        {"name": "torch", "version": "2.4.1", "vulns": [], "extra": True},
        {"name": "torch", "version": "2.4.1", "vulns": [{"id": "CVE", "fix_versions": []}]},
        {"name": "torch", "version": "2.4.1", "vulns": [{"id": "CVE", "fix_versions": [], "aliases": [], "extra": True}]},
    ],
)
def test_normalize_rejects_unknown_or_missing_actual_pip_audit_2_10_fields(dependency: dict[str, object]) -> None:
    with pytest.raises(AdvisoryBaselineError):
        normalize_pip_audit("torch", _payload(dependency))


def test_normalize_rejects_duplicate_normalized_dependencies_even_at_the_same_version() -> None:
    with pytest.raises(AdvisoryBaselineError, match="duplicate"):
        normalize_pip_audit(
            "torch",
            _payload(_dependency("torch", "2.4.1"), _dependency("Torch", "2.4.1")),
        )


def test_compare_accepts_exact_observations() -> None:
    result = compare_baseline(
        _baseline(),
        _complete_observations(
            combined=_payload(_dependency("torch", "2.4.1", "CVE-2025-2148")),
            torch=_payload(_dependency("torch", "2.4.1", "CVE-2025-2148")),
        ),
    )

    assert result.errors == ()
    assert result.notices == ()


def test_compare_fails_on_new_primary_id() -> None:
    result = compare_baseline(
        _baseline(),
        _complete_observations(
            combined=_payload(_dependency("torch", "2.4.1", "CVE-2025-2148", "CVE-2025-9999")),
            torch=_payload(_dependency("torch", "2.4.1", "CVE-2025-2148")),
        ),
    )

    assert result.errors == ("new advisory: torch 2.4.1 CVE-2025-9999 on combined-runtime",)


@pytest.mark.parametrize(
    "combined,torch,expected",
    [
        (
            _payload(_dependency("other", "2.4.1", "CVE-2025-2148")),
            _payload(_dependency("torch", "2.4.1", "CVE-2025-2148")),
            "new advisory: other 2.4.1 CVE-2025-2148 on combined-runtime",
        ),
        (
            _payload(_dependency("torch", "2.4.1", "CVE-2025-2148")),
            _payload(),
            "surface drift: torch 2.4.1 CVE-2025-2148 expected [combined-runtime, torch]; observed [combined-runtime]",
        ),
    ],
)
def test_compare_fails_on_package_or_surface_drift(combined: object, torch: object, expected: str) -> None:
    result = compare_baseline(_baseline(), _complete_observations(combined=combined, torch=torch))

    assert expected in result.errors


def test_compare_fails_on_accepted_version_drift_with_advisory_present() -> None:
    result = compare_baseline(
        _baseline(),
        _complete_observations(
            combined=_payload(_dependency("torch", "2.5.0", "CVE-2025-2148")),
            torch=_payload(_dependency("torch", "2.5.0", "CVE-2025-2148")),
        ),
    )

    assert result.errors == ("accepted version drift: torch expected 2.4.1; observed 2.5.0",)


def test_compare_fails_on_accepted_version_drift_with_empty_vulnerability_list() -> None:
    result = compare_baseline(
        _baseline(),
        _complete_observations(
            combined=_payload(_dependency("torch", "2.5.0")),
            torch=_payload(_dependency("torch", "2.5.0")),
        ),
    )

    assert result.errors == ("accepted version drift: torch expected 2.4.1; observed 2.5.0",)


def test_compare_checks_accepted_versions_on_clean_unexpected_surfaces() -> None:
    result = compare_baseline(
        _baseline(),
        _observations(
            ("combined-runtime", _payload(_dependency("torch", "2.4.1", "CVE-2025-2148"))),
            ("torch", _payload(_dependency("torch", "2.4.1", "CVE-2025-2148"))),
            ("documentation", _payload(_dependency("torch", "9.9.9"))),
            ("atlas-contract", _payload()),
        ),
    )

    assert result.errors == ("accepted version drift: torch expected 2.4.1; observed 9.9.9",)


def test_compare_rejects_clean_accepted_version_on_an_unexpected_surface() -> None:
    result = compare_baseline(
        _baseline(),
        _observations(
            ("combined-runtime", _payload(_dependency("torch", "2.4.1", "CVE-2025-2148"))),
            ("torch", _payload(_dependency("torch", "2.4.1", "CVE-2025-2148"))),
            ("documentation", _payload(_dependency("torch", "2.4.1"))),
            ("atlas-contract", _payload()),
        ),
    )

    assert result.errors == (
        "surface drift: torch 2.4.1 observed on documentation; expected [combined-runtime, torch]",
    )


def test_compare_rejects_cross_product_advisory_versions_even_when_surfaces_match() -> None:
    baseline = Baseline(
        schema_version=1,
        audited_surfaces=("combined-runtime", "torch", "documentation", "atlas-contract"),
        accepted_advisories=(
            AcceptedAdvisory("torch", "ADVISORY-A", "1.0", ("combined-runtime",)),
            AcceptedAdvisory("torch", "ADVISORY-B", "2.0", ("torch",)),
        ),
    )
    observations = _complete_observations(
        combined=_payload(_dependency("torch", "2.0", "ADVISORY-A")),
        torch=_payload(_dependency("torch", "2.0", "ADVISORY-B")),
    )

    result = compare_baseline(baseline, observations)

    assert result.errors == (
        "accepted version drift: torch expected 1.0; observed 2.0",
        "surface drift: torch 2.0 observed on combined-runtime; expected [torch]",
    )
    assert result.notices == ("reconcile removed advisory: torch 1.0 ADVISORY-A",)


def test_compare_rejects_clean_cross_product_versions_on_the_wrong_surface() -> None:
    baseline = Baseline(
        schema_version=1,
        audited_surfaces=("combined-runtime", "torch", "documentation", "atlas-contract"),
        accepted_advisories=(
            AcceptedAdvisory("torch", "ADVISORY-A", "1.0", ("combined-runtime",)),
            AcceptedAdvisory("torch", "ADVISORY-B", "2.0", ("torch",)),
        ),
    )
    result = compare_baseline(
        baseline,
        _complete_observations(
            combined=_payload(_dependency("torch", "2.0")),
            torch=_payload(_dependency("torch", "2.0", "ADVISORY-B")),
        ),
    )

    assert result.errors
    assert "accepted version drift: torch expected 1.0; observed 2.0" in result.errors
    assert result.notices == ("reconcile removed advisory: torch 1.0 ADVISORY-A",)


def _versioned_identity_baseline() -> Baseline:
    return Baseline(
        schema_version=1,
        audited_surfaces=("combined-runtime", "torch", "documentation", "atlas-contract"),
        accepted_advisories=(
            AcceptedAdvisory("torch", "ADVISORY", "1.0", ("combined-runtime",)),
            AcceptedAdvisory("torch", "ADVISORY", "2.0", ("torch",)),
        ),
    )


def test_load_baseline_sorts_versions_before_advisory_ids(tmp_path: Path) -> None:
    path = tmp_path / "baseline.json"
    entries = [
        {"advisory_id": "ADVISORY-A", "package": "torch", "accepted_version": "2.0", "surfaces": ["torch"]},
        {"advisory_id": "ADVISORY-Z", "package": "torch", "accepted_version": "1.0", "surfaces": ["torch"]},
    ]
    _write_policy(path, _canonical_document(entries=entries))

    with pytest.raises(AdvisoryBaselineError, match="sorted"):
        load_baseline(path)


def test_compare_accepts_same_primary_id_at_two_exact_accepted_versions() -> None:
    result = compare_baseline(
        _versioned_identity_baseline(),
        _complete_observations(
            combined=_payload(_dependency("torch", "1.0", "ADVISORY")),
            torch=_payload(_dependency("torch", "2.0", "ADVISORY")),
        ),
    )

    assert result.errors == ()
    assert result.notices == ()


def test_compare_reports_only_the_removed_versioned_identity() -> None:
    result = compare_baseline(
        _versioned_identity_baseline(),
        _complete_observations(
            combined=_payload(),
            torch=_payload(_dependency("torch", "2.0", "ADVISORY")),
        ),
    )

    assert result.errors == ()
    assert result.notices == ("reconcile removed advisory: torch 1.0 ADVISORY",)


def test_compare_fails_when_a_versioned_identity_moves_to_the_wrong_surface() -> None:
    result = compare_baseline(
        _versioned_identity_baseline(),
        _observations(
            ("combined-runtime", _payload()),
            ("torch", _payload(_dependency("torch", "2.0", "ADVISORY"))),
            ("documentation", _payload(_dependency("torch", "1.0", "ADVISORY"))),
            ("atlas-contract", _payload()),
        ),
    )

    assert result.errors == (
        "surface drift: torch 1.0 ADVISORY expected [combined-runtime]; observed [documentation]",
        "surface drift: torch 1.0 observed on documentation; expected [combined-runtime]",
    )
    assert result.notices == ()


def test_compare_reports_removed_identity_without_failing() -> None:
    result = compare_baseline(
        _baseline(),
        _complete_observations(
            combined=_payload(_dependency("torch", "2.4.1")),
            torch=_payload(_dependency("torch", "2.4.1")),
        ),
    )

    assert result.errors == ()
    assert result.notices == ("reconcile removed advisory: torch 2.4.1 CVE-2025-2148",)


def test_compare_reports_removed_package_without_failing() -> None:
    result = compare_baseline(
        _baseline(),
        _complete_observations(combined=_payload(), torch=_payload()),
    )

    assert result.errors == ()
    assert result.notices == ("reconcile removed advisory: torch 2.4.1 CVE-2025-2148",)


def test_alias_only_and_fix_version_changes_do_not_change_policy_identity() -> None:
    payload = _payload(
        {
            "name": "torch",
            "version": "2.4.1",
            "vulns": [
                {
                    "id": "CVE-2025-2148",
                    "aliases": ["updated-alias"],
                    "fix_versions": ["100.0.0"],
                }
            ],
        }
    )
    result = compare_baseline(
        _baseline(),
        _complete_observations(combined=payload, torch=payload),
    )

    assert result.errors == ()
    assert result.notices == ()


def _audit_runner(
    *,
    returncode: int = 0,
    payload: object = None,
    write_output: bool = True,
    raw_output: str | None = None,
):
    commands: list[tuple[tuple[str, ...], dict[str, object]]] = []

    def run(command: list[str], **kwargs: object) -> SimpleNamespace:
        commands.append((tuple(command), kwargs))
        if write_output:
            output = Path(command[command.index("--output") + 1])
            projection = Path(command[command.index("-r") + 1])
            dependencies = tuple(
                _dependency(*line.split("==", 1))
                for line in projection.read_text(encoding="utf-8").splitlines()
            )
            audit_payload = _payload(*dependencies) if payload is None else payload
            if "pyg-extensions" in output.name:
                audit_payload = _payload(*dependencies)
            output.write_text(
                raw_output if raw_output is not None else json.dumps(audit_payload),
                encoding="utf-8",
            )
        return SimpleNamespace(returncode=returncode, stdout="", stderr="")

    return commands, run


def _assert_exact_audit_commands(commands: list[tuple[tuple[str, ...], dict[str, object]]], repo: Path) -> None:
    expected_outputs = (
        "combined-runtime-resolver.json",
        "combined-runtime-pyg-extensions.json",
        "torch-resolver.json",
        "torch-pyg-extensions.json",
        "documentation.json",
        "atlas-contract.json",
    )
    fixed_flags = (
        "--strict",
        "--vulnerability-service",
        "pypi",
        "--format",
        "json",
        "--aliases",
        "on",
        "--desc",
        "off",
        "--progress-spinner",
        "off",
    )
    assert len(commands) == len(expected_outputs)
    output_directories = set()
    for (command, kwargs), output_name in zip(commands, expected_outputs, strict=True):
        output = Path(command[-1])
        projection = Path(command[command.index("-r") + 1])
        assert command == (
            sys.executable,
            "-m",
            "pip_audit",
            "--disable-pip",
            "--no-deps",
            "-r",
            str(projection),
            *fixed_flags,
            "--output",
            str(output),
        )
        assert output.name == output_name
        assert output.parent != repo
        assert projection.parent == output.parent / "projections"
        assert projection.name == output_name.removesuffix(".json") + ".requirements.txt"
        output_directories.add(output.parent)
        assert kwargs == {"cwd": repo, "check": False, "capture_output": True, "text": True}
    assert len(output_directories) == 1


def test_audit_commands_match_the_complete_four_surface_contract(tmp_path: Path) -> None:
    commands, runner = _audit_runner()

    run_audit_surfaces(tmp_path, runner=runner)

    _assert_exact_audit_commands(commands, tmp_path)


def test_audit_surfaces_define_six_physical_commands_for_four_logical_observations() -> None:
    assert [
        (surface.name, surface.projection_kind, surface.output_name)
        for surface in AUDIT_SURFACES
    ] == [
        ("combined-runtime", "main", "combined-runtime-resolver"),
        ("combined-runtime", "pyg-extensions", "combined-runtime-pyg-extensions"),
        ("torch", "main", "torch-resolver"),
        ("torch", "pyg-extensions", "torch-pyg-extensions"),
        ("documentation", "main", "documentation"),
        ("atlas-contract", "main", "atlas-contract"),
    ]


def test_audit_merges_nonempty_pyg_extension_supplements_into_logical_surfaces(tmp_path: Path) -> None:
    calls = 0

    def runner(command: list[str], **kwargs: object) -> SimpleNamespace:
        nonlocal calls
        output = Path(command[command.index("--output") + 1])
        projection = Path(command[command.index("-r") + 1])
        extension = "pyg-extensions" in output.name
        dependencies = tuple(
            _dependency(name, version, "CVE-2099-0001")
            if extension and name == "torch-scatter"
            else _dependency(name, version)
            for name, version in (line.split("==", 1) for line in projection.read_text().splitlines())
        )
        output.write_text(json.dumps(_payload(*dependencies)), encoding="utf-8")
        calls += 1
        return SimpleNamespace(returncode=1 if extension else 0, stdout="", stderr="")

    observations = run_audit_surfaces(tmp_path, runner=runner)

    assert calls == 6
    assert [item.surface for item in observations] == list(SURFACE_ORDER)
    assert all(
        {("torch-scatter", "2.1.2"), ("torch-sparse", "0.6.18")} <= set(item.resolved_versions)
        for item in observations[:2]
    )
    assert all(
        item.advisories == (("torch-scatter", "2.1.2", "CVE-2099-0001"),)
        for item in observations[:2]
    )
    assert all(("setuptools", "81.0.0") in item.resolved_versions for item in observations)
    assert all(
        package != "pyg-lib"
        for observation in observations
        for package, _ in observation.resolved_versions
    )


@pytest.mark.parametrize(
    ("target_output", "surface"),
    (
        ("combined-runtime-pyg-extensions.json", "combined-runtime"),
        ("torch-pyg-extensions.json", "torch"),
    ),
)
@pytest.mark.parametrize(
    ("case", "dependencies"),
    (
        ("empty", ()),
        ("missing-scatter", (("torch-sparse", "0.6.18"),)),
        ("missing-sparse", (("torch-scatter", "2.1.2"),)),
        (
            "extra-package",
            (
                ("rogue-package", "1.0.0"),
                ("torch-scatter", "2.1.2"),
                ("torch-sparse", "0.6.18"),
            ),
        ),
        (
            "scatter-version-drift",
            (("torch-scatter", "2.1.3"), ("torch-sparse", "0.6.18")),
        ),
        (
            "sparse-version-drift",
            (("torch-scatter", "2.1.2"), ("torch-sparse", "0.6.19")),
        ),
    ),
)
def test_audit_rejects_noncanonical_pyg_extension_supplement_identity(
    tmp_path: Path,
    target_output: str,
    surface: str,
    case: str,
    dependencies: tuple[tuple[str, str], ...],
) -> None:
    del case
    expected = (("torch-scatter", "2.1.2"), ("torch-sparse", "0.6.18"))
    assert dependencies != expected
    target_calls = 0

    def runner(command: list[str], **kwargs: object) -> SimpleNamespace:
        nonlocal target_calls
        output = Path(command[command.index("--output") + 1])
        selected = dependencies if output.name == target_output else expected
        projection = Path(command[command.index("-r") + 1])
        if "pyg-extensions" not in output.name:
            selected = tuple(line.split("==", 1) for line in projection.read_text().splitlines())
        payload = _payload(*(_dependency(package, version) for package, version in selected))
        output.write_text(json.dumps(payload), encoding="utf-8")
        if output.name == target_output:
            target_calls += 1
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    with pytest.raises(AuditSurfaceError, match=rf"{surface}: invalid-schema"):
        run_audit_surfaces(tmp_path, runner=runner)

    assert target_calls == 1


def test_audit_rejects_overlapping_resolver_and_supplement_packages(tmp_path: Path) -> None:
    def runner(command: list[str], **kwargs: object) -> SimpleNamespace:
        output = Path(command[command.index("--output") + 1])
        projection = Path(command[command.index("-r") + 1])
        selected = [line.split("==", 1) for line in projection.read_text().splitlines()]
        if "pyg-extensions" not in output.name:
            selected.append(["torch-scatter", "2.1.2"])
        payload = _payload(*(_dependency(package, version) for package, version in selected))
        output.write_text(json.dumps(payload), encoding="utf-8")
        return SimpleNamespace(returncode=1, stdout="", stderr="")

    with pytest.raises(AuditSurfaceError, match="combined-runtime: invalid-schema"):
        run_audit_surfaces(tmp_path, runner=runner)


@pytest.mark.parametrize(
    "command_index,mutation",
    [
        (0, lambda command: command.__setitem__(command.index("off", command.index("--progress-spinner")), "on")),
        (2, lambda command: command.__setitem__(command.index("-r") + 1, "requirements.txt")),
        (5, lambda command: command.__setitem__(command.index("-r") + 1, "requirements.txt")),
        (0, lambda command: command.remove("--strict")),
        (0, lambda command: command.insert(command.index("--format"), "--ignore-vuln")),
        (
            0,
            lambda command: command.__setitem__(
                slice(command.index("--format"), command.index("--format") + 2),
                ["json", "--format"],
            ),
        ),
    ],
)
def test_exact_audit_command_contract_rejects_weakened_or_reordered_tokens(
    tmp_path: Path,
    command_index: int,
    mutation,
) -> None:
    commands, runner = _audit_runner()
    run_audit_surfaces(tmp_path, runner=runner)
    mutated = list(commands)
    command, kwargs = mutated[command_index]
    command_list = list(command)
    mutation(command_list)
    mutated[command_index] = (tuple(command_list), kwargs)

    with pytest.raises(AssertionError):
        _assert_exact_audit_commands(mutated, tmp_path)


def test_audit_commands_cover_exact_four_surfaces_and_flags(tmp_path: Path) -> None:
    commands, runner = _audit_runner()

    observations = run_audit_surfaces(tmp_path, runner=runner)

    assert [observation.surface for observation in observations] == [
        "combined-runtime",
        "torch",
        "documentation",
        "atlas-contract",
    ]
    assert [command[:3] for command, _ in commands] == [(sys.executable, "-m", "pip_audit")] * 6
    for command, kwargs in commands:
        assert kwargs["cwd"] == tmp_path
        assert kwargs["check"] is False
        assert {"--strict", "--vulnerability-service", "pypi", "--format", "json"} <= set(command)
        assert {"--aliases", "on", "--desc", "off", "--progress-spinner"} <= set(command)
        assert command[3:7] == ("--disable-pip", "--no-deps", "-r", command[6])
        assert command.count("-r") == 1
        assert "--ignore-vuln" not in command


def test_documentation_audit_uses_disable_pip(tmp_path: Path) -> None:
    commands, runner = _audit_runner()

    run_audit_surfaces(tmp_path, runner=runner)

    documentation = commands[4][0]
    assert documentation[3:7] == ("--disable-pip", "--no-deps", "-r", documentation[6])
    assert Path(documentation[6]).name == "documentation.requirements.txt"


@pytest.mark.parametrize("returncode", [0, 1])
def test_audit_exit_zero_and_one_are_completed_observations(tmp_path: Path, returncode: int) -> None:
    _, runner = _audit_runner(returncode=returncode)

    observations = run_audit_surfaces(tmp_path, runner=runner)

    assert len(observations) == 4
    assert all(("setuptools", "81.0.0") in observation.resolved_versions for observation in observations)


@pytest.mark.parametrize(
    "runner",
    [
        _audit_runner(returncode=2)[1],
        _audit_runner(write_output=False)[1],
        _audit_runner(payload="not-json")[1],
    ],
)
def test_audit_other_exit_missing_output_and_malformed_json_fail_closed(tmp_path: Path, runner) -> None:
    with pytest.raises(AdvisoryBaselineError):
        run_audit_surfaces(tmp_path, runner=runner)


def _assert_safe_audit_failure_output(
    captured: pytest.CaptureResult[str],
    *,
    surface: str,
    category: str,
) -> None:
    assert captured.out == ""
    assert captured.err == f"advisory audit failed: {surface}: {category}\n"


@pytest.mark.parametrize(
    ("failure", "failure_index", "expected_surface", "expected_category"),
    [
        ("unexpected-exit", 0, "combined-runtime", "unexpected-exit"),
        ("runner-oserror", 2, "torch", "execution-error"),
        ("runner-unicode-error", 4, "documentation", "execution-error"),
        ("missing-output", 5, "atlas-contract", "missing-output"),
        ("bootstrap-error", 2, "torch", "bootstrap-error"),
        ("resolution-error", 4, "documentation", "resolution-error"),
        ("service-error", 4, "documentation", "service-error"),
        ("unsupported-package", 5, "atlas-contract", "unsupported-package"),
        ("unavailable-output", 2, "torch", "unavailable-output"),
        ("invalid-json", 4, "documentation", "invalid-json"),
        ("invalid-schema", 5, "atlas-contract", "invalid-schema"),
    ],
)
def test_cli_reports_every_audit_failure_with_only_fixed_safe_output(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failure: str,
    failure_index: int,
    expected_surface: str,
    expected_category: str,
) -> None:
    import scripts.advisory_baseline as module

    unsafe = "https://user:secret@example.test /private/tmp/audit.json --index-url"
    calls = 0
    load_calls = 0
    original_load = module._load_pip_audit_output

    def runner(command: list[str], **kwargs: object) -> SimpleNamespace:
        nonlocal calls
        index = calls
        calls += 1
        if index == failure_index and failure == "runner-oserror":
            raise OSError(unsafe)
        if index == failure_index and failure == "runner-unicode-error":
            raise UnicodeDecodeError("utf-8", b"\\xff", 0, 1, unsafe)
        output = Path(command[command.index("--output") + 1])
        if index == failure_index and failure in {
            "missing-output",
            "bootstrap-error",
            "resolution-error",
            "service-error",
            "unsupported-package",
        }:
            marker = {
                "bootstrap-error": "Failed to upgrade `pip`",
                "resolution-error": "Failed to install packages",
                "service-error": "Tip: your network may be blocking this service",
                "unsupported-package": "Dependency not found on PyPI and could not be audited",
            }.get(failure, "")
            return SimpleNamespace(returncode=1, stdout=unsafe, stderr=f"{marker}: {unsafe}")
        if index == failure_index and failure == "invalid-json":
            output.write_text("not-json", encoding="utf-8")
        elif index == failure_index and failure == "invalid-schema":
            output.write_text("[]", encoding="utf-8")
        else:
            projection = Path(command[command.index("-r") + 1])
            dependencies = (
                _dependency(*line.split("==", 1))
                for line in projection.read_text(encoding="utf-8").splitlines()
            )
            output.write_text(json.dumps(_payload(*dependencies)), encoding="utf-8")
        return SimpleNamespace(
            returncode=2 if index == failure_index and failure == "unexpected-exit" else 1,
            stdout=unsafe,
            stderr=unsafe,
        )

    def unavailable_loader(path: Path) -> object:
        nonlocal load_calls
        index = load_calls
        load_calls += 1
        if index == failure_index and failure == "unavailable-output":
            raise PermissionError(unsafe)
        return original_load(path)

    monkeypatch.setattr(module, "load_baseline", lambda path: _baseline())
    monkeypatch.setattr(module, "_load_pip_audit_output", unavailable_loader)
    monkeypatch.setattr(module, "run_audit_surfaces", lambda repo: run_audit_surfaces(repo, runner=runner))

    assert main(["--repo-root", "."]) == 1
    captured = capsys.readouterr()
    _assert_safe_audit_failure_output(captured, surface=expected_surface, category=expected_category)
    assert calls == failure_index + 1
    assert all(value not in captured.out + captured.err for value in ("secret", "/private/tmp", "--index-url"))


def test_safe_audit_failure_output_assertions_reject_stdout_and_category_mutations() -> None:
    expected = "advisory audit failed: torch: execution-error\n"

    with pytest.raises(AssertionError):
        _assert_safe_audit_failure_output(
            SimpleNamespace(out="unsafe stdout\n", err=expected),
            surface="torch",
            category="execution-error",
        )
    with pytest.raises(AssertionError):
        _assert_safe_audit_failure_output(
            SimpleNamespace(out="", err="advisory audit failed: torch: invalid-json\n"),
            surface="torch",
            category="execution-error",
        )
    with pytest.raises(AssertionError):
        _assert_safe_audit_failure_output(
            SimpleNamespace(out="", err="advisory audit failed: combined-runtime: bootstrap-error\n"),
            surface="torch",
            category="bootstrap-error",
        )


def test_audit_rejects_duplicate_pip_audit_root_keys_from_serialized_output(tmp_path: Path) -> None:
    def runner(command: list[str], **kwargs: object) -> SimpleNamespace:
        output = Path(command[command.index("--output") + 1])
        output.write_text('{"dependencies": [], "dependencies": [], "fixes": []}', encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    with pytest.raises(AdvisoryBaselineError):
        run_audit_surfaces(tmp_path, runner=runner)


def test_cli_returns_nonzero_for_comparison_errors_and_zero_for_notices(monkeypatch: pytest.MonkeyPatch) -> None:
    import scripts.advisory_baseline as module

    monkeypatch.setattr(module, "load_baseline", lambda path: _baseline())
    monkeypatch.setattr(
        module,
        "run_audit_surfaces",
        lambda repo: _complete_observations(
            combined=_payload(_dependency("torch", "2.4.1", "CVE-2025-9999")),
            torch=_payload(_dependency("torch", "2.4.1")),
        ),
    )
    assert main(["--repo-root", "."]) == 1

    monkeypatch.setattr(
        module,
        "run_audit_surfaces",
        lambda repo: _complete_observations(combined=_payload(), torch=_payload()),
    )
    assert main(["--repo-root", "."]) == 0


def test_cli_has_no_accept_or_in_place_write_option(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as error:
        main(["--help"])

    assert error.value.code == 0
    help_text = capsys.readouterr().out
    assert "--accept" not in help_text
    assert "--write" not in help_text


def test_cli_diagnostics_do_not_disclose_index_credentials_or_temp_paths(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import scripts.advisory_baseline as module

    monkeypatch.setattr(
        module,
        "load_baseline",
        lambda path: (_ for _ in ()).throw(
            AdvisoryBaselineError("https://user:secret@example.test /private/tmp/advisory.json")
        ),
    )

    assert main(["--repo-root", "."]) == 1
    output = capsys.readouterr().err
    assert "secret" not in output
    assert "/private/tmp" not in output


def _copy_lock_audit_contract(destination: Path) -> Path:
    for relative in (
        "requirements",
        "requirements.txt",
        "bootstrap-requirements.txt",
        "compiler-requirements.txt",
        "nlp-model-requirements.txt",
        "torch-core-requirements.txt",
        "torch-ecosystem-requirements.txt",
        "torch-requirements.txt",
        "torch-audit-requirements.txt",
        "pyg-extension-audit-requirements.txt",
        "vulnerability-audit-requirements.txt",
        "atlas-contract-requirements.txt",
        "docs-requirements.in",
        "docs-requirements.txt",
        "Makefile",
        "Dockerfile",
        ".devcontainer",
        ".github/workflows",
        "scripts",
    ):
        source = REPO_ROOT / relative
        target = destination / relative
        if source.is_dir():
            shutil.copytree(source, target)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    for relative in verifier_module._DOCUMENTATION_CONTRACT:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO_ROOT / relative, target)
    return destination


def _refresh_lock_header_input_hash(repo: Path, relative: str, old_bytes: bytes) -> None:
    old_hash = hashlib.sha256(old_bytes).hexdigest()
    new_hash = hashlib.sha256((repo / relative).read_bytes()).hexdigest()
    assert old_hash != new_hash
    locks = tuple((repo / "requirements" / "locks").rglob("*.txt")) + (
        repo / "docs-requirements.txt",
    )
    for lock in locks:
        text = lock.read_text(encoding="utf-8")
        assert f"{relative}={old_hash}" in text
        lock.write_text(
            text.replace(f"{relative}={old_hash}", f"{relative}={new_hash}"),
            encoding="utf-8",
        )


def test_lock_audit_surfaces_are_exact_no_resolve_projections(tmp_path: Path) -> None:
    repo = _copy_lock_audit_contract(tmp_path / "repo")
    projection_root = tmp_path / "projections"

    prepared = derive_lock_audit_surfaces(repo, projection_root)

    assert [(item.surface.name, item.surface.output_name) for item in prepared] == [
        ("combined-runtime", "combined-runtime-resolver"),
        ("combined-runtime", "combined-runtime-pyg-extensions"),
        ("torch", "torch-resolver"),
        ("torch", "torch-pyg-extensions"),
        ("documentation", "documentation"),
        ("atlas-contract", "atlas-contract"),
    ]
    for item in prepared:
        lines = item.projection.read_text(encoding="utf-8").splitlines()
        assert lines == [f"{name}=={version}" for name, version in item.expected_versions]
        assert all(" @ " not in line and ";" not in line for line in lines)
    assert all(
        "setuptools==81.0.0" in item.projection.read_text()
        for item in prepared
        if item.surface.projection_kind == "main"
    )
    bootstrap = {"packaging", "pip", "setuptools", "wheel"}
    assert all(
        bootstrap <= {name for name, _ in item.expected_versions}
        for item in prepared
        if item.surface.projection_kind == "main"
    )
    audit = {
        (item.name, item.version.public)
        for item in parse_lock(repo / "requirements" / "locks" / "audit.txt")
    }
    combined = next(item for item in prepared if item.surface.output_name == "combined-runtime-resolver")
    assert audit <= set(combined.expected_versions)
    assert all(
        token not in item.projection.read_text()
        for item in prepared
        for token in ("en-core-web-sm", "pyg-lib")
    )
    assert [item for item in prepared if item.surface.output_name.endswith("pyg-extensions")]
    assert all(
        set(item.expected_versions) == {("torch-scatter", "2.1.2"), ("torch-sparse", "0.6.18")}
        for item in prepared
        if item.surface.output_name.endswith("pyg-extensions")
    )


def test_lock_audit_surfaces_record_lock_hashes_and_only_approved_non_pypi(tmp_path: Path) -> None:
    repo = _copy_lock_audit_contract(tmp_path / "repo")

    prepared = derive_lock_audit_surfaces(repo, tmp_path / "projections")
    logical = {item.surface.name: item for item in prepared if not item.surface.output_name.endswith("pyg-extensions")}

    assert {record.package for record in logical["combined-runtime"].non_pypi} == {
        "en-core-web-sm",
        "pyg-lib",
    }
    assert {record.package for record in logical["torch"].non_pypi} == {"pyg-lib"}
    assert logical["documentation"].non_pypi == ()
    assert logical["atlas-contract"].non_pypi == ()
    assert {
        name: tuple(lock.path for lock in item.lock_inputs)
        for name, item in logical.items()
    } == {
        "combined-runtime": (
            "requirements/locks/bootstrap.txt",
            "requirements/locks/audit.txt",
            "requirements/locks/darwin-arm64/root.txt",
            "requirements/locks/linux-x86_64/root.txt",
            "requirements/locks/linux-aarch64/root.txt",
        ),
        "torch": (
            "requirements/locks/bootstrap.txt",
            "requirements/locks/darwin-arm64/core.txt",
            "requirements/locks/darwin-arm64/runtime.txt",
            "requirements/locks/linux-x86_64/core.txt",
            "requirements/locks/linux-x86_64/runtime.txt",
            "requirements/locks/linux-aarch64/core.txt",
            "requirements/locks/linux-aarch64/runtime.txt",
        ),
        "documentation": ("requirements/locks/bootstrap.txt", "docs-requirements.txt"),
        "atlas-contract": (
            "requirements/locks/bootstrap.txt",
            "requirements/locks/atlas-contract.txt",
        ),
    }
    for record in logical["combined-runtime"].non_pypi + logical["torch"].non_pypi:
        assert isinstance(record, NonPyPIRecord)
        assert record.audited is False
        assert record.reason == "non-pypi"
        assert record.source.startswith("https://")
        assert record.hashes and all(len(value) == 64 for value in record.hashes)
        assert record.public_version in {"0.8.0", "3.8.0"}
    assert all(len(item.lock_inputs) >= 2 for item in logical.values())
    assert all(len(lock.sha256) == 64 for item in logical.values() for lock in item.lock_inputs)


def test_source_range_mutation_does_not_change_lock_audit_projection(tmp_path: Path) -> None:
    repo = _copy_lock_audit_contract(tmp_path / "repo")
    before = derive_lock_audit_surfaces(repo, tmp_path / "before")
    source = repo / "requirements.txt"
    original = source.read_bytes()
    source.write_bytes(original + b"# source-only change\n")
    _refresh_lock_header_input_hash(repo, "requirements.txt", original)

    after = derive_lock_audit_surfaces(repo, tmp_path / "after")

    assert [item.expected_versions for item in after] == [item.expected_versions for item in before]


@pytest.mark.parametrize(
    ("relative", "old", "new", "message"),
    (
        (
            "torch-audit-requirements.txt",
            "torch_geometric==2.8.0.post1",
            "torch_geometric==2.8.1",
            "human Torch audit projection",
        ),
        (
            "pyg-extension-audit-requirements.txt",
            "torch-sparse==0.6.18",
            "torch-sparse==0.6.19",
            "human PyG audit projection",
        ),
    ),
)
def test_human_audit_supplement_must_equal_lock_projection(
    tmp_path: Path,
    relative: str,
    old: str,
    new: str,
    message: str,
) -> None:
    repo = _copy_lock_audit_contract(tmp_path / "repo")
    source = repo / relative
    original = source.read_bytes()
    mutated = original.decode().replace(old, new, 1).encode()
    assert mutated != original
    source.write_bytes(mutated)
    _refresh_lock_header_input_hash(repo, relative, original)

    with pytest.raises(AdvisoryBaselineError, match=message):
        derive_lock_audit_surfaces(repo, tmp_path / "projections")


def test_lock_version_mutation_fails_before_any_audit_command(tmp_path: Path) -> None:
    repo = _copy_lock_audit_contract(tmp_path / "repo")
    lock = repo / "requirements" / "locks" / "bootstrap.txt"
    original = lock.read_text(encoding="utf-8")
    mutated = original.replace("setuptools==81.0.0", "setuptools==81.0.1", 1)
    assert mutated != original
    lock.write_text(mutated, encoding="utf-8")

    with pytest.raises(AdvisoryBaselineError, match="dependency locks"):
        derive_lock_audit_surfaces(repo, tmp_path / "projections")


def test_live_policy_accepts_bootstrap_setuptools_on_all_four_surfaces() -> None:
    baseline = load_baseline(REPO_ROOT / "security" / "accepted-advisories.json")
    setuptools = next(item for item in baseline.accepted_advisories if item.package == "setuptools")

    assert setuptools.accepted_version == "81.0.0"
    assert setuptools.advisory_id == "PYSEC-2026-3447"
    assert setuptools.surfaces == SURFACE_ORDER


def test_observations_retain_exact_lock_and_non_pypi_evidence(tmp_path: Path) -> None:
    _, runner = _audit_runner()

    observations = run_audit_surfaces(tmp_path, runner=runner)

    assert [item.surface for item in observations] == list(SURFACE_ORDER)
    assert all(item.lock_inputs for item in observations)
    assert {item.package for item in observations[0].non_pypi} == {
        "en-core-web-sm",
        "pyg-lib",
    }
    assert {item.package for item in observations[1].non_pypi} == {"pyg-lib"}
    assert observations[2].non_pypi == observations[3].non_pypi == ()
