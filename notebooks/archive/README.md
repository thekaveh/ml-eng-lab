# Archived experiments

This directory holds ML experiments preserved as historical artifacts. **Not maintained.** Code may reference packages or library versions that no longer install cleanly. Notebook outputs reflect a point-in-time training run from 2023 and are not reproducible against the current environment.

If you want to revive any of these, treat it as starting from scratch with the original notebook as a reference.

The CodeXGLUE cross-language notebooks now resolve paths relative to their archived folders
and skip their evaluation/preview cells when the historical `repos/CodeXGLUE` checkout,
checkpoint, or `model/test_0.*` files are absent. Restore those artifacts or rerun the old
training cells to reproduce the original comparisons.

---

## 1. codexglue_summarization/

22 sub-experiments around the [CodeXGLUE](https://github.com/microsoft/CodeXGLUE) code-summarization benchmark. Two families:

### 1.1. Cross-language experiments

Each notebook trains a model on one language's data and evaluates on another, exploring transfer behavior across programming languages.

| Folder | Source → Target |
|---|---|
| codexglue-summarization-cross-java-on-go | Java → Go |
| codexglue-summarization-cross-java-on-jacascript | Java → JavaScript *(note: folder name has typo `jacascript`)* |
| codexglue-summarization-cross-java-on-php | Java → PHP |
| codexglue-summarization-cross-java-on-python | Java → Python |
| codexglue-summarization-cross-java-on-ruby | Java → Ruby |
| codexglue-summarization-cross-python-on-go | Python → Go |
| codexglue-summarization-cross-python-on-java | Python → Java |
| codexglue-summarization-cross-python-on-javascript | Python → JavaScript |
| codexglue-summarization-cross-python-on-php | Python → PHP |
| codexglue-summarization-cross-python-on-ruby | Python → Ruby |

### 1.2. RoBERTa-based model experiments

Each notebook fine-tunes a pre-trained model (CodeBERT / GraphCodeBERT / RoBERTa variants) on one language for code-summarization.

| Folder | Model · Language |
|---|---|
| codexglue-summarization-roberta-codebert-java | codebert · Java |
| codexglue-summarization-roberta-codebert-python | codebert · Python |
| codexglue-summarization-roberta-codeberta-java | codeberta · Java |
| codexglue-summarization-roberta-codeberta-python | codeberta · Python |
| codexglue-summarization-roberta-codebertmlm-java | codebertmlm · Java |
| codexglue-summarization-roberta-codebertmlm-python | codebertmlm · Python |
| codexglue-summarization-roberta-graphcodebert-java | graphcodebert · Java |
| codexglue-summarization-roberta-graphcodebert-python | graphcodebert · Python |
| codexglue-summarization-roberta-roberta-java | roberta · Java |
| codexglue-summarization-roberta-roberta-python | roberta · Python |
| codexglue-summarization-roberta-unixcoder-java | unixcoder · Java |
| codexglue-summarization-roberta-unixcoder-python | unixcoder · Python |

### 1.3. Other

| Folder | Notes |
|---|---|
| src | Shared source code / utilities used across the sub-experiments above |

---

## 2. Why archived

The codexglue experiments depend on the older `transformers` API and have not been touched since Aug 2023. The active task index in the root README (`README.md` §4.1) now spans 21 task folders covering vision, tabular, graph, NLP, generative, PEFT, MoE, JEPA, DPO and more. Codexglue stays here as a reference snapshot.
