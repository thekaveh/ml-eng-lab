FROM quay.io/jupyter/datascience-notebook:python-3.11

WORKDIR /usr/src/app

COPY . .

RUN pip install --no-cache-dir --upgrade pip \
  && pip install --no-cache-dir --upgrade setuptools \
  && pip install --no-cache-dir -r torch-requirements.txt \
  && pip install --no-cache-dir -r requirements.txt

# Two Tier-A NLP tasks (text_classification-agnews-spacy-mlp,
# sentiment_classification-vader-mlp) need a spaCy model + an NLTK
# lexicon that pip install doesn't pull. Mirror what
# .github/workflows/ci.yml's tier-a-papermill job does so the local
# `docker build` + `docker run` path can execute every Tier-A notebook
# without a "model not found" surprise.
RUN python -m spacy download en_core_web_sm \
  && python -c "import nltk; nltk.download('vader_lexicon', quiet=True)"
