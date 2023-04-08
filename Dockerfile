FROM jupyter/datascience-notebook:latest

WORKDIR /usr/src/app

COPY . .

RUN pip install --no-cache-dir --upgrade pip \
  && pip install --no-cache-dir -r requirements.txt

RUN python -m nltk.downloader all
RUN python -m spacy download en_core_web_sm