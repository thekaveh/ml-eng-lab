FROM quay.io/jupyter/datascience-notebook:python-3.11

WORKDIR /usr/src/app

COPY . .

RUN pip install --no-cache-dir --upgrade pip \
  && pip install --no-cache-dir --upgrade setuptools \
  && pip install --no-cache-dir -r torch-requirements.txt \
  && pip install --no-cache-dir -r requirements.txt
