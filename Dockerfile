FROM python:3.9

RUN apt-get update && apt-get install build-essential swig python-dev -y && \
	pip install --no-cache-dir --upgrade pip

RUN pip install s3fs
RUN pip install --pre autogluon
RUN pip install fairlearn
