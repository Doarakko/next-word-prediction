FROM python:3.7

ENV PYTHONUNBUFFERED=1
COPY ./app/pyproject.toml ./
RUN pip install poetry
RUN poetry install
RUN wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin