FROM python:3.7

ENV PYTHONUNBUFFERED=1
COPY ./app/pyproject.toml ./
RUN pip install -U pip
RUN pip install poetry
RUN poetry config virtualenvs.create false
RUN poetry install
RUN rm -rf ~/.cache
RUN wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin