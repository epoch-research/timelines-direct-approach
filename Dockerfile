FROM python:3.11-bullseye

EXPOSE 8000

WORKDIR /app

RUN mkdir static
RUN apt-get update
RUN apt-get install libopenblas-dev -y
RUN python -m pip install --upgrade pip
RUN pip install poetry

COPY pyproject.toml poetry.lock ./
RUN poetry install --only main --no-root

COPY app.py ./
COPY data data
COPY *.py ./

CMD ["poetry", "run", "uvicorn", "--host", "0.0.0.0", "--port", "8000", "app:app"]
