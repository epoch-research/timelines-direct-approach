FROM python:3.11-bullseye

EXPOSE 8000

WORKDIR /app

RUN mkdir static
RUN apt-get update
RUN python -m pip install --upgrade pip
RUN pip install pipenv

COPY Pipfile Pipfile.lock ./
RUN pipenv install --dev --system --deploy

COPY blog_post.py ./
COPY certs certs
COPY data data
COPY *.py ./

CMD ["pipenv", "run", "uvicorn", "--host", "0.0.0.0", "--port", "8000", "--ssl-keyfile", "certs/privkey.pem", "--ssl-certfile", "certs/fullchain.pem", "blog_post:app"]
