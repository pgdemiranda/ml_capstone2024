FROM python:3.10.9-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install poetry

WORKDIR /app

COPY ["pyproject.toml", "poetry.lock", "./"]
RUN poetry config virtualenvs.create false
RUN poetry install --no-root
RUN poetry cache clear --all pypi

COPY ./final_project/final_model.pkl ./
COPY ./final_project/predict.py ./

EXPOSE 8000

ENTRYPOINT ["poetry", "run", "uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "8000"]