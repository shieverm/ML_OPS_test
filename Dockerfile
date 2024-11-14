FROM python:3.9-slim

WORKDIR /app

COPY pyproject.toml /app/
RUN pip install poetry
RUN poetry install

COPY src/ /app/src/
COPY data/ /app/data/

CMD ["poetry", "run", "uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]