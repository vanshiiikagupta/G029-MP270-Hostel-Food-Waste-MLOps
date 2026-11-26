FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY models/champion_model.joblib ./models/champion_model.joblib
COPY src/mp270/__init__.py ./src/mp270/__init__.py
COPY src/mp270/preprocessing.py ./src/mp270/preprocessing.py

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]