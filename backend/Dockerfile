FROM python:3.10-slim

WORKDIR /app

# system deps
RUN apt-get update && apt-get install -y build-essential git && rm -rf /var/lib/apt/lists/*

COPY ./app /app
COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r /app/requirements.txt

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
