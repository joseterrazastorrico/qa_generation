FROM python:3.10-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r /app/requirements.txt

EXPOSE 8080

CMD ["streamlit", "run", "./app.py", "--server.port=8080"]
