FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY main.py .

# Railway uses PORT env var
ENV PORT=8000
EXPOSE 8000

CMD ["python", "main.py"]
