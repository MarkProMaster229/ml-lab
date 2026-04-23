FROM python:3.11-slim
WORKDIR /app


RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/MarkProMaster229/ml-lab.git .

RUN pip install --no-cache-dir -r requirements.txt




CMD ["python", "InferenceModelApp/main.py"]
