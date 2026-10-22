FROM python:3.11-slim

# ffmpeg и libsndfile — чтение/запись MP3 и WAV; dejavu — шрифт с кириллицей для обложек
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg libsndfile1 fonts-dejavu-core \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Некоторые хостинги (Hugging Face Spaces) запускают контейнер не от root — даём права на папки с данными
RUN mkdir -p /data && chmod -R 777 /data /app

# База пользователей хранится в томе /data и переживает перезапуски и обновления
ENV NBS_DATA_DIR=/data
EXPOSE 8501

# Хостинги вроде Railway и Render передают порт в переменной PORT; на своём сервере берётся 8501
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import os, urllib.request; urllib.request.urlopen('http://localhost:%s/_stcore/health' % os.environ.get('PORT', '8501'))" || exit 1

CMD streamlit run app.py \
    --server.port=${PORT:-8501} --server.address=0.0.0.0 \
    --server.headless=true --server.maxUploadSize=200 \
    --browser.gatherUsageStats=false
