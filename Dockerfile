# Python 3.11 slim imajını kullanalım
FROM python:3.11-slim


# Sistem paketlerini kur
RUN apt-get update && apt-get install -y \
    build-essential \
    libffi-dev \
    gcc \
    python3-dev \
    rustc \
    cargo \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Çalışma dizini oluştur ve ayarla
WORKDIR /app

# Gereksinim dosyasını kopyala
COPY requirements.txt .

# Pip'i güncelle ve gerekli paketleri kur
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Uygulama dosyalarını kopyala
COPY . .

# Streamlit portu için expose et
EXPOSE 8501

# Uygulama çalıştırma komutu
CMD ["streamlit", "run", "app.py"]
