# Agentic AI Chatbot 

Akıllı Yönlendirmeli Agentic AI Chatbot Sistemi

Bu proje, kullanıcıların sorduğu soruları içeriklerine göre analiz ederek en uygun bilgi kaynağına yönlendiren ve anlamlı cevaplar üreten modüler bir yapay zeka sohbet botudur. Supervisor-Agent mimarisi ile geliştirilmiş olan bu sistem, hem belge tabanlı sorguları (Resmi Gazete gibi) hem de genel bilgi aramalarını (Wikipedia, haberler) destekler.

##  Özellikler

- Supervisor-Agent mimarisi ile modüler yapı
- LangGraph kullanarak akıllı yönlendirme iş akışı
- RAG (Retrieval-Augmented Generation) destekli belge arama
- Wikipedia & Web (Tavily) üzerinden dış bilgi alma
- Fallback Agent ile güvenli yanıt sistemi
- Streamlit tabanlı kullanıcı dostu arayüz
- Docker ile kolay kurulum ve dağıtım

##  Kullanım Mimarisi

**Supervisor (Denetleyici):**  
Sorunun hangi bilgi türüne ait olduğunu belirler ve ilgili alt agent'a yönlendirir.

**Alt Agent'lar:**
-  **Resmi Gazete Agent:** Resmi belgeleri içeren özel vektör veritabanında (ChromaDB) arama yapar.
-  **Haber/Genel Bilgi Agent:** Wikipedia ve Tavily üzerinden güncel veya ansiklopedik bilgi sağlar.
-  **Fallback Agent:** Sistem dışı sorulara güvenli ve nazik yanıtlar üretir.

##  Kullanılan Teknolojiler

- **LangChain & LangGraph:** Agent yapıları ve iş akışı
- **LLM:** Google Gemini Flash modeli
- **ChromaDB:** Vektör tabanlı belge arama
- **Wikipedia & Tavily:** Dış bilgi kaynakları
- **Streamlit:** Arayüz
- **Docker & Docker Compose:** Kolay kurulum ve taşıma

##  Kurulum

```bash
git clone https://github.com/kullanici-adin/agentic-ai-chatbot.git
cd agentic-ai-chatbot
cp .env.example .env # API anahtarlarını buraya ekleyin
docker-compose up --build

