# supervisor.py
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import logging
import traceback
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- LLM Ayarları ---
# Yönlendirme için Flash modeli
LLM_MODEL_NAME = "gemini-1.5-flash-latest"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Yönlendirme LLM'i
router_llm = None
try:
    router_llm = ChatGoogleGenerativeAI(model=LLM_MODEL_NAME, google_api_key=GOOGLE_API_KEY, temperature=0.1)
    output_parser = StrOutputParser()
    logging.info(f"Yönlendirme LLM'i başarıyla yüklendi: {LLM_MODEL_NAME}")
except Exception as e:
    logging.error(f"Yönlendirme LLM'i ({LLM_MODEL_NAME}) yüklenirken HATA: {e}", exc_info=True)
    raise RuntimeError("Yönlendirme LLM'i yüklenemedi.") from e


# --- Yönlendirme Mantığı ---
routing_prompt_template = """
Görevin, verilen kullanıcı sorusunu DİKKATLİCE analiz edip aşağıdaki kategorilerden hangisine ait olduğunu BELİRLEMEKTİR. Önceliğin, sorunun içeriğiyle en alakalı ve spesifik kategoriyi seçmek olmalı.

Kategoriler ve Öncelikler:

1.  **Resmi Gazete**:
    *   **İçerik:** Türkiye Cumhuriyeti Resmi Gazetesi'nde yayınlanmış veya yayınlanması beklenen konular: Kanunlar, yasalar, yönetmelikler, kararnameler, tebliğler, genelgeler, cumhurbaşkanı kararları, bakanlık duyuruları, resmi ilanlar, **devlet destekleri, yardımlar, teşvikler, hibeler, krediler (bunların şartları, başvuru süreçleri, kimleri kapsadığı)**, yönetmelik değişiklikleri, **personel alımları (öğretim üyesi, memur vb. ilanlar), kadro ihdasları, atama kararları**, yargı kararları özetleri, **ihale ilanları (satış, kiralama, yapım işleri)** vb. resmi ve yasal duyurular.
    *   **Anahtar Kelimeler/İfadeler:** "Resmi Gazete", "yasa", "kanun", "yönetmelik", "tebliğ", "kararname", "ilan", "**destek**", "**yardım**", "**teşvik**", "**şartlar**", "**koşullar**", "**başvuru**", "**yayınlandı mı**", "mevzuat", "yürürlük", "madde", "fıkra", "cumhurbaşkanı kararı", "bakanlık kararı", "**kadro**", "**alım ilanı**", "**öğretim üyesi alımı**", "**ihale**", "duyuru".
    *   **ÖNCELİK:** Eğer soru, bir konuyla ilgili **resmi bir düzenlemeyi, devletin sağladığı bir imkanı, bunların şartlarını, bir personel alımını, bir ihaleyi veya resmi bir duyuruyu** soruyorsa, **KESİNLİKLE BU KATEGORİYİ SEÇ**.

2.  **Haber/Genel Bilgi**:
    *   **İçerik:** Güncel olaylar (siyaset, ekonomi (örneğin **enflasyon oranı**, borsa durumu), spor), son dakika haberleri, genel kültür bilgileri, **kişi biyografileri (örn: Albert Einstein)**, yer bilgileri (**örn: Türkiye'nin başkenti**), kavram tanımları, bilimsel/teknolojik konular, tarihsel olaylar. Resmi Gazete konusu olmayan, ansiklopedik bilgi veya güncel haber/web araması gerektiren her şey.
    *   **Anahtar Kelimeler/İfadeler:** "nedir", "kimdir", "nerede", "ne zaman", "son durum", "haberler", "biyografi", "tanımı", "açıkla", "özetle", "son dakika", "borsa", "hava durumu", "**başkanı kim**", "**enflasyon oranı**".

3.  **İlgisiz/Diğer**: Yukarıdaki iki kategoriye girmeyen sorular.

Sadece kategori adını yaz: Resmi Gazete, Haber/Genel Bilgi, veya İlgisiz/Diğer

Soru:
{question}

Kategori:
"""

routing_prompt = ChatPromptTemplate.from_template(routing_prompt_template)
router_chain = None
if router_llm:
    router_chain = routing_prompt | router_llm | output_parser
else:
    logging.error("Yönlendirme zinciri oluşturulamadı: Yönlendirme LLM'i yüklenemedi.")


def route_question(state: dict):
    """Gelen soruyu LLM kullanarak sınıflandırır ve ilgili agent'a yönlendirir."""
    logging.info("--- Supervisor (Yönlendirici) Çalışıyor ---")
    question = state.get("question")

    if not router_chain or not question:
        logging.error(f"Yönlendirme yapılamıyor. Soru: {question is not None}, Router Chain: {router_chain is not None}")
        return "fallback_agent"

    logging.info(f"Soru sınıflandırılıyor: {question}")
    try:
        time.sleep(0.5) # API Limiti
        predicted_category = router_chain.invoke({"question": question})
        predicted_category = predicted_category.strip()
        logging.info(f"LLM Kategori Tahmini: '{predicted_category}'")

        cat_lower = predicted_category.lower().replace(" ", "").replace("/", "")

        if "resmigazete" in cat_lower:
            logging.info("Yönlendirme: Resmi Gazete Agent")
            return "gazette_agent"
        elif "habergenelbilgi" in cat_lower or "haber" in cat_lower or "genelbilgi" in cat_lower:
             logging.info("Yönlendirme: Haber/Genel Bilgi Agent")
             return "news_agent"
        else:
            logging.warning(f"Tahmin edilen kategori ('{predicted_category}') anlaşılamadı, Fallback Agent'a yönlendiriliyor.")
            return "fallback_agent"
    except Exception as e:
        logging.error(f"Hata: Yönlendirme sırasında LLM çağrısı başarısız: {e}", exc_info=True)
        return "fallback_agent"