# agents.py
import os
import re
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import Tool
import traceback
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()

# --- LLM Ayarları ---
LLM_MODEL_NAME = "gemini-1.5-flash-latest" # Flash modelini kullanıyoruz
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY") # Tavily anahtarını yükle

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY ortam değişkeni bulunamadı.")
if not TAVILY_API_KEY:
    logging.warning("TAVILY_API_KEY ortam değişkeni bulunamadı. Web arama devre dışı.")

# --- Ana LLM ---
llm = None
try:
    
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL_NAME, google_api_key=GOOGLE_API_KEY, temperature=0.3)
    output_parser = StrOutputParser()
    logging.info(f"Ana LLM başarıyla yüklendi: {LLM_MODEL_NAME}")
except Exception as e:
     logging.error(f"Ana LLM yüklenirken HATA: {e}", exc_info=True)
     raise RuntimeError("Ana LLM yüklenemedi, uygulama başlatılamıyor.") from e

# --- Araçlar (Wikipedia ve Tavily) ---
wikipedia_langchain_tool = None
try:
    api_wrapper = WikipediaAPIWrapper(lang="tr", top_k_results=2, doc_content_chars_max=4000)
    wikipedia_tool = WikipediaQueryRun(api_wrapper=api_wrapper, name="wikipedia_search")
    wikipedia_langchain_tool = Tool(
        name="wikipedia_search",
        func=wikipedia_tool.run,
        description="Ansiklopedik bilgi, tarihi olaylar, kişiler(örn:Albert Einstein), yerler, kavram tanımları gibi genel ve oturmuş bilgiler için kullanılır."
    )
    logging.info("Wikipedia aracı başarıyla yüklendi.")
except Exception as e:
    logging.error(f"Hata: Wikipedia aracı yüklenirken: {e}", exc_info=True)

# Tavily aracını tekrar ekleyelim
tavily_langchain_tool = None
if TAVILY_API_KEY:
    try:
        # Güncel bilgiler için 5 sonuç yeterli olabilir
        tavily_search = TavilySearchResults(max_results=5, name="web_search")
        tavily_langchain_tool = Tool(
            name="web_search",
            func=tavily_search.run, # Liste dönebilir
            description="En güncel olaylar(örn: geçen haftaki enflasyon oranı), anlık haberler, son dakika gelişmeleri veya Wikipedia'da bulunamayan spesifik bilgiler için internette arama yapar."
        )
        logging.info("Tavily Search aracı başarıyla yüklendi.")
    except Exception as e:
        logging.error(f"Hata: Tavily Search aracı yüklenirken: {e}", exc_info=True)
else:
     logging.info("Tavily API Anahtarı bulunmadığı için Tavily Search aracı yüklenmedi.")


# --- Agent Fonksiyonları ---

# 1. Resmi Gazete Agent 
def create_rag_chain(retriever):
    """Verilen retriever ile bir RAG zinciri oluşturur."""
    # Flash modeli için RAG prompt'u
    rag_prompt_template = """
    GÖREV: Sana verilen Resmi Gazete metin parçalarını ('Bağlam Parçaları') kullanarak kullanıcının sorusunu ('Soru') yanıtla.
    ADIMLAR:
    1.  Tüm 'Bağlam Parçaları'nı oku.
    2.  'Soru'nun cevabını bu parçalarda ara.
    3.  Cevabı bulursan: Sadece parçalardaki bilgiyi kullanarak, soruyu doğrudan ve net bir şekilde Türkçe yanıtla. Varsa madde/tarih gibi detayları ekle.
    4.  Cevabı bulamazsan: Sadece şu cümleyi yaz: "Sağlanan Resmi Gazete belgelerinde bu konuyla ilgili spesifik bir bilgiye rastlanmadı."
    5.  ASLA dışarıdan bilgi ekleme veya tahmin yapma.

    BAĞLAM PARÇALARI:
    {context}

    SORU: {input}

    YANIT (Türkçe):
    """
    rag_prompt = ChatPromptTemplate.from_template(rag_prompt_template)
    if not llm:
         raise ValueError("RAG zinciri oluşturulamadı: Ana LLM yüklenmemiş.")
    rag_chain = create_stuff_documents_chain(llm, rag_prompt)
    return rag_chain

def run_gazette_agent(state: dict):
    """Resmi Gazete ile ilgili soruları RAG kullanarak yanıtlar."""
    logging.info("--- Resmi Gazete Agent Çalışıyor ---")
    question = state.get("question")
    retriever = state.get("retriever") # MultiQueryRetriever bekleniyor
    rag_chain = state.get("rag_chain")

    if not question or not retriever or not rag_chain:
        logging.error("Gazette Agent Hatası: Gerekli RAG bileşenleri eksik.")
        return {"answer": "Üzgünüm, RAG sistemi yapılandırmasında bir sorun var.", "source": "Gazette Agent (Yapılandırma Hatası)"}

    # 1. Belgeleri Al
    retrieved_docs = []
    try:
        logging.info(f"MultiQueryRetriever'a gönderilen soru: {question}")
        retrieved_docs = retriever.invoke(question)
        logging.info(f"MultiQueryRetriever {len(retrieved_docs)} belge buldu.")
        if not retrieved_docs:
            logging.warning("MultiQueryRetriever soruyla ilgili HİÇ belge bulamadı.")
            return {"answer": "Resmi Gazete arşivinde bu konuyla ilgili bir belge bulunamadı.", "source": "Gazette Agent (Belge Bulunamadı - Retriever)"}
        logging.info(f"İlk bulunan belge (metadata): {retrieved_docs[0].metadata}")

    except Exception as e:
        logging.error(f"HATA: MultiQueryRetriever sorgusu sırasında: {e}", exc_info=True)
        return {"answer": "Üzgünüm, Resmi Gazete belgelerinde arama yaparken teknik bir sorun oluştu (MultiQuery).", "source": "Gazette Agent (Retrieval Hatası)"}

    # 2. Cevabı Üret
    answer = ""
    source = "Gazette Agent (Bilinmeyen)"
    if not llm:
        logging.error("Gazette Agent: Ana LLM yüklenmemiş.")
        return {"answer": "Üzgünüm, cevap üretme servisinde bir sorun var.", "source": "Gazette Agent (LLM Hatası)"}

    try:
        logging.info(f"RAG zinciri LLM'i {len(retrieved_docs)} belge ile çağırıyor...")
        time.sleep(1) # API Limiti
        context_char_count = sum(len(doc.page_content) for doc in retrieved_docs)
        logging.info(f"RAG LLM'e gönderilen context karakter sayısı: {context_char_count}")

        # Context limiti kontrolü ve kırpma (Flash için önemli)
        TOKEN_LIMIT_APPROX_FLASH = 20000
        if context_char_count > TOKEN_LIMIT_APPROX_FLASH:
             logging.warning(f"Context boyutu ({context_char_count} karakter) Flash LLM limiti ({TOKEN_LIMIT_APPROX_FLASH}) aşabilir! Belgeler kırpılıyor...")
             trimmed_docs = []
             current_chars = 0
             for doc in retrieved_docs:
                  doc_len = len(doc.page_content)
                  if current_chars + doc_len <= TOKEN_LIMIT_APPROX_FLASH:
                       trimmed_docs.append(doc)
                       current_chars += doc_len
                  else:
                       # Kalan karakter kadarını almayı dene (opsiyonel, basitlik için atlayabiliriz)
                       # remaining_space = TOKEN_LIMIT_APPROX_FLASH - current_chars
                       # if remaining_space > 100: # Çok küçükse eklemeye değmez
                       #      trimmed_doc_part = Document(page_content=doc.page_content[:remaining_space], metadata=doc.metadata)
                       #      trimmed_docs.append(trimmed_doc_part)
                       #      logging.warning(f"Son belge kısmen eklendi.")
                       logging.warning(f"Context limiti nedeniyle {len(retrieved_docs) - len(trimmed_docs)} belge atlandı.")
                       break
             retrieved_docs = trimmed_docs
             if not retrieved_docs:
                  logging.error("Context limiti nedeniyle tüm belgeler atlandı!")
                  return {"answer": "Üzgünüm, bulunan ilgili belgeler işlenemeyecek kadar uzun.", "source": "Gazette Agent (Context Limit Hatası)"}

        answer = rag_chain.invoke({"input": question, "context": retrieved_docs})
        logging.info(f"Gazette Agent LLM Ham Cevabı (kısaltılmış): {answer[:500]}...")

        if "rastlanmadı" in answer or len(answer.strip()) < 15:
            source = "Gazette Agent (Bilgi Bulunamadı - LLM)"
        else:
            source = "Resmi Gazete Agent (RAG)"

    except Exception as e:
        if "payload size exceeds the limit" in str(e) or "context length" in str(e).lower():
             logging.error(f"HATA: RAG LLM çağrısı - Context Window aşımı: {e}", exc_info=False)
             answer = f"Üzgünüm, bu soru için çok fazla ilgili belge bulundu ve hepsini aynı anda işleyemedim. (Context Limiti)"
             source = "Gazette Agent (Context Limit Hatası)"
        else:
            logging.error(f"HATA: RAG LLM çağrısı (Genel): {e}", exc_info=True)
            answer = "Üzgünüm, bulunan belgelerden cevabı oluştururken teknik bir sorunla karşılaştım."
            source = "Gazette Agent (LLM Hatası)"

    return {"answer": answer, "source": source}


# 2. Güncel Haber / Genel Bilgi Agent (WIKIPEDIA + TAVILY )
def run_news_agent(state: dict):
    """Genel bilgi sorularını önce Wikipedia, başarısız olursa Tavily kullanarak yanıtlar."""
    logging.info("--- Haber/Genel Bilgi Agent Çalışıyor ---")
    question = state.get("question")
    if not question:
         logging.warning("News Agent: Soru alınamadı.")
         return {"answer": "Üzgünüm, sorunuzu alamadım.", "source": "News Agent (Hata)"}

    final_answer = ""
    final_source = "Bilinmiyor"
    attempted_wiki = False
    attempted_tavily = False

    if not llm:
        logging.error("News Agent: Ana LLM yüklenmemiş.")
        return {"answer": "Üzgünüm, cevap üretme servisinde bir sorun var.", "source": "News Agent (LLM Hatası)"}

    # --- Adım 1: Wikipedia'yı Dene (Ansiklopedik bilgi için) ---
    if wikipedia_langchain_tool:
        logging.info(f"Wikipedia'da (TR) '{question}' aranıyor...")
        attempted_wiki = True
        try:
            wiki_response_raw = wikipedia_langchain_tool.run(question)
            logging.info(f"--- Wikipedia Ham Yanıtı (Başlangıç) ---")
            logging.info(wiki_response_raw[:400] + "..." if len(wiki_response_raw) > 400 else wiki_response_raw)
            logging.info(f"--------------------------------------")

            # Temel hata/boşluk/belirsizlik kontrolü
            if (wiki_response_raw and len(wiki_response_raw.strip()) > 50 and
                "Did not find results" not in wiki_response_raw and
                "No good Wikipedia Search Result" not in wiki_response_raw and
                "may refer to" not in wiki_response_raw and
                "Page id" not in wiki_response_raw):

                content_to_process = wiki_response_raw.split("Summary:", 1)[-1].strip() if "Summary:" in wiki_response_raw else wiki_response_raw
                content_to_process = re.sub(r"^Page:.*?\n", "", content_to_process).strip()

                # Alaka Kontrolü
                relevance_check_prompt = PromptTemplate.from_template(
                    "Aşağıdaki metin, '{soru}' sorusuyla alakalı mı? Sadece EVET veya HAYIR de.\n\nMetin:\n{metin}"
                )
                relevance_chain = relevance_check_prompt | llm | StrOutputParser()
                logging.info("LLM ile Wikipedia metninin alaka düzeyi kontrol ediliyor...")
                time.sleep(0.5)
                relevance_decision = relevance_chain.invoke({"soru": question, "metin": content_to_process[:1000]})
                logging.info(f"Alaka Kontrolü Sonucu: {relevance_decision}")

                if "evet" in relevance_decision.lower():
                    # Cevap Üretme  prompt
                    wiki_processing_prompt = PromptTemplate.from_template("""
                    GÖREV: Sağlanan Wikipedia Metnini kullanarak aşağıdaki Soruyu yanıtla.
                    TALİMATLAR:
                    1. Metni oku.
                    2. Sorunun cevabını metinde ara.
                    3. Cevabı bulursan, bilgiyi doğrudan ve kısa bir şekilde Türkçe olarak yaz.
                    4. Cevabı bulamazsan, SADECE "Sağlanan Wikipedia metninde bu sorunun cevabı bulunmuyor." yaz.
                    5. "Metne göre" gibi ifadeler kullanma.

                    SAĞLANAN WIKIPEDIA METNİ:
                    {wikipedia_icerigi}

                    SORU: {soru}

                    YANITIN (Türkçe):
                    """)
                    wiki_chain = wiki_processing_prompt | llm | output_parser
                    logging.info("Alakalı bulunan Wikipedia yanıtı LLM ile işleniyor...")
                    time.sleep(1)
                    processed_wiki_answer = wiki_chain.invoke({"soru": question, "wikipedia_icerigi": content_to_process})
                    logging.info(f"Wikipedia İşleme Sonucu (kısaltılmış): {processed_wiki_answer[:500]}...")

                    # Cevap kontrolü
                    failure_phrases_final = ["bulunmuyor", "yoktur", "rastlanmadı"]
                    if not any(phrase in processed_wiki_answer.lower() for phrase in failure_phrases_final) and len(processed_wiki_answer.strip()) > 10 :
                         final_answer = processed_wiki_answer
                         final_source = "Wikipedia (LLM ile İşlendi)"
                    else:
                         logging.info("LLM, Wikipedia içeriğiyle cevap bulamadı.")
                else:
                    logging.warning("LLM, bulunan Wikipedia metnini soruyla alakasız buldu.")
            else:
                logging.info("Wikipedia'dan yeterli/anlamlı sonuç alınamadı.")
        except Exception as wiki_e:
            logging.error(f"HATA: Wikipedia aracı çağrılırken/işlenirken: {wiki_e}", exc_info=True)
    else:
        logging.info("Wikipedia aracı mevcut değil.")

    # --- Adım 2: Wikipedia Başarısız Olduysa Web Aramayı Dene (Tavily varsa - GÜNCEL bilgiler için) ---
    # Özellikle soru "geçen hafta", "bugün", "son durum", "açıklandı mı" gibi ifadeler içeriyorsa Tavily daha uygun olabilir.
    # Şimdilik basit fallback mantığıyla devam edelim:
    if not final_answer and tavily_langchain_tool:
        logging.info(f"Wikipedia yetersiz/başarısız, Web'de (Tavily) '{question}' aranıyor...")
        attempted_tavily = True
        try:
            search_results_list = tavily_langchain_tool.invoke(question)
            logging.info(f"\n--- TAM Tavily Ham Yanıtı (Liste) ---")
            logging.info(search_results_list)
            logging.info(f"-----------------------------------\n")

            if isinstance(search_results_list, list) and search_results_list:
                 formatted_results = ""
                 for i, result in enumerate(search_results_list[:5]): # İlk 5 sonucu alalım
                     title = result.get('title', '')
                     content = result.get('content', '')
                     # url = result.get('url', '') # URL'i LLM'e vermeyebiliriz
                     formatted_results += f"Başlık: {title}\nÖzet: {content}\n\n"

                 # Tavily için Basit Prompt
                 search_processing_prompt = PromptTemplate.from_template(
                     "Aşağıdaki web arama sonuçlarını kullanarak '{soru}' sorusunu Türkçe yanıtla. Sonuçlardan en alakalı bilgiyi özetle.\n\n{search_results}\n\nYanıt:"
                 )
                 search_chain = search_processing_prompt | llm | output_parser
                 logging.info("Tavily arama sonuçları LLM ile işleniyor...")
                 time.sleep(1)
                 processed_tavily_answer = search_chain.invoke({"soru": question, "search_results": formatted_results.strip()})
                 logging.info(f"Tavily İşleme Sonucu (kısaltılmış): {processed_tavily_answer[:500]}...")

                 # Sonuç Kontrolü
                 if processed_tavily_answer and len(processed_tavily_answer.strip()) > 15:
                     very_negative_phrases = ["üzgünüm", "bulamadım", "bilgi yok", "cevap yok", "net bir cevap bulunamadı"]
                     if not any(phrase in processed_tavily_answer.lower() for phrase in very_negative_phrases):
                          final_answer = processed_tavily_answer
                          final_source = "Web Search (Tavily ile İşlendi)"
                     else:
                          logging.info("LLM, Tavily sonuçlarından olumsuz bir yanıt üretti.")
                 else:
                      logging.info("LLM, Tavily sonuçlarından anlamlı/yeterli bir özet çıkaramadı.")
            else:
                 logging.info("Tavily'den liste formatında anlamlı sonuç alınamadı.")
        except Exception as tavily_e:
            logging.error(f"HATA: Tavily aracı çağrılırken/işlenirken: {tavily_e}", exc_info=True)

    elif not final_answer:
         logging.info("Tavily aracı mevcut değil veya kullanılmadı.")

    # --- Adım 3: Hiçbir Yerden Cevap Bulunamadıysa ---
    if not final_answer:
        logging.warning("Hem Wikipedia hem de Web Search'ten (denendiyse) başarılı bir cevap alınamadı.")
        tried_sources = []
        if wikipedia_langchain_tool and attempted_wiki: tried_sources.append("Wikipedia")
        if tavily_langchain_tool and attempted_tavily: tried_sources.append("Web Search")
        sources_str = " ve ".join(tried_sources) if tried_sources else "mevcut"

        final_answer = f"Üzgünüm, '{question}' hakkında {sources_str} kaynaklarımda yaptığım aramalarda net veya yeterli bir bilgi bulamadım."
        final_source = "Haber/Genel Bilgi Agent (Bilgi Bulunamadı)"

    return {"answer": final_answer, "source": final_source}


# 3. Fallback Agent
def run_fallback_agent(state: dict, custom_message: str = None):
    """İlgisiz veya cevaplanamayan sorular için standart bir yanıt verir."""
    logging.info("--- Fallback Agent Çalışıyor ---")
    question = state.get("question","")
    if custom_message:
         answer = custom_message
    else:
        answer = f"Üzgünüm, '{question}' sorusuna şu anki bilgi ve yeteneklerimle yanıt veremiyorum. Farklı bir şekilde sormayı deneyebilirsiniz."
    return {"answer": answer, "source": "Fallback Agent"}