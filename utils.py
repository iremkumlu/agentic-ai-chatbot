# utils.py
import os
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
import google.generativeai as genai
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY ortam değişkeni bulunamadı.")

genai.configure(api_key=GOOGLE_API_KEY)

# --- Ayarlar ---
PDF_DATA_PATH = "data"
CHROMA_PERSIST_DIR = "./chroma_data"
EMBEDDING_MODEL_NAME = "models/embedding-001"
CHUNK_SIZE = 750 
CHUNK_OVERLAP = 150
# MultiQuery'nin temelindeki her arama için getirilecek parça sayısı
BASE_RETRIEVER_K = 8 

QUERY_GEN_LLM_MODEL = "gemini-1.5-flash-latest"
query_gen_llm = None
try:
    query_gen_llm = ChatGoogleGenerativeAI(model=QUERY_GEN_LLM_MODEL, google_api_key=GOOGLE_API_KEY, temperature=0.1)
    logging.info(f"Sorgu üretimi için LLM yüklendi: {QUERY_GEN_LLM_MODEL}")
except Exception as e:
    logging.error(f"Sorgu üretimi LLM'i ({QUERY_GEN_LLM_MODEL}) yüklenirken HATA: {e}", exc_info=True)

# MultiQuery Prompt 
QUERY_PROMPT_TEMPLATE = """
Sen bir yapay zeka dil modelisin ve görevin, kullanıcıların sorduğu soruları temel alarak,
bir vektör veritabanında arama yapmak için 3 ila 5 adet alternatif ve çeşitli arama sorgusu üretmektir.
Kullanıcının sorusuna farklı perspektiflerden bakarak, bu sorunun cevabını içerebilecek
alternatif sorular oluştur. SADECE bu soruları üret, başka hiçbir açıklama veya metin ekleme.
Her soruyu yeni bir satıra yaz.

Orijinal Soru: {question}

Oluşturulan Arama Sorguları:
"""
QUERY_PROMPT = PromptTemplate(input_variables=["question"],template=QUERY_PROMPT_TEMPLATE,)

def get_embedding_function():
    logging.info(f"Embedding modeli yükleniyor: {EMBEDDING_MODEL_NAME}")
    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL_NAME, google_api_key=GOOGLE_API_KEY)

def load_and_split_pdfs(pdf_folder_path):
    
    documents = []
    logging.info(f"PDF'ler yükleniyor: {pdf_folder_path}")
    valid_doc_count = 0
    total_pages = 0
    for filename in os.listdir(pdf_folder_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(pdf_folder_path, filename)
            try:
                loader = PyPDFLoader(file_path, extract_images=False)
                loaded_docs = loader.load()
                page_count = len(loaded_docs)
                total_pages += page_count
                current_valid_docs = [doc for doc in loaded_docs if doc.page_content and doc.page_content.strip()]
                if current_valid_docs:
                    documents.extend(current_valid_docs)
                    logging.info(f"{filename} yüklendi ({page_count} sayfa), {len(current_valid_docs)} geçerli sayfa bulundu.")
                    valid_doc_count += len(current_valid_docs)
                else:
                    logging.warning(f"{filename} yüklendi ({page_count} sayfa) ancak içerik bulunamadı veya boş.")
            except Exception as e:
                logging.error(f"Hata: {filename} yüklenirken sorun oluştu: {e}", exc_info=True)

    if not documents:
        logging.warning("Hiçbir PDF belgesi yüklenemedi veya içerikleri boş.")
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )
    split_docs = text_splitter.split_documents(documents)
    logging.info(f"Toplam {total_pages} sayfa ({valid_doc_count} geçerli sayfa), {len(split_docs)} parçaya bölündü (Chunk Size: {CHUNK_SIZE}, Overlap: {CHUNK_OVERLAP}).")
    return split_docs

def create_or_load_vector_store(persist_directory=CHROMA_PERSIST_DIR, pdf_folder=PDF_DATA_PATH, force_recreate=False):
    
    embedding_func = get_embedding_function()
    vector_store = None
    if os.path.exists(persist_directory) and not force_recreate:
        logging.info(f"Mevcut vektör veritabanı yükleniyor: {persist_directory}")
        try:
            vector_store = Chroma(persist_directory=persist_directory, embedding_function=embedding_func)
            vector_store.similarity_search("test", k=1)
            logging.info("Mevcut Vektör veritabanı başarıyla yüklendi ve doğrulandı.")
        except Exception as e:
            logging.warning(f"Mevcut vektör veritabanı yüklenirken/doğrulanırken hata oluştu: {e}. Yeniden oluşturulacak.")
            try:
                shutil.rmtree(persist_directory)
                logging.info(f"Sorunlu veritabanı klasörü silindi: {persist_directory}")
            except OSError as oe:
                 logging.error(f"Veritabanı klasörü silinirken hata: {oe}", exc_info=True)
            return create_or_load_vector_store(persist_directory, pdf_folder, force_recreate=True)
    else:
        logging.info(f"Yeni vektör veritabanı oluşturuluyor: {persist_directory}")
        split_documents = load_and_split_pdfs(pdf_folder)
        if not split_documents:
             logging.error("PDF'lerden belge okunamadığı için vektör veritabanı oluşturulamıyor.")
             return None
        logging.info(f"{len(split_documents)} belge parçası ile vektör veritabanı oluşturuluyor...")
        try:
            vector_store = Chroma.from_documents(
                documents=split_documents,
                embedding=embedding_func,
                persist_directory=persist_directory
            )
            logging.info("Vektör veritabanı başarıyla oluşturuldu ve kaydedildi.")
        except Exception as e:
            logging.error(f"Hata: ChromaDB oluşturulurken sorun oluştu: {e}", exc_info=True)
            return None
    return vector_store


def get_retriever(vector_store):
    """Verilen vektör deposundan bir MultiQueryRetriever nesnesi oluşturur."""
    if not vector_store:
        logging.error("Retriever oluşturulamadı: Vektör deposu mevcut değil.")
        return None

    # Temel retriever
    try:
        base_retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={'k': BASE_RETRIEVER_K} 
        )
        logging.info(f"Temel retriever başarıyla oluşturuldu (k={BASE_RETRIEVER_K}).")
    except Exception as e:
         logging.error(f"Temel retriever oluşturulurken hata: {e}", exc_info=True)
         return None

    # MultiQuery sadece LLM varsa oluşturulur
    if not query_gen_llm:
        logging.warning("MultiQuery LLM yüklenemediği için sadece temel retriever kullanılıyor.")
        return base_retriever

    # MultiQueryRetriever
    try:
        multi_query_retriever = MultiQueryRetriever.from_llm(
            retriever=base_retriever,
            llm=query_gen_llm,
            prompt=QUERY_PROMPT
            # include_original=True # Orijinal soruyu da aratmak için eklenebilir
        )
        logging.info("MultiQueryRetriever başarıyla oluşturuldu.")
        return multi_query_retriever
    except Exception as e:
        logging.error(f"Hata: MultiQueryRetriever oluşturulurken: {e}", exc_info=True)
        logging.warning("MultiQueryRetriever oluşturulamadı, temel retriever kullanılacak.")
        return base_retriever

