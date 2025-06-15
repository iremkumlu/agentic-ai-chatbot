import streamlit as st
from dotenv import load_dotenv
import os

# Proje bileşenlerini import et
from utils import create_or_load_vector_store, get_retriever
from agents import create_rag_chain # RAG zinciri oluşturma fonksiyonu
from graph import create_agent_graph

# --- Sayfa Ayarları ve Başlangıç ---
st.set_page_config(page_title="Agentic AI Chatbot", layout="wide")
st.title("📄📰 Agentic AI Chatbot: Resmi Gazete & Haberler")
st.caption("Supervisor-Agent mimarisi ile sorularınızı yanıtlar.")

# --- API Anahtarı ve İlk Yüklemeler ---
# @st.cache_resource gibi yapılar pahalı işlemleri önbelleğe almak için kullanılır.
# API anahtarını yükle
@st.cache_resource
def load_environment():
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("GOOGLE_API_KEY bulunamadı. Lütfen .env dosyasını kontrol edin.")
        st.stop()
    # print("API Anahtarı yüklendi.") # Geliştirme sırasında kontrol için
    return api_key

# Vektör veritabanını yükle/oluştur ve retriever'ı al
@st.cache_resource
def initialize_vector_store_and_retriever():
    with st.spinner("Vektör veritabanı hazırlanıyor (Bu işlem ilk çalıştırmada biraz sürebilir)..."):
        vector_store = create_or_load_vector_store()
        if vector_store is None:
             st.error("Vektör veritabanı oluşturulamadı veya yüklenemedi. 'data' klasörünü ve PDF dosyalarını kontrol edin.")
             st.stop() # Vektör deposu yoksa devam etme
        retriever = get_retriever(vector_store)
        if retriever is None:
             st.warning("Retriever oluşturulamadı, Resmi Gazete sorguları çalışmayabilir.")
        st.success("Vektör veritabanı ve retriever hazır!")
    return retriever

# RAG zincirini oluştur
@st.cache_resource
def initialize_rag_chain(_retriever): # Retriever'ı argüman olarak alması cache'lemeyi tetikler
    if _retriever:
        rag_chain = create_rag_chain(_retriever)
        # print("RAG Zinciri oluşturuldu.") # Geliştirme sırasında kontrol için
        return rag_chain
    return None

# Agent grafiğini oluştur
@st.cache_resource
def initialize_graph(_retriever, _rag_chain): # Diğer bileşenleri argüman olarak alması cache'lemeyi tetikler
    if _retriever and _rag_chain:
        graph = create_agent_graph(_retriever, _rag_chain)
        # print("Agent Grafiği oluşturuldu.") # Geliştirme sırasında kontrol için
        return graph
    elif _retriever: # Sadece retriever varsa (rag chain oluşturulamadıysa?)
        # Belki sadece haber agent'ı çalışacak bir graf oluşturulabilir veya uyarı verilebilir.
        # Şimdilik eksik bilgi ile graf oluşturmayı deneyelim (bazı nodelar hata verebilir)
         st.warning("RAG zinciri oluşturulamadığı için Resmi Gazete Agent'ı düzgün çalışmayabilir.")
         graph = create_agent_graph(_retriever, None) # Eksik rag_chain ile graf oluştur
         return graph
    else:
         st.error("Graf oluşturmak için gerekli Retriever ve/veya RAG Zinciri eksik.")
         st.stop()


# --- Ana Akış ---
api_key = load_environment()
retriever = initialize_vector_store_and_retriever()
rag_chain = initialize_rag_chain(retriever) # Retriever'ı argüman olarak geçir
app_graph = initialize_graph(retriever, rag_chain) # Bileşenleri argüman olarak geçir


# --- Kullanıcı Arayüzü ---
st.sidebar.header("Bilgi")
st.sidebar.info(
    "Bu chatbot, sorduğunuz sorunun konusuna göre (Resmi Gazete veya Genel Haber/Bilgi) "
    "farklı bilgi kaynaklarını kullanır. \n\n"
    "**Örnek Sorular:**\n"
    "- Son çıkan torba yasada emekliler için ne var?\n"
    "- 12.03.2022 tarihli Resmi Gazete'yi özetler misin?\n"
    "- Türkiye'nin başkenti neresidir?\n"
    "- Yapay zeka hakkında bilgi verir misin?"
)
st.sidebar.warning("Resmi Gazete verileri sadece `data` klasöründeki PDF'lerden alınmıştır.")

# Sohbet geçmişini session state'de tutalım
if "messages" not in st.session_state:
    st.session_state.messages = []

# Geçmiş mesajları göster
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Kullanıcıdan yeni soru al
if prompt := st.chat_input("Sorunuzu buraya yazın..."):
    # Kullanıcının mesajını ekle
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Chatbot'un cevabını al
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        with st.spinner("Düşünüyorum..."):
            try:
                # Agent grafiğini çalıştır
                initial_state = {"question": prompt}
                final_state = app_graph.invoke(initial_state)

                # Cevabı ve kaynağı al
                answer = final_state.get("answer", "Üzgünüm, bir cevap alamadım.")
                source = final_state.get("source", "Bilinmeyen Kaynak")

                # Cevabı ekrana yazdır
                full_response = f"{answer}\n\n*[Kaynak: {source}]*"
                message_placeholder.markdown(full_response)

            except Exception as e:
                st.error(f"Bir hata oluştu: {e}")
                full_response = "Üzgünüm, isteğinizi işlerken bir sorunla karşılaştım."
                message_placeholder.markdown(full_response)

        # Asistanın cevabını geçmişe ekle
        st.session_state.messages.append({"role": "assistant", "content": full_response})