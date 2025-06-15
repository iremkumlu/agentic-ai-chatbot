# graph.py
from typing import TypedDict, Sequence, Optional
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage

# Agent fonksiyonlarını import et
from agents import run_gazette_agent, run_news_agent, run_fallback_agent
# Supervisor (router) fonksiyonunu import et
from supervisor import route_question 

# --- Grafiğin Durum (State) Tanımı ---
class AgentState(TypedDict):
    question: str
    answer: Optional[str]
    source: Optional[str]

# --- Yönlendirici Node Fonksiyonu ---
# Bu fonksiyon, "router" adlı node çalıştığında çağrılır.
# Asıl yönlendirme işini yapmaz, sadece bir geçiş noktasıdır.
# State'i değiştirmesi gerekmiyorsa boş bir dict döndürebilir.
def router_node_placeholder(state: AgentState) -> dict:
    """Yönlendirme kararından hemen önce çalışan node. State'i değiştirmez."""
    print("--- Router Node Çalıştırıldı (Yönlendirme Kararı Öncesi) ---")
    # Bu node'un kendisi state'i değiştirmiyor, karar conditional_edge'de veriliyor.
    # LangGraph bir dict beklediği için boş bir dict döndürüyoruz.
    return {}

# --- LangGraph İş Akışı ---
def create_agent_graph(retriever, rag_chain):
    """LangGraph iş akışını tanımlar ve derler."""

    if not retriever:
        print("Uyarı: Retriever nesnesi olmadan graf oluşturuluyor. Resmi Gazete Agent çalışmayabilir.")
    if not rag_chain:
        print("Uyarı: RAG zinciri olmadan graf oluşturuluyor. Resmi Gazete Agent çalışmayabilir.")

    workflow = StateGraph(AgentState)

    # Nodeları tanımla
    # "router" node'u için yeni placeholder fonksiyonu kullan
    workflow.add_node("router", router_node_placeholder)

    # Gazette agent nodu: state'i alır VE dışarıdan gelen retriever/rag_chain'i kullanır
    def gazette_node_wrapper(state):
        input_dict = {**state, "retriever": retriever, "rag_chain": rag_chain}
        # run_gazette_agent bir dict döndürmeli (answer, source içeren)
        return run_gazette_agent(input_dict)
    workflow.add_node("gazette_agent", gazette_node_wrapper)

    # News agent nodu state'i alır ve bir dict döndürmeli
    workflow.add_node("news_agent", run_news_agent)

    # Fallback agent nodu state'i alır ve bir dict döndürmeli
    workflow.add_node("fallback_agent", run_fallback_agent)


    # Başlangıç noktasını belirle
    workflow.set_entry_point("router")

    # Koşullu kenarları tanımla
    # "router" node'undan SONRA, route_question fonksiyonunu çağırarak karar ver.
    # route_question'ın döndürdüğü string'e göre ilgili agent node'una git.
    workflow.add_conditional_edges(
        "router",          # Hangi node'dan sonra karar verilecek: "router"
        route_question,   # Kararı hangi fonksiyon verecek (state'i alır, string döndürür)
        {                 # Dönen string'e göre hangi node'a gidilecek eşleşmesi
            "gazette_agent": "gazette_agent",
            "news_agent": "news_agent",
            "fallback_agent": "fallback_agent",
        }
    )

    # Agent nodelarından sonra iş akışını bitir (END)
    workflow.add_edge("gazette_agent", END)
    workflow.add_edge("news_agent", END)
    workflow.add_edge("fallback_agent", END)

    # Grafiği derle
    agent_graph = workflow.compile()
    print("LangGraph başarıyla derlendi.")
    return agent_graph