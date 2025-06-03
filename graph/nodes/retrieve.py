from typing import Any, Dict
from graph.state import GraphState
from ingestion import retriever

def retrieve(state: GraphState) -> Dict[str, Any]:
    print("---RETRIEVE---")
    question = state["question"]
    history = state.get("history", [])
    
    # Tarihçeyi formatla (sadece bilgi amaçlı, retriever'a göndermiyoruz)
    chat_history = "\n".join(
        [f"User: {h['user']}\nAssistant: {h['ai']}" 
         for h in history[-10:]]) if history else "No history available"
    
    # RETRIEVER'A SADECE STRING SORUYU GÖNDER
    documents = retriever.invoke(question)  # Dikkat: Artık direkt string
        
    return {
        "documents": documents,
        "question": question,
        "history": history  # Orijinal history'yi koru
    }