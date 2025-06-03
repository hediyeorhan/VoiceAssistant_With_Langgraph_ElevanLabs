from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Any, Dict
from dotenv import load_dotenv
from graph.state import GraphState
from langchain_core.messages import HumanMessage, SystemMessage
import os

load_dotenv()

def direct_response(state: GraphState) -> Dict[str, Any]:
    model = ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL"))
    question = state["question"]
    history = state.get("history", [])

    # Tarihçeyi formatla
    chat_history = "\n".join(
        [f"User: {h['user']}\nAssistant: {h['ai']}" 
         for h in history[-3:]]) if history else "No history available"

    # Doğru input formatını oluştur
    messages = [
        SystemMessage(content="Sen yardımcı bir asistansın. Yanıt verirken konuşma geçmişini dikkate al. Cevapları türkçe olarak üret."),
        *[HumanMessage(content=f"User: {h['user']}\nAssistant: {h['ai']}") for h in history[-3:]],
        HumanMessage(content=question)
    ]

    # Modeli çağır
    response = model.invoke(messages)
    
    return {
        "generation": response.content,
        "question": question,
        "history": history
    }