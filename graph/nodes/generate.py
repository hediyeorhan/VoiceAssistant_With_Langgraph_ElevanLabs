from typing import Any, Dict
from graph.chains.generation import generation_chain
from graph.state import GraphState

def generate(state: GraphState) -> Dict[str, Any]:
    print("---GENERATE ANSWER WITH HISTORY---")
    
    # State'ten gerekli verileri al
    question = state["question"]
    documents = state["documents"]
    history = state.get("history", [])  # History'yi al (yoksa boş liste)

    # History'yi prompt formatına dönüştür
    chat_history = "\n".join(
        [f"User: {h['user']}\nAssistant: {h['ai']}" 
         for h in history[:]]) if history else "No history available"
    
    #print("DEBUG HISTORY FORMAT:", chat_history)
    
    # Gelişmiş prompt template (history'yi içerecek şekilde)
    prompt_context = f"""
    Conversation History:
    {chat_history}
    
    Retrieved Documents:
    {documents}
    
    Current Question:
    {question}
    """
    
    # Generation chain'i çağır
    generation = generation_chain.invoke({
        "context": documents,
        "question": question,
        "history": chat_history  # Chain'in history'yi kullanabilmesi için
    })
    
    return {
        "documents": documents,
        "question": question,
        "generation": generation,
        # History'yi de state'te korumaya devam ediyoruz
        "history": history  
    }