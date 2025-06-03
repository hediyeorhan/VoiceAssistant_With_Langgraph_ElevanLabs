from typing import List, Dict

class ConversationHistory:
    def __init__(self, max_length=10):
        self.history: List[Dict[str, str]] = []
        self.max_length = max_length
    
    def add_interaction(self, user_message: str, ai_response: str):
        self.history.append({"user": user_message, "ai": ai_response})
        if len(self.history) > self.max_length:
            self.history.pop(0)
    
    def get_history(self) -> List[Dict[str, str]]:
        return self.history.copy()