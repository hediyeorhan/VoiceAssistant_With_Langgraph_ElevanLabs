from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableSequence
from langchain_google_genai import ChatGoogleGenerativeAI
from graph.state import GraphState
from dotenv import load_dotenv

import os

load_dotenv()

model = ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL"))


class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: bool = Field(
        description="Cevap gerçeklere dayanıyor, 'evet' veya 'hayır'"
    )


structured_llm_grader = model.with_structured_output(GradeHallucinations)

system = """Sen, bir değerlendiricisin ve bir LLM üretiminin gerçeklerle ve konuşma geçmişiyle ne kadar örtüştüğünü değerlendiriyorsun.
Konuşma Geçmişi:
{history}

Değerlendirme Kuralı: Cevap, hem gerçeklerle hem de geçmişle uyumlu olmalıdır.
İkili bir puan ver 'evet' veya 'hayır'."""
hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)

hallucination_grader: RunnableSequence = hallucination_prompt | structured_llm_grader