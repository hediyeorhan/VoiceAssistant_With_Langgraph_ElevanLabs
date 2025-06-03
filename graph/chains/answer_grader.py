from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableSequence
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv


import os

load_dotenv()

model = ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL"))

class GradeAnswer(BaseModel):

    binary_score: bool = Field(
        description="Cevap soruyu yanıtlıyor, 'evet' veya 'hayır'"
    )

structured_llm_grader = model.with_structured_output(GradeAnswer)

system = """Sen, bir cevabın soruyu ne kadar yanıtladığını / çözüme kavuşturduğunu değerlendiren bir değerlendiricisin. 
Cevaba 'evet' veya 'hayır' şeklinde ikili bir puan ver. 'Evet' demek, cevabın soruyu çözdüğü anlamına gelir."""
answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ]
)

answer_grader: RunnableSequence = answer_prompt | structured_llm_grader