from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv

import os

load_dotenv()

model = ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL"))


class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Belgeler soru ile ilgili, 'evet' veya 'hayır'"
    )


structured_llm_grader = model.with_structured_output(GradeDocuments)

system = """Sen, bir değerlendiricisin ve geri getirilen belgelerin bir kullanıcı sorusuna ve konuşma geçmişine ne kadar uygun olduğunu değerlendiriyorsun.

Değerlendirme Bağlamı:
- Konuşma Geçmişi: {history}

Değerlendirme Kriterleri:
1. Belge ve soru arasındaki anahtar kelime / anlamsal uyum
2. Konuşma bağlamıyla tutarlılık
3. Bilgi yeniliği (daha yeni / son zamanlarda tartışılan bilgilerin tercih edilmesi)

İkili bir 'evet' veya 'hayır' puanı ver."""
    
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader