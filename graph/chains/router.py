from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv

import os


load_dotenv()

class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["direct_response",  "vectorstore", "register", "websearch"] = Field(
        ...,
        description="Kullanıcının sorduğu bir soruya göre, ya soruyu kendin yanıtla, ya web aramasına yönlendir, ya da bir vectorstore deposuna ilet. Cevaplar türkçe olacak!",
    )


model = ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL"))
structured_llm_router = model.with_structured_output(RouteQuery)

system = f"""Sen, bir kullanıcı sorusunu ya kendi bilgine dayanarak ya da mevcut bağlamdan faydalanarak ya da bir vektör veri deposuna veya web aramasına yönlendirerek en iyi şekilde cevaplayan bir uzmansın.
Öncelikle, soruları kendi bilgine dayanarak cevaplamaya çalış. Eğer bu mümkün değilse, vektör veri deposunu kullan.
Vektör veri deposu, {os.getenv("TOPIC")} ile ilgili belgeleri içermektedir.Bu konularla ilgili sorular için vektör veri deposunu kullan. 
Diğer tüm konular için web aramasını kullan. Kullanıcılara tatlı ve sıcakkanlı bir şekilde cevap ver. Cevaplar türkçe olsun!
"""

route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

question_router = route_prompt | structured_llm_router    