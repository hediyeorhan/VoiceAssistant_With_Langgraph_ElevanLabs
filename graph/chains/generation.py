from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv

import os


load_dotenv()

model = ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL"))

prompt = hub.pull("rlm/rag-prompt")

generation_chain = prompt | model | StrOutputParser()