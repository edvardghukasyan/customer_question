import os
from langchain_openai import ChatOpenAI
from src.utils.decorators import singleton
from configs.config import OPENAI_API_KEY, TEMPERATURE, MODEL_NAME

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


@singleton
class OpenAIClient:
    
    def __init__(self):
        self.model_name = MODEL_NAME
        self.temperature = TEMPERATURE
        self.llm = ChatOpenAI(model_name=self.model_name, temperature=self.temperature)
        