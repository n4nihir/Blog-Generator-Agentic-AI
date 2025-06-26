from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

class OpenAILLM:
    def __init__(self):
        load_dotenv()

    def get_llm(self):
        try:
            os.environ["OPENAI_API_KEY"] = self.openai_qpi_key = os.getenv("OPENAI_API_KEY")
            llm = ChatOpenAI(
                api_key=self.openai_qpi_key,
                model_name="gpt-4o"
            )
            return llm
        except Exception as e:
            print(f"Error initializing OpenAILLM: {e}")


