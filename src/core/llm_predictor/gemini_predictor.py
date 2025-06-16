import os
from google import genai
from google.genai import types
from src.core.llm_predictor.llm_predictor import LLMPredictor   
from dotenv import load_dotenv, find_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Load environment variables
load_dotenv(find_dotenv())

class GeminiPredictor(LLMPredictor):
    """Predictor using Google's Gemini API."""
    
    def __init__(self, model: str):
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.client = genai.Client(api_key=self.api_key)
        self.model = model


    @retry(
        stop=stop_after_attempt(3),                     # tenta no máximo 3 vezes
        wait=wait_exponential(multiplier=1, min=2, max=5),  # espera exponencial entre tentativas
        retry=retry_if_exception_type(Exception),       # você pode customizar para tipos específicos de erro
        reraise=True                                     # relança o erro se todas as tentativas falharem
    )
    def predict(self, system_prompt: str, user_prompt: str, temperature: float, max_output_tokens: int) -> str:
        response = self.client.models.generate_content(
            model=self.model,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                max_output_tokens=max_output_tokens,
                temperature=temperature
            )
        )
        return response.text.strip()
