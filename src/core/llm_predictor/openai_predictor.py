import os
from openai import OpenAI
from src.core.llm_predictor.llm_predictor import LLMPredictor 
from dotenv import load_dotenv, find_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Load environment variables
load_dotenv(find_dotenv())

class OpenAIPredictor(LLMPredictor):
    """Predictor using OpenAI's API."""
    
    def __init__(self, model: str):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key = self.api_key)
        self.model = model  # Model name passed as a parameter

    @retry(
        stop=stop_after_attempt(3),                     # tenta no máximo 3 vezes
        wait=wait_exponential(multiplier=1, min=2, max=5),  # espera exponencial entre tentativas
        retry=retry_if_exception_type(Exception),       # você pode customizar para tipos específicos de erro
        reraise=True                                     # relança o erro se todas as tentativas falharem
    )
    def predict(self, system_prompt: str, user_prompt: str, temperature: float, max_output_tokens: int) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            max_tokens=max_output_tokens
        )
        return response.choices[0].message.content.strip()