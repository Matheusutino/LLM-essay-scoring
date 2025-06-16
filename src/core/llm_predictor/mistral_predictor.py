import os
from mistralai import Mistral
from src.core.llm_predictor.llm_predictor import LLMPredictor
from dotenv import load_dotenv, find_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Load environment variables
load_dotenv(find_dotenv())

class MistralPredictor(LLMPredictor):
    """Predictor using Mistral's API."""

    def __init__(self, model: str):
        self.api_key = os.getenv("MISTRAL_API_KEY")
        self.client = Mistral(api_key=self.api_key)
        self.model = model

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=5),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    def predict(self, system_prompt: str, user_prompt: str, temperature: float, max_output_tokens: int) -> str:
        response = self.client.chat.complete(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            max_tokens=max_output_tokens
        )
        return response.choices[0].message.content.strip()
