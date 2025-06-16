import os
from openai import OpenAI
from src.core.llm_predictor.llm_predictor import LLMPredictor 
from dotenv import load_dotenv, find_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Load environment variables
load_dotenv(find_dotenv())

class NVIDIAPredictor(LLMPredictor):
    """Predictor using NVIDIA's API (OpenAI-compatible)."""

    def __init__(self, model: str):
        self.api_key = os.getenv("NVIDIA_API_KEY")  # use a variável correta do .env
        self.base_url = "https://integrate.api.nvidia.com/v1"
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.model = model

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=5),
        retry=retry_if_exception_type(Exception),
        reraise=True
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
