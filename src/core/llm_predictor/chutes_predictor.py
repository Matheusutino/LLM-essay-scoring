import os
import aiohttp
import asyncio
import json
from typing import List, Dict, Optional
from dotenv import load_dotenv, find_dotenv
from src.core.llm_predictor.llm_predictor import LLMPredictor   
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Load environment variables
load_dotenv(find_dotenv())

class ChutesPredictor(LLMPredictor):
    """Prediction model implementation for Chutes API."""

    def __init__(self, model_name: str, api_key: Optional[str] = None, base_url: str = "https://llm.chutes.ai/v1"):
        """
        Initializes the ChutesPredictor with an API key and optional base_url.
        """
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("CHUTES_API_TOKEN")
        self.base_url = base_url
        self.endpoint = f"{self.base_url}/chat/completions"

    @retry(
        stop=stop_after_attempt(3),                     # tenta no máximo 3 vezes
        wait=wait_exponential(multiplier=1, min=2, max=5),  # espera exponencial entre tentativas
        retry=retry_if_exception_type(Exception),       # você pode customizar para tipos específicos de erro
        reraise=True                                     # relança o erro se todas as tentativas falharem
    )
    async def predict_async(self, system_prompt: str, user_prompt: str, temperature: float, max_output_tokens: int, stream: bool = False) -> str:
        """
        Predicts using Chutes API (streaming ou normal).

        Args:
            messages (List[Dict[str, str]]): Lista de mensagens no formato chat.
            max_tokens (int): Máximo de tokens na resposta.
            temperature (float): Temperatura da amostragem.
            stream (bool): Se True, usa streaming da resposta.

        Returns:
            str: Resultado da predição.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
        ]

        body = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_output_tokens,
            "temperature": temperature,
            "stream": stream
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(self.endpoint, headers=headers, json=body) as response:
                if response.status != 200:
                    raise Exception(f"Chutes API error: HTTP {response.status}")

                if stream:
                    full_text = ""
                    async for line in response.content:
                        line = line.decode("utf-8").strip()
                        if line.startswith("data: "):
                            data = line[6:]
                            if data == "[DONE]":
                                break
                            try:
                                json_data = json.loads(data)
                                delta = json_data["choices"][0]["delta"]
                                if "content" in delta:
                                    full_text += delta["content"]
                            except Exception as e:
                                raise Exception(f"Error parsing chunk: {e}")
                    return full_text
                else:
                    result = await response.json()
                    return result["choices"][0]["message"]["content"]
                
    def predict(self, system_prompt: str, user_prompt: str, temperature: float, max_output_tokens: int):
        try:
            loop = asyncio.get_running_loop()
            return loop.run_until_complete(self.predict_async(system_prompt, user_prompt, temperature, max_output_tokens))
        except RuntimeError:
            return asyncio.run(self.predict_async(system_prompt, user_prompt, temperature, max_output_tokens))


