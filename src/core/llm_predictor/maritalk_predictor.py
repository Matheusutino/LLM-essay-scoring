import os
import maritalk
from src.core.llm_predictor.llm_predictor import LLMPredictor
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class MariTalkPredictor(LLMPredictor):
    """Predictor using the MariTalk API."""
    
    def __init__(self, model: str):
        self.api_key = os.getenv("MARITALK_API_KEY")  # Load the API key from .env
        self.model = model  # Model name passed as a parameter
        self.client = maritalk.MariTalk(key=self.api_key, model=self.model)  # Using the MariTalk API
        

    def predict(self, system_prompt: str, user_prompt: str, temperature: float, max_output_tokens: int) -> str:
        # Call the MariTalk API to generate the response
        response = self.client.generate(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            max_tokens=max_output_tokens
        )
        return response["answer"].strip() 