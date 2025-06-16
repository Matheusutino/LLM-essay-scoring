import ollama
from src.core.llm_predictor.llm_predictor import LLMPredictor

class OllamaPredictor(LLMPredictor):
    """Prediction model implementation for Ollama."""

    def __init__(self, model: str):
        """
        Initializes the OllamaPredictor with a model name and device.

        Args:
            model (str): The name of the Ollama model.
        """
        self.model = model
        ollama.pull(self.model)

    def predict(self, system_prompt: str, user_prompt: str, temperature: float, max_output_tokens: int, context_window: int = 10000):
        """
        Generates text based on the input prompt using the Ollama model.

        Args:
            messages (str): The formatted prompt for Ollama, including system and user sections.
            max_tokens (int): Maximum number of tokens to generate.
            temperature (float): Sampling temperature. Lower values make the output more deterministic.
            **kwargs: Additional arguments for prediction (e.g., seed).
        
        Returns:
            str: The generated text.
        """
        try:
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            response = ollama.chat(
                model=self.model,
                messages=messages,
                options={
                    'num_predict': max_output_tokens,
                    'temperature': temperature,
                    'num_ctx': context_window
                }
            )
            return response['message']['content'].strip()
        except Exception as e:
            raise RuntimeError(f"Error generating text: {e}")

