from abc import ABC, abstractmethod

class LLMPredictor(ABC):
    """Abstract class defining the interface for LLM predictors."""

    @abstractmethod
    def predict(self, system_prompt: str, user_prompt: str, temperature: float, max_output_tokens: int) -> str:
        """Receives prompts and parameters, returns the model's response."""
        pass