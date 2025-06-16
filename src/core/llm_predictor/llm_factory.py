from src.core.llm_predictor.llm_predictor import LLMPredictor
from src.core.llm_predictor.openai_predictor import OpenAIPredictor
from src.core.llm_predictor.gemini_predictor import GeminiPredictor
from src.core.llm_predictor.nvidia_predictor import NVIDIAPredictor
from src.core.llm_predictor.maritalk_predictor import MariTalkPredictor
from src.core.llm_predictor.mistral_predictor import MistralPredictor
from src.core.llm_predictor.groq_predictor import GroqPredictor
from src.core.llm_predictor.chutes_predictor import ChutesPredictor
from src.core.llm_predictor.ollama_predictor import OllamaPredictor


class LLMFactory:
    """Factory to create predictors dynamically based on the .env configuration."""

    @staticmethod
    def get_predictor(llm_provider: str, model_name: str) -> LLMPredictor:
        if llm_provider == "openai":
            return OpenAIPredictor(model_name)
        elif llm_provider == "gemini":
            return GeminiPredictor(model_name)
        elif llm_provider == "nvidia":
            return NVIDIAPredictor(model_name)
        elif llm_provider == "mistral":
            return MistralPredictor(model_name)
        elif llm_provider == "groq":
            return GroqPredictor(model_name)
        elif llm_provider == "chutes":
            return ChutesPredictor(model_name)
        elif llm_provider == "maritalk":
            return MariTalkPredictor(model_name)
        elif llm_provider == "ollama":
            return OllamaPredictor(model_name)
        else:
            raise ValueError(f"Provider '{llm_provider}' is not supported.")