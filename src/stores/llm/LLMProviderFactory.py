from .LLMEnums import LLMEnums
from .providers import OpenAIProvider, CoHereProvider, HuggingFaceProvider

class LLMProviderFactory:
    def __init__(self, config: dict):
        self.config = config

    def create(self, provider: str):
        if provider == LLMEnums.OPENAI.value:
            # Clean API URL if it's empty
            api_url = None
            if hasattr(self.config, 'OPENAI_API_URL') and self.config.OPENAI_API_URL:
                api_url = self.config.OPENAI_API_URL
                
            # Create provider with explicit parameters only
            return OpenAIProvider(
                api_key=self.config.OPENAI_API_KEY,
                api_url=api_url,
                default_input_max_characters=self.config.INPUT_DAFAULT_MAX_CHARACTERS,
                default_generation_max_output_tokens=self.config.GENERATION_DAFAULT_MAX_TOKENS,
                default_generation_temperature=self.config.GENERATION_DAFAULT_TEMPERATURE
            )

        if provider == LLMEnums.COHERE.value:
            return CoHereProvider(
                api_key=self.config.COHERE_API_KEY,
                default_input_max_characters=self.config.INPUT_DAFAULT_MAX_CHARACTERS,
                default_generation_max_output_tokens=self.config.GENERATION_DAFAULT_MAX_TOKENS,
                default_generation_temperature=self.config.GENERATION_DAFAULT_TEMPERATURE
            )
            
        if provider == LLMEnums.HUGGINGFACE.value:
            # Clean API URL if it's empty
            api_url = None
            if hasattr(self.config, 'HUGGINGFACE_API_URL') and self.config.HUGGINGFACE_API_URL:
                api_url = self.config.HUGGINGFACE_API_URL
                
            return HuggingFaceProvider(
                api_key=self.config.HUGGINGFACE_API_KEY,
                api_url=api_url,
                default_input_max_characters=self.config.INPUT_DAFAULT_MAX_CHARACTERS,
                default_generation_max_output_tokens=self.config.GENERATION_DAFAULT_MAX_TOKENS,
                default_generation_temperature=self.config.GENERATION_DAFAULT_TEMPERATURE
            )

        return None
