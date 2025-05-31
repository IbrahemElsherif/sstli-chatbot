from ..LLMInterface import LLMInterface
from ..LLMEnums import CoHereEnums, DocumentTypeEnum
import cohere
import logging
import time
from cohere.errors import TooManyRequestsError
from typing import List, Dict, Optional, Any, Union

class CoHereProvider(LLMInterface):

    def __init__(self, api_key: str,
                       default_input_max_characters: int=1000,
                       default_generation_max_output_tokens: int=1000,
                       default_generation_temperature: float=0.1):
        
        self.api_key = api_key

        self.default_input_max_characters = default_input_max_characters
        self.default_generation_max_output_tokens = default_generation_max_output_tokens
        self.default_generation_temperature = default_generation_temperature

        self.generation_model_id = None

        self.embedding_model_id = None
        self.embedding_size = None

        self.client = cohere.Client(api_key=self.api_key)

        self.enums = CoHereEnums
        self.logger = logging.getLogger(__name__)

    def set_generation_model(self, model_id: str) -> None:
        self.generation_model_id = model_id

    def set_embedding_model(self, model_id: str, embedding_size: int) -> None:
        self.embedding_model_id = model_id
        self.embedding_size = embedding_size

    def process_text(self, text: str) -> str:
        return text[:self.default_input_max_characters].strip()

    def generate_text(self, prompt: str, 
                      chat_history: Optional[List[Dict[str, str]]] = None,
                      max_output_tokens: Optional[int] = None,
                      temperature: Optional[float] = None) -> Optional[str]:

        if not self.client:
            self.logger.error("CoHere client was not set")
            return None

        if not self.generation_model_id:
            self.logger.error("Generation model for CoHere was not set")
            return None
        
        if chat_history is None:
            chat_history = []
            
        max_output_tokens = max_output_tokens if max_output_tokens else self.default_generation_max_output_tokens
        temperature = temperature if temperature else self.default_generation_temperature

        try:
            response = self.client.chat(
                model=self.generation_model_id,
                chat_history=chat_history,
                message=self.process_text(prompt),
                temperature=temperature,
                max_tokens=max_output_tokens
            )

            if not response or not response.text:
                self.logger.error("Error while generating text with CoHere")
                return None
            
            return response.text
            
        except Exception as e:
            self.logger.error(f"Error generating text: {str(e)}")
            return None
    
    def embed_text(self, text: str, document_type: Optional[str] = None) -> Optional[List[float]]:
        max_retries = 3
        retry_delay = 2  # seconds

        if not document_type:
            document_type = DocumentTypeEnum.SEARCH_DOCUMENT.value

        for attempt in range(max_retries):
            try:
                response = self.client.embed(
                    texts=[self.process_text(text)],
                    model=self.embedding_model_id,
                    input_type=document_type
                )
                return response.embeddings[0]
            except TooManyRequestsError:
                if attempt < max_retries - 1:  # don't sleep on the last attempt
                    time.sleep(retry_delay)
                    retry_delay *= 2  # exponential backoff
                else:
                    self.logger.error("Rate limit exceeded after retries")
                    return None
            except Exception as e:
                self.logger.error(f"Error while embedding text: {e}")
                return None
    
    def embed_batch(self, texts: List[str], batch_size: int = 25) -> List[Optional[List[float]]]:
        """
        Batch embedding with Cohere's API, with rate limit handling
        """
        max_retries = 3
        initial_retry_delay = 2  # seconds
        all_embeddings = []
        document_type = DocumentTypeEnum.SEARCH_DOCUMENT.value
        
        # Process in smaller batches to avoid rate limits
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            processed_batch = [self.process_text(text) for text in batch]
            
            retry_delay = initial_retry_delay
            for attempt in range(max_retries):
                try:
                    response = self.client.embed(
                        texts=processed_batch,
                        model=self.embedding_model_id,
                        input_type=document_type
                    )
                    all_embeddings.extend(response.embeddings)
                    break  # Success, exit retry loop
                except TooManyRequestsError:
                    if attempt < max_retries - 1:
                        self.logger.warning(f"Rate limit hit, retrying in {retry_delay}s")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # exponential backoff
                    else:
                        self.logger.error("Rate limit exceeded after retries")
                        # Add None values for the failed batch
                        all_embeddings.extend([None] * len(batch))
                except Exception as e:
                    self.logger.error(f"Error batch embedding texts: {str(e)}")
                    # Add None values for the failed batch
                    all_embeddings.extend([None] * len(batch))
                    break  # Exit retry loop for other errors
            
            # Add delay between batches to avoid rate limits
            if i + batch_size < len(texts):
                time.sleep(1)
        
        return all_embeddings
    
    def construct_prompt(self, prompt: str, role: str) -> Dict[str, str]:
        return {
            "role": role,
            "text": self.process_text(prompt)
        }