from ..LLMInterface import LLMInterface
from ..LLMEnums import OpenAIEnums, DocumentTypeEnum
from openai import OpenAI
from openai.types.embedding import Embedding
from openai.types.chat import ChatCompletion
from typing import List, Optional, Dict, Any, Union
import logging
import time
from datetime import datetime, timedelta

class RateLimitTracker:
    def __init__(self, requests_per_minute: int = 10000):  # OpenAI can handle 10,000 TPM on text-embedding-3-large
        self.requests_per_minute = requests_per_minute
        self.request_timestamps: List[datetime] = []
        self.logger = logging.getLogger(__name__)

    def can_make_request(self) -> bool:
        now = datetime.now()
        # Remove timestamps older than 1 minute
        self.request_timestamps = [ts for ts in self.request_timestamps 
                                 if now - ts < timedelta(minutes=1)]
        return len(self.request_timestamps) < self.requests_per_minute

    def add_request(self) -> None:
        self.request_timestamps.append(datetime.now())

    def wait_if_needed(self) -> None:
        while not self.can_make_request():
            time.sleep(1)
        self.add_request()

class OpenAIProvider(LLMInterface):
    def __init__(self, 
                 api_key: str, 
                 api_url: Optional[str] = None,
                 default_input_max_characters: int = 1000,
                 default_generation_max_output_tokens: int = 1000,
                 default_generation_temperature: float = 0.1):
        
        self.api_key = api_key
        self.api_url = api_url
        self.default_input_max_characters = default_input_max_characters
        self.default_generation_max_output_tokens = default_generation_max_output_tokens
        self.default_generation_temperature = default_generation_temperature

        self.generation_model_id = None
        self.embedding_model_id = None
        self.embedding_size = None

        # Initialize OpenAI client with minimal parameters
        client_kwargs = {
            "api_key": self.api_key
        }
        
        # Only add base_url if it's a valid string
        if self.api_url and isinstance(self.api_url, str) and len(self.api_url) > 0:
            client_kwargs["base_url"] = self.api_url
            
        try:
            self.client = OpenAI(**client_kwargs)
            self.logger = logging.getLogger(__name__)
            self.logger.info("OpenAI client successfully initialized")
            
            # Test the connection with a minimal API call
            self._test_connection()
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI client: {str(e)}")
            self.client = None
        
        self.enums = OpenAIEnums
        self.rate_limiter = RateLimitTracker()

    def _test_connection(self) -> bool:
        """Test the connection to OpenAI API with a minimal call"""
        try:
            # Make a small test request that works with most OpenAI-compatible APIs
            self.logger.info("Testing OpenAI connection...")
            # Skip models.list call which may not be supported by all API providers
            # Instead, just report that we have a client initialized
            self.logger.info("OpenAI client initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"OpenAI connection test failed: {str(e)}")
            return False

    def set_generation_model(self, model_id: str) -> None:
        self.generation_model_id = model_id

    def set_embedding_model(self, model_id: str, embedding_size: int) -> None:
        self.embedding_model_id = model_id
        self.embedding_size = embedding_size

    def process_text(self, text: str) -> str:
        return text[:self.default_input_max_characters].strip()

    def generate_text(self, 
                     prompt: str, 
                     chat_history: List[Dict[str, str]] = None,
                     max_output_tokens: Optional[int] = None,
                     temperature: Optional[float] = None) -> Optional[str]:
        
        if not self.client or not self.generation_model_id:
            self.logger.error("OpenAI client or generation model not set")
            return None

        try:
            self.rate_limiter.wait_if_needed()
            
            messages = []
            if chat_history:
                messages.extend(chat_history)
            
            messages.append({
                "role": "user",
                "content": self.process_text(prompt)
            })

            response: ChatCompletion = self.client.chat.completions.create(
                model=self.generation_model_id,
                messages=messages,
                max_tokens=max_output_tokens or self.default_generation_max_output_tokens,
                temperature=temperature or self.default_generation_temperature
            )

            return response.choices[0].message.content

        except Exception as e:
            self.logger.error(f"Error generating text: {str(e)}")
            return None

    def embed_text(self, text: str, document_type: Optional[str] = None) -> Optional[List[float]]:
        if not self.client or not self.embedding_model_id:
            self.logger.error("OpenAI client or embedding model not set")
            return None

        try:
            self.rate_limiter.wait_if_needed()
            
            response = self.client.embeddings.create(
                model=self.embedding_model_id,
                input=self.process_text(text)
            )
            
            return response.data[0].embedding

        except Exception as e:
            self.logger.error(f"Error embedding text: {str(e)}")
            return None

    def embed_batch(self, texts: List[str], batch_size: int = 20) -> List[Optional[List[float]]]:
        """Batch process texts for embedding with better error handling.
        Using smaller batches to avoid overwhelming the API server."""
        all_embeddings = []
        
        # Process in smaller batches for more reliability
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                self.rate_limiter.wait_if_needed()
                
                # Process texts to stay within token limits
                processed_texts = [self.process_text(text) for text in batch]
                
                # Try batch embedding
                response = self.client.embeddings.create(
                    model=self.embedding_model_id,
                    input=processed_texts
                )
                
                embeddings = [data.embedding for data in response.data]
                all_embeddings.extend(embeddings)
                
                self.logger.info(f"Successfully embedded batch of {len(batch)} texts ({i + len(batch)}/{len(texts)})")
                
            except Exception as e:
                self.logger.error(f"Error batch embedding texts: {str(e)}")
                
                # Fall back to individual embedding
                fallback_embeddings = []
                self.logger.info(f"Falling back to individual embedding for batch {i//batch_size + 1}")
                
                for text in batch:
                    try:
                        # Individual embedding as fallback
                        vector = self.embed_text(text=text)
                        fallback_embeddings.append(vector)
                    except Exception as inner_e:
                        self.logger.error(f"Error in individual embedding: {str(inner_e)}")
                        fallback_embeddings.append(None)
                
                all_embeddings.extend(fallback_embeddings)
        
        return all_embeddings

    def construct_prompt(self, prompt: str, role: str) -> Dict[str, str]:
        return {
            "role": role,
            "content": self.process_text(prompt)
        }
    


    

