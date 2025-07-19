from ..LLMInterface import LLMInterface
# قد تحتاج لتعديل هذا بناءً على مكان تعريف GeminiEnums و DocumentTypeEnum
from ..LLMEnums import GeminiEnums, DocumentTypeEnum 
import google.generativeai as genai
from typing import List, Optional, Dict, Any, Union
import logging
import time
from datetime import datetime, timedelta

class RateLimitTracker:
    def __init__(self, requests_per_minute: int = 60):  # Typical free tier rate for Gemini, adjust as needed
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
            time.sleep(1) # Wait for 1 second and re-check
        self.add_request()

class GeminiProvider(LLMInterface):
    def __init__(self, 
                 api_key: str, 
                 # api_url is not typically used directly with google-generativeai for public APIs
                 # default_input_max_characters for Gemini depends on model, e.g., gemini-pro has 30k input tokens
                 default_input_max_characters: int = 30000, 
                 default_generation_max_output_tokens: int = 1000,
                 default_generation_temperature: float = 0.9): # Gemini often performs better with higher temperature initially
        
        self.api_key = api_key
        # self.api_url = api_url # Not directly used for standard Gemini API
        self.default_input_max_characters = default_input_max_characters
        self.default_generation_max_output_tokens = default_generation_max_output_tokens
        self.default_generation_temperature = default_generation_temperature

        self.generation_model_id: Optional[str] = None
        self.embedding_model_id: Optional[str] = None
        self.embedding_size: Optional[int] = None

        self.logger = logging.getLogger(__name__)
        
        try:
            # Configure the Google Generative AI library with the API key
            genai.configure(api_key=self.api_key)
            self.logger.info("Gemini API configured successfully")
            
            # Test the connection indirectly by listing models
            self._test_connection()
        except Exception as e:
            self.logger.error(f"Failed to configure Gemini API or test connection: {str(e)}")
        
        # You might need to define GeminiEnums in your LLMEnums.py
        self.enums = GeminiEnums 
        self.rate_limiter = RateLimitTracker()

    def _test_connection(self) -> bool:
        """Test the connection to Gemini API by listing models"""
        try:
            self.logger.info("Testing Gemini connection by listing models...")
            # Attempt to list models to confirm API key is valid and connection works
            models = genai.list_models()
            for model in models:
                self.logger.debug(f"Found Gemini model: {model.name}")
            self.logger.info("Gemini connection test successful. Models listed.")
            return True
        except Exception as e:
            self.logger.error(f"Gemini connection test failed: {str(e)}. Make sure your API key is correct and services are enabled.")
            return False

    def set_generation_model(self, model_id: str) -> None:
        self.generation_model_id = model_id

    def set_embedding_model(self, model_id: str, embedding_size: int) -> None:
        self.embedding_model_id = model_id
        self.embedding_size = embedding_size

    def process_text(self, text: str) -> str:
        # Gemini handles longer inputs, but still good to have a limit
        return text[:self.default_input_max_characters].strip()

    def generate_text(self, 
                     prompt: str, 
                     chat_history: List[Dict[str, str]] = None,
                     max_output_tokens: Optional[int] = None,
                     temperature: Optional[float] = None) -> Optional[str]:
        
        if not self.generation_model_id:
            self.logger.error("Gemini generation model not set.")
            return None

        try:
            self.rate_limiter.wait_if_needed()
            
            model = genai.GenerativeModel(self.generation_model_id)
            chat = model.start_chat(history=[]) # Start with empty history, then add provided chat_history

            # Add chat history if provided
            if chat_history:
                for message in chat_history:
                    role = message["role"]
                    content = message["content"]
                    # Gemini roles are 'user' and 'model' (for bot/assistant)
                    if role == "assistant": # OpenAI uses 'assistant', Gemini uses 'model'
                        chat.history.append({'role': 'model', 'parts': [content]})
                    else: # Assuming 'user'
                        chat.history.append({'role': role, 'parts': [content]})
            
            # Send the current prompt
            response = chat.send_message(
                self.process_text(prompt),
                generation_config={
                    "max_output_tokens": max_output_tokens or self.default_generation_max_output_tokens,
                    "temperature": temperature or self.default_generation_temperature,
                }
            )
            
            return response.text

        except Exception as e:
            self.logger.error(f"Error generating text with Gemini: {str(e)}")
            return None

    def embed_text(self, text: str, document_type: Optional[str] = None) -> Optional[List[float]]:
        if not self.embedding_model_id:
            self.logger.error("Gemini embedding model not set.")
            return None

        try:
            self.rate_limiter.wait_if_needed()
            
            # Gemini embedding function
            # The model name for embedding is usually 'models/embedding-001'
            response = genai.embed_content(
                model=self.embedding_model_id,
                content=self.process_text(text),
                # If you need specific task_type, add it here (e.g., for search_query, retrieval_document)
                # task_type=DocumentTypeEnum.QUERY.value if document_type == DocumentTypeEnum.QUERY.value else DocumentTypeEnum.DOCUMENT.value
            )
            
            return response['embedding']

        except Exception as e:
            self.logger.error(f"Error embedding text with Gemini: {str(e)}")
            return None

    def embed_batch(self, texts: List[str], batch_size: int = 100) -> List[Optional[List[float]]]:
        """Batch process texts for embedding with Gemini."""
        all_embeddings = []
        
        # Gemini embedding batch size can be up to 100,000 for inputs, but rate limits apply.
        # Stick to a reasonable batch size for API stability.
        # The genai.embed_content can take a list of strings for batching directly.
        
        # Process in smaller batches for more reliability, respecting Gemini's batch capabilities
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                self.rate_limiter.wait_if_needed()
                
                # Process texts to stay within token limits (if applied by Gemini)
                processed_texts = [self.process_text(text) for text in batch]
                
                # Directly use genai.embed_content for batching
                response = genai.embed_content(
                    model=self.embedding_model_id,
                    content=processed_texts,
                    # Optional: task_type=[DocumentTypeEnum.DOCUMENT.value]*len(processed_texts)
                )
                
                # The response structure is a dictionary with 'embedding' key containing a list of embeddings
                embeddings = response['embedding']
                all_embeddings.extend(embeddings)
                
                self.logger.info(f"Successfully embedded batch of {len(batch)} texts ({i + len(batch)}/{len(texts)})")
                
            except Exception as e:
                self.logger.error(f"Error batch embedding texts with Gemini: {str(e)}")
                # If batch fails, attempt individual embedding as fallback
                fallback_embeddings = []
                self.logger.info(f"Falling back to individual embedding for batch {i//batch_size + 1}")
                
                for text in batch:
                    try:
                        vector = self.embed_text(text=text)
                        fallback_embeddings.append(vector)
                    except Exception as inner_e:
                        self.logger.error(f"Error in individual embedding fallback: {str(inner_e)}")
                        fallback_embeddings.append(None)
                
                all_embeddings.extend(fallback_embeddings)
        
        return all_embeddings

    def construct_prompt(self, prompt: str, role: str) -> Dict[str, str]:
        # Gemini expects roles 'user' or 'model'
        gemini_role = 'model' if role == 'assistant' else role
        return {
            "role": gemini_role,
            "parts": [self.process_text(prompt)] # Gemini expects 'parts' list
        }