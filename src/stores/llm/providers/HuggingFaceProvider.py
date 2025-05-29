from ..LLMInterface import LLMInterface
from ..LLMEnums import DocumentTypeEnum
from typing import List, Dict, Optional, Any, Union
import logging
import time
import requests
import json

class HuggingFaceProvider(LLMInterface):

    def __init__(self, 
                 api_key: str,
                 api_url: Optional[str] = None,
                 default_input_max_characters: int = 1000,
                 default_generation_max_output_tokens: int = 1000,
                 default_generation_temperature: float = 0.1):
        
        self.api_key = api_key
        self.api_url = api_url or "https://api-inference.huggingface.co/models"
        self.default_input_max_characters = default_input_max_characters
        self.default_generation_max_output_tokens = default_generation_max_output_tokens
        self.default_generation_temperature = default_generation_temperature

        self.generation_model_id = None
        self.embedding_model_id = None
        self.embedding_size = None
        self.embedding_format = "inputs"  # Default to inputs format now
        
        # Dictionary to store model-specific formats
        self.model_formats = {}
        
        # Flag to identify Qwen models
        self.is_qwen_model = False
        self.thinking_enabled = True

        # No actual client initialization needed, we'll use the API directly
        self.client = True
        self.logger = logging.getLogger(__name__)
        self.logger.info("HuggingFace provider initialized")

    def set_generation_model(self, model_id: str) -> None:
        self.generation_model_id = model_id
        # Check if this is a Qwen model
        self.is_qwen_model = "qwen" in model_id.lower() or "qwen3" in model_id.lower()
        if self.is_qwen_model:
            self.logger.info(f"Qwen model detected: {model_id}. Enabling specific handling.")
        self.logger.info(f"HuggingFace generation model set to: {model_id}")

    def set_embedding_model(self, model_id: str, embedding_size: int, embedding_format: str = None) -> None:
        self.embedding_model_id = model_id
        self.embedding_size = embedding_size
        
        # Auto-detect the correct format based on model name
        if embedding_format is None:
            if "sentence-transformers" in model_id.lower():
                embedding_format = "sentences"
                self.logger.info(f"Detected sentence-transformer model, setting format to 'sentences'")
            else:
                embedding_format = "inputs"
                self.logger.info(f"Using default 'inputs' format for non-sentence-transformer model")
        
        self.embedding_format = embedding_format
        self.model_formats[model_id] = embedding_format
        self.logger.info(f"HuggingFace embedding model set to: {model_id} with size {embedding_size}, format: {embedding_format}")

    def process_text(self, text: str) -> str:
        if not text:
            return ""
        return text[:self.default_input_max_characters].strip()

    def generate_text(self, 
                     prompt: str, 
                     chat_history: Optional[List[Dict[str, str]]] = None,
                     max_output_tokens: Optional[int] = None,
                     temperature: Optional[float] = None) -> Optional[str]:
        
        if not self.generation_model_id:
            self.logger.error("HuggingFace generation model not set")
            return None

        try:
            # Using Hugging Face Inference API
            API_URL = self._get_model_api_url(self.generation_model_id)
            headers = {"Authorization": f"Bearer {self.api_key}"}
            
            # Check if this is a Qwen model and handle differently
            if self.is_qwen_model:
                return self._generate_with_qwen(prompt, chat_history, max_output_tokens, temperature, API_URL, headers)
            
            # Process chat history into a prompt format for non-Qwen models
            processed_prompt = self.process_text(prompt)
            if chat_history:
                # Simple approach to combine chat history
                processed_prompt = "\n".join([
                    f"{msg.get('role', 'user')}: {msg.get('content', '')}" 
                    for msg in chat_history
                ]) + f"\nuser: {processed_prompt}"
            
            payload = {
                "inputs": processed_prompt,
                "parameters": {
                    "max_length": max_output_tokens or self.default_generation_max_output_tokens,
                    "temperature": temperature or self.default_generation_temperature,
                    "return_full_text": False
                }
            }
            
            response = requests.post(
                API_URL, 
                headers=headers, 
                json=payload,
                timeout=120  # Longer timeout for generation
            )
            
            if response.status_code != 200:
                self.logger.error(f"HuggingFace API error: {response.status_code} - {response.text}")
                return None
                
            return response.json()[0].get("generated_text")
            
        except Exception as e:
            self.logger.error(f"Error generating text: {str(e)}")
            return None
            
    def _generate_with_qwen(self, 
                          prompt: str, 
                          chat_history: Optional[List[Dict[str, str]]] = None,
                          max_output_tokens: Optional[int] = None,
                          temperature: Optional[float] = None,
                          API_URL: str = None,
                          headers: Dict[str, str] = None) -> Optional[str]:
        """Special handling for Qwen models which require different formatting"""
        try:
            # Format messages for Qwen
            messages = []
            
            # Add chat history if available
            if chat_history:
                messages.extend(chat_history)
                
            # Add current prompt as user message
            messages.append({"role": "user", "content": prompt})
            
            # Prepare parameters for Qwen3
            params = {
                "max_new_tokens": max_output_tokens or self.default_generation_max_output_tokens,
                "temperature": temperature or self.default_generation_temperature,
                "top_p": 0.95 if self.thinking_enabled else 0.8,
                "top_k": 20,
                "return_full_text": False
            }
            
            # Special handling for Qwen3 models
            if "qwen3" in self.generation_model_id.lower():
                payload = {
                    "inputs": {
                        "messages": messages
                    },
                    "parameters": {
                        **params,
                        "enable_thinking": self.thinking_enabled
                    }
                }
            else:
                # Fallback for other Qwen models
                payload = {
                    "inputs": {
                        "messages": messages
                    },
                    "parameters": params
                }
                
            self.logger.info(f"Sending request to Qwen model: {self.generation_model_id}")
            
            response = requests.post(
                API_URL, 
                headers=headers, 
                json=payload,
                timeout=180  # Longer timeout for Qwen models
            )
            
            if response.status_code != 200:
                self.logger.error(f"Qwen API error: {response.status_code} - {response.text}")
                return None
            
            result = response.json()
            
            # Handle different response formats
            if isinstance(result, list) and len(result) > 0:
                # Some models return a list with the generated text
                generated_text = result[0].get("generated_text")
                
                # For Qwen3 models, we need to parse the thinking and content
                if "qwen3" in self.generation_model_id.lower() and self.thinking_enabled:
                    # Check if there's a thinking section
                    if generated_text and "<think>" in generated_text and "</think>" in generated_text:
                        try:
                            # Extract the non-thinking content (after </think>)
                            thinking_end = generated_text.find("</think>") + len("</think>")
                            content = generated_text[thinking_end:].strip()
                            return content
                        except Exception as e:
                            self.logger.error(f"Error parsing Qwen3 thinking: {str(e)}")
                            # Return full text as fallback
                            return generated_text
                
                return generated_text
            elif isinstance(result, dict):
                # Some models return a dict with the generated text
                return result.get("generated_text")
            
            self.logger.error(f"Unexpected Qwen response format: {result}")
            return None
            
        except Exception as e:
            self.logger.error(f"Error generating text with Qwen: {str(e)}")
            return None
            
    def set_thinking_mode(self, enabled: bool = True) -> None:
        """Enable or disable thinking mode for Qwen3 models"""
        self.thinking_enabled = enabled
        self.logger.info(f"Qwen thinking mode set to: {enabled}")
    
    def embed_text(self, text: str, document_type: Optional[str] = None) -> Optional[List[float]]:
        if not self.embedding_model_id:
            self.logger.error("HuggingFace embedding model not set")
            return None
            
        try:
            # Use Sentence Transformers API
            API_URL = self._get_model_api_url(self.embedding_model_id)
            headers = {"Authorization": f"Bearer {self.api_key}"}
            
            # Process the text first
            processed_text = self.process_text(text)
            
            # Always use inputs format instead of trying both
            # BGE models require "inputs" format
            payload = {"inputs": processed_text}
            
            self.logger.info(f"Using 'inputs' format for model: {self.embedding_model_id}")
            response = requests.post(API_URL, headers=headers, json=payload, timeout=10)
            
            if response.status_code != 200:
                error_msg = ""
                try:
                    error_msg = response.json().get("error", "")
                except:
                    error_msg = response.text
                
                self.logger.error(f"HuggingFace API error: {response.status_code} - {error_msg}")
                return None
            
            # Get the embedding from the response
            embedding = response.json()
            
            # Handle different response formats
            if isinstance(embedding, list):
                if len(embedding) > 0:
                    if isinstance(embedding[0], list):
                        # Handle case where API returns a nested list [[0.1, 0.2, ...]]
                        return embedding[0]
                    else:
                        # Handle case where API returns a flat list [0.1, 0.2, ...]
                        return embedding
                else:
                    self.logger.error("Empty embedding response from API")
                    return None
            else:
                self.logger.error(f"Unexpected embedding format: {embedding}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error embedding text: {str(e)}")
            return None
    
    def embed_batch(self, texts: List[str], batch_size: int = 5) -> List[Optional[List[float]]]:
        if not self.embedding_model_id:
            self.logger.error("HuggingFace embedding model not set")
            return [None] * len(texts)
            
        all_embeddings = []
        
        # Process in smaller batches for Arabic text
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                # Process batch texts
                processed_batch = [self.process_text(text) for text in batch]
                
                API_URL = self._get_model_api_url(self.embedding_model_id)
                headers = {"Authorization": f"Bearer {self.api_key}"}
                
                # For sentence-transformers models, send the sentences directly
                payload = processed_batch
                self.logger.info(f"Processing batch of {len(batch)} texts with model: {self.embedding_model_id}")
                
                # Increased timeout for Arabic text processing
                response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
                
                if response.status_code != 200:
                    error_msg = ""
                    try:
                        error_msg = response.json().get("error", "")
                    except:
                        error_msg = response.text
                    
                    self.logger.error(f"HuggingFace API batch error: {response.status_code} - {error_msg}")
                    all_embeddings.extend([None] * len(batch))
                    continue
                
                # Process the embeddings
                embeddings = response.json()
                
                # Handle different response formats
                if isinstance(embeddings, list):
                    if len(embeddings) > 0:
                        if isinstance(embeddings[0], list):
                            # Handle case where API returns a list of lists [[0.1, 0.2, ...], [0.3, 0.4, ...]]
                            all_embeddings.extend(embeddings)
                        else:
                            # Handle case where API returns a single list [0.1, 0.2, ...]
                            all_embeddings.append(embeddings)
                    else:
                        self.logger.error("Empty embedding response from API")
                        all_embeddings.extend([None] * len(batch))
                else:
                    self.logger.error(f"Unexpected embedding format: {embeddings}")
                    all_embeddings.extend([None] * len(batch))
                
                self.logger.info(f"Successfully embedded batch of {len(batch)} texts ({i + len(batch)}/{len(texts)})")
                
            except Exception as e:
                self.logger.error(f"Error batch embedding texts: {str(e)}")
                # Add None values for failed embeddings
                all_embeddings.extend([None] * len(batch))
            
            # Longer delay between batches for Arabic text
            if i + batch_size < len(texts):
                time.sleep(2)
        
        return all_embeddings
    
    def construct_prompt(self, prompt: str, role: str) -> Dict[str, str]:
        return {
            "role": role,
            "content": self.process_text(prompt)
        }

    def _get_model_api_url(self, model_id: str) -> str:
        """Construct the full API URL for a specific model"""
        # Remove trailing slash if it exists
        base_url = self.api_url.rstrip('/')
        
        # If the base URL already includes /models, don't add it again
        if '/models' in base_url:
            return f"{base_url}/{model_id}"
        else:
            return f"{base_url}/{model_id}" 