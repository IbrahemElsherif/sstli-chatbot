from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any, Union
from .LLMEnums import ModelType
from helpers.config import Settings
import logging

class LLMInterface(ABC):
    """Abstract base class for LLM providers with standardized interface."""
    
    @abstractmethod
    def set_generation_model(self, model_id: str) -> None:
        """Set the model to use for text generation."""
        pass

    @abstractmethod
    def set_embedding_model(self, model_id: str, embedding_size: int) -> None:
        """Set the model to use for text embeddings."""
        pass

    @abstractmethod
    def generate_text(self, 
                    prompt: str, 
                    chat_history: Optional[List[Dict[str, str]]] = None, 
                    max_output_tokens: Optional[int] = None,
                    temperature: Optional[float] = None) -> Optional[str]:
        """Generate text based on the prompt and optional chat history."""
        pass

    @abstractmethod
    def embed_text(self, 
                text: str, 
                document_type: Optional[str] = None) -> Optional[List[float]]:
        """Generate embeddings for the given text."""
        pass
    
    @abstractmethod
    def construct_prompt(self, prompt: str, role: str) -> Dict[str, str]:
        """Construct a prompt in the format expected by the model."""
        pass

    def embed_batch(self, 
                    texts: List[str], 
                    batch_size: int = 100,
                    document_type: Optional[str] = None) -> List[Optional[List[float]]]:
        """
        Process a batch of texts for embedding with error handling and progress tracking.
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process in each batch
            document_type: Optional document type for embedding
            
        Returns:
            List of embeddings, with None for failed embeddings
        """
        logger = logging.getLogger(__name__)
        
        results = []
        total = len(texts)
        
        for i in range(0, total, batch_size):
            batch = texts[i:i + batch_size]
            try:
                # Log batch processing start
                logger.info(f"Processing batch {i//batch_size + 1} of {(total + batch_size - 1)//batch_size}")
                
                # Process each text in the batch
                batch_results = []
                for text in batch:
                    try:
                        # Check if text is valid
                        if not text or not isinstance(text, str):
                            logger.warning(f"Invalid text in batch: {text}")
                            batch_results.append(None)
                            continue
                            
                        # Process the text
                        embedding = self.embed_text(text, document_type)
                        batch_results.append(embedding)
                        
                    except Exception as text_error:
                        logger.error(f"Error processing text in batch: {str(text_error)}")
                        batch_results.append(None)
                
                results.extend(batch_results)
                logger.info(f"Successfully processed batch {i//batch_size + 1}")
                
            except Exception as e:
                logger.error(f"Error processing batch {i//batch_size + 1}: {str(e)}")
                results.extend([None] * len(batch))
                
        return results

    def validate_model_config(self, model_type: ModelType, settings: Settings) -> bool:
        """
        Validate model configuration before use.
        
        Args:
            model_type: Type of model (generation or embedding)
            settings: Application settings
            
        Returns:
            bool: True if configuration is valid
        """
        if model_type == ModelType.GENERATION:
            return bool(settings.GENERATION_MODEL_ID)
        elif model_type == ModelType.EMBEDDING:
            return bool(settings.EMBEDDING_MODEL_ID and settings.EMBEDDING_MODEL_SIZE)
        return False

    def process_chat_history(self, 
                            chat_history: Optional[List[Dict[str, str]]] = None) -> List[Dict[str, str]]:
        """
        Process and validate chat history.
        
        Args:
            chat_history: Optional list of chat messages
            
        Returns:
            Processed chat history
        """
        if not chat_history:
            return []
            
        return [
            msg for msg in chat_history 
            if isinstance(msg, dict) and 'role' in msg and 'content' in msg
        ]
