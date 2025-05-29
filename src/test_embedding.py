import os
import sys
import logging
from dotenv import load_dotenv

# Add the current directory to the path to ensure imports work correctly
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from stores.llm.providers.HuggingFaceProvider import HuggingFaceProvider

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv('.env')

def main():
    # Get API key from environment
    api_key = os.getenv('HUGGINGFACE_API_KEY')
    if not api_key:
        logger.error("HUGGINGFACE_API_KEY not found in .env file")
        sys.exit(1)
        
    # Create the HuggingFace provider
    provider = HuggingFaceProvider(
        api_key=api_key,
        api_url="https://api-inference.huggingface.co/models",
        default_input_max_characters=4096,
        default_generation_max_output_tokens=2048,
        default_generation_temperature=0.6
    )
    
    # Set the model for embeddings
    model_id = os.getenv('EMBEDDING_MODEL_ID', 'intfloat/multilingual-e5-base')
    embedding_size = int(os.getenv('EMBEDDING_MODEL_SIZE', '768'))
    
    provider.set_embedding_model(model_id, embedding_size)
    
    # Test texts in both English and Arabic
    test_texts = [
        # English texts
        "This is a test sentence for embeddings.",
        "Large language models are transforming AI applications.",
        "Vector embeddings allow semantic search capabilities.",
        # Arabic texts
        "هذه جملة اختبار للتضمينات.",
        "النماذج اللغوية الكبيرة تحول تطبيقات الذكاء الاصطناعي.",
        "تسمح التضمينات المتجهة بإمكانيات البحث الدلالي."
    ]
    
    logger.info("Testing embedding with model: %s", model_id)
    
    # Test single embedding for English
    english_embedding = provider.embed_text(test_texts[0])
    if english_embedding:
        logger.info("Successfully generated English embedding with length: %s", len(english_embedding))
    else:
        logger.error("Failed to generate English embedding")
        
    # Test single embedding for Arabic
    arabic_embedding = provider.embed_text(test_texts[3])
    if arabic_embedding:
        logger.info("Successfully generated Arabic embedding with length: %s", len(arabic_embedding))
    else:
        logger.error("Failed to generate Arabic embedding")
        
    # Get batch embeddings
    logger.info("Testing batch embedding for mixed languages...")
    batch_embeddings = provider.embed_batch(test_texts)
    successful_embeds = sum(1 for embedding in batch_embeddings if embedding is not None)
    
    if successful_embeds == len(test_texts):
        logger.info("Successfully generated all %s batch embeddings", len(test_texts))
    else:
        logger.error("Generated only %s/%s batch embeddings", successful_embeds, len(test_texts))
    
    # Print sample of embeddings
    if english_embedding and len(english_embedding) > 5:
        print("\nSample of English embedding vector (first 5 dimensions):")
        print(english_embedding[:5])
        
    if arabic_embedding and len(arabic_embedding) > 5:
        print("\nSample of Arabic embedding vector (first 5 dimensions):")
        print(arabic_embedding[:5])
        
    print(f"\nTotal dimensions: {embedding_size}")

if __name__ == "__main__":
    main() 