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

def test_embeddings(provider):
    """Test the embedding functionality"""
    # Set the model for embeddings
    model_id = os.getenv('EMBEDDING_MODEL_ID', 'BAAI/bge-small-en-v1.5')
    embedding_size = int(os.getenv('EMBEDDING_MODEL_SIZE', '384'))
    
    provider.set_embedding_model(model_id, embedding_size)
    
    # Test text
    test_text = "This is a test sentence for embeddings."
    
    logger.info("Testing embedding with model: %s", model_id)
    
    # Get single embedding
    single_embedding = provider.embed_text(test_text)
    if single_embedding:
        logger.info("Successfully generated embedding with length: %s", len(single_embedding))
        print("\nSample of embedding vector (first 5 dimensions):")
        print(single_embedding[:5])
        return True
    else:
        logger.error("Failed to generate embedding")
        return False

def test_generation(provider):
    """Test the text generation functionality"""
    # Set the model for generation
    model_id = os.getenv('GENERATION_MODEL_ID', 'Qwen/Qwen3-235B-A22B')
    provider.set_generation_model(model_id)
    
    # Enable thinking mode for Qwen3
    provider.thinking_enabled = True
    
    # Test prompt
    test_prompt = "Give me a short introduction to large language models."
    
    logger.info("Testing generation with model: %s", model_id)
    
    # Generate text
    response = provider.generate_text(
        prompt=test_prompt,
        chat_history=None,
        max_output_tokens=1024,
        temperature=0.6
    )
    
    if response:
        logger.info("Successfully generated response:")
        print("\n" + "="*50)
        print(response)
        print("="*50 + "\n")
        return True
    else:
        logger.error("Failed to generate response")
        return False

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
    
    # Test embeddings first
    logger.info("\n===== TESTING EMBEDDINGS =====")
    embedding_success = test_embeddings(provider)
    
    # Test generation
    logger.info("\n===== TESTING GENERATION =====")
    generation_success = test_generation(provider)
    
    # Report overall status
    if embedding_success and generation_success:
        logger.info("\n✅ All tests passed successfully!")
    elif embedding_success:
        logger.info("\n⚠️ Embedding test passed, but generation test failed")
    elif generation_success:
        logger.info("\n⚠️ Generation test passed, but embedding test failed")
    else:
        logger.error("\n❌ Both tests failed")

if __name__ == "__main__":
    main() 