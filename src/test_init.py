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

def test_init():
    """Test proper initialization with the correct model formats"""
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
    
    # Get models from environment
    embedding_model_id = os.getenv('EMBEDDING_MODEL_ID', 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
    embedding_size = int(os.getenv('EMBEDDING_MODEL_SIZE', '768'))
    
    # Set the embedding model
    provider.set_embedding_model(embedding_model_id, embedding_size)
    
    # Print current configuration
    print("\n===== CURRENT CONFIGURATION =====")
    print(f"Embedding Model: {provider.embedding_model_id}")
    print(f"Embedding Size: {provider.embedding_size}")
    print(f"Embedding Format: {provider.embedding_format}")
    print(f"Model-specific format: {provider.model_formats.get(embedding_model_id)}")
    
    # Test a simple embedding
    test_text = "Simple test text for embedding initialization."
    print("\n===== TESTING EMBEDDING =====")
    print(f"Test text: {test_text}")
    
    try:
        # Try to embed the text
        embedding = provider.embed_text(test_text)
        if embedding:
            print(f"✅ Embedding successful! Vector size: {len(embedding)}")
            print(f"First 5 values: {embedding[:5]}")
        else:
            print("❌ Failed to generate embedding")
    except Exception as e:
        print(f"❌ Error during embedding: {str(e)}")
    
if __name__ == "__main__":
    test_init() 