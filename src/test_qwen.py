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
    
    # Set the model for generation
    model_id = os.getenv('GENERATION_MODEL_ID', 'Qwen/Qwen3-235B-A22B')
    provider.set_generation_model(model_id)
    
    # Test prompt
    test_prompt = "Give me a short introduction to large language models."
    
    # Enable thinking mode for Qwen3
    provider.thinking_enabled = True
    
    logger.info("Generating text with model: %s", model_id)
    logger.info("Prompt: %s", test_prompt)
    
    # Generate text
    response = provider.generate_text(
        prompt=test_prompt,
        chat_history=None,
        max_output_tokens=1024,
        temperature=0.6
    )
    
    if response:
        logger.info("Response received:")
        print("\n" + "="*50)
        print(response)
        print("="*50 + "\n")
    else:
        logger.error("Failed to generate response")

if __name__ == "__main__":
    main() 