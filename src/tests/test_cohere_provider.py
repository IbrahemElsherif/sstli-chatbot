import unittest
from unittest.mock import Mock, patch, MagicMock
from src.stores.llm.providers.CoHereProvider import CoHereProvider
from src.stores.llm.LLMEnums import CoHereEnums, DocumentTypeEnum
from typing import List
from cohere.errors import TooManyRequestsError
import time

class TestCoHereProvider(unittest.TestCase):
    def setUp(self):
        self.api_key = "test_key"
        self.provider = CoHereProvider(api_key=self.api_key)
        self.provider.set_generation_model("command-alpha")
        self.provider.set_embedding_model("embed-multilingual-light-v3.0", 1024)

    def test_process_text(self):
        long_text = "a" * 2000
        processed = self.provider.process_text(long_text)
        self.assertEqual(len(processed), 1000)
        
        short_text = "hello world"
        processed = self.provider.process_text(short_text)
        self.assertEqual(processed, short_text)

    @patch('cohere.Client')
    def test_generate_text(self, mock_client):
        # Setup mock client
        mock_client_instance = Mock()
        mock_response = Mock()
        mock_response.text = "Test response"
        mock_client_instance.chat.return_value = mock_response
        
        # Apply mock to provider
        self.provider.client = mock_client_instance
        
        # Test generate_text
        result = self.provider.generate_text(
            prompt="Test prompt",
            chat_history=[{"role": "SYSTEM", "text": "You are a helpful assistant"}]
        )
        
        # Verify result
        self.assertEqual(result, "Test response")
        mock_client_instance.chat.assert_called_once()

    @patch('cohere.Client')
    def test_embed_text(self, mock_client):
        # Setup mock client
        mock_client_instance = Mock()
        mock_response = Mock()
        mock_response.embeddings = [[0.1, 0.2, 0.3]]
        mock_client_instance.embed.return_value = mock_response
        
        # Apply mock to provider
        self.provider.client = mock_client_instance
        
        # Test embed_text
        result = self.provider.embed_text("Test text")
        
        # Verify result
        self.assertEqual(result, [0.1, 0.2, 0.3])
        mock_client_instance.embed.assert_called_once()
        
    @patch('cohere.Client')
    def test_embed_text_with_rate_limit(self, mock_client):
        # Setup mock client to raise rate limit error on first call
        mock_client_instance = Mock()
        mock_response = Mock()
        mock_response.embeddings = [[0.1, 0.2, 0.3]]
        
        # First call raises error, second call succeeds
        mock_client_instance.embed.side_effect = [
            TooManyRequestsError("Rate limit exceeded"),
            mock_response
        ]
        
        # Apply mock to provider
        self.provider.client = mock_client_instance
        
        # Test embed_text with rate limit handling
        result = self.provider.embed_text("Test text")
        
        # Verify result
        self.assertEqual(result, [0.1, 0.2, 0.3])
        self.assertEqual(mock_client_instance.embed.call_count, 2)

    @patch('cohere.Client')
    def test_embed_batch(self, mock_client):
        # Setup mock client
        mock_client_instance = Mock()
        mock_response = Mock()
        mock_response.embeddings = [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6]
        ]
        mock_client_instance.embed.return_value = mock_response
        
        # Apply mock to provider
        self.provider.client = mock_client_instance
        
        # Test embed_batch
        texts = ["Text 1", "Text 2"]
        result = self.provider.embed_batch(texts, batch_size=2)
        
        # Verify result
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], [0.1, 0.2, 0.3])
        self.assertEqual(result[1], [0.4, 0.5, 0.6])
        mock_client_instance.embed.assert_called_once()
        
    @patch('cohere.Client')
    def test_embed_batch_with_rate_limit(self, mock_client):
        # Setup mock client with rate limit error then success
        mock_client_instance = Mock()
        mock_response = Mock()
        mock_response.embeddings = [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6]
        ]
        
        # First call raises error, second call succeeds
        mock_client_instance.embed.side_effect = [
            TooManyRequestsError("Rate limit exceeded"),
            mock_response
        ]
        
        # Apply mock to provider
        self.provider.client = mock_client_instance
        
        # Test embed_batch with rate limit handling
        texts = ["Text 1", "Text 2"]
        result = self.provider.embed_batch(texts, batch_size=2)
        
        # Verify result
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], [0.1, 0.2, 0.3])
        self.assertEqual(result[1], [0.4, 0.5, 0.6])
        self.assertEqual(mock_client_instance.embed.call_count, 2)

    def test_construct_prompt(self):
        prompt = "Test prompt"
        role = CoHereEnums.USER.value
        
        result = self.provider.construct_prompt(prompt, role)
        
        self.assertEqual(result["role"], role)
        self.assertEqual(result["text"], prompt)

if __name__ == '__main__':
    unittest.main() 