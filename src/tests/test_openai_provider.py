import unittest
from unittest.mock import Mock, patch
from src.stores.llm.providers.OpenAIProvider import OpenAIProvider, RateLimitTracker
from src.stores.llm.LLMEnums import OpenAIEnums
from datetime import datetime, timedelta
import time

class TestRateLimitTracker(unittest.TestCase):
    def setUp(self):
        self.tracker = RateLimitTracker(requests_per_minute=2)

    def test_can_make_request(self):
        self.assertTrue(self.tracker.can_make_request())
        self.tracker.add_request()
        self.assertTrue(self.tracker.can_make_request())
        self.tracker.add_request()
        self.assertFalse(self.tracker.can_make_request())

    def test_request_expiration(self):
        self.tracker.request_timestamps = [
            datetime.now() - timedelta(minutes=2)
        ]
        self.assertTrue(self.tracker.can_make_request())

class TestOpenAIProvider(unittest.TestCase):
    def setUp(self):
        self.api_key = "test_key"
        self.provider = OpenAIProvider(api_key=self.api_key)
        self.provider.set_generation_model("gpt-3.5-turbo")
        self.provider.set_embedding_model("text-embedding-ada-002", 1536)

    def test_process_text(self):
        long_text = "a" * 2000
        processed = self.provider.process_text(long_text)
        self.assertEqual(len(processed), 1000)
        
        short_text = "hello world"
        processed = self.provider.process_text(short_text)
        self.assertEqual(processed, short_text)

    @patch('openai.OpenAI')
    def test_generate_text(self, mock_openai):
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Test response"))]
        mock_client.chat.completions.create.return_value = mock_response
        
        self.provider.client = mock_client
        
        result = self.provider.generate_text(
            prompt="Test prompt",
            chat_history=[{"role": "system", "content": "You are a helpful assistant"}]
        )
        
        self.assertEqual(result, "Test response")
        mock_client.chat.completions.create.assert_called_once()

    @patch('openai.OpenAI')
    def test_embed_text(self, mock_openai):
        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
        mock_client.embeddings.create.return_value = mock_response
        
        self.provider.client = mock_client
        
        result = self.provider.embed_text("Test text")
        
        self.assertEqual(result, [0.1, 0.2, 0.3])
        mock_client.embeddings.create.assert_called_once()

    @patch('openai.OpenAI')
    def test_embed_batch(self, mock_openai):
        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [
            Mock(embedding=[0.1, 0.2, 0.3]),
            Mock(embedding=[0.4, 0.5, 0.6])
        ]
        mock_client.embeddings.create.return_value = mock_response
        
        self.provider.client = mock_client
        
        texts = ["Text 1", "Text 2"]
        result = self.provider.embed_batch(texts, batch_size=2)
        
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], [0.1, 0.2, 0.3])
        self.assertEqual(result[1], [0.4, 0.5, 0.6])

    def test_construct_prompt(self):
        prompt = "Test prompt"
        role = OpenAIEnums.USER.value
        
        result = self.provider.construct_prompt(prompt, role)
        
        self.assertEqual(result["role"], role)
        self.assertEqual(result["content"], prompt)

if __name__ == '__main__':
    unittest.main() 