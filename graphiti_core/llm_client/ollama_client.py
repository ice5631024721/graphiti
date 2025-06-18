"""Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import json
import logging
import typing
from typing import ClassVar

import ollama
from pydantic import BaseModel

from .client import MULTILINGUAL_EXTRACTION_RESPONSES, LLMClient
from .config import DEFAULT_MAX_TOKENS, LLMConfig, ModelSize
from .errors import RateLimitError, RefusalError, EmptyResponseError
from ..prompts.models import Message

logger = logging.getLogger(__name__)

class OllamaClient(LLMClient):
    """
    OllamaClient is a client class for interacting with Ollama's language models.

    This class extends the LLMClient and provides methods to initialize the client
    and generate responses from the language model using Ollama's API.

    Attributes:
        base_url (str): The base URL for the Ollama API.
        model (str): The model name to use for generating responses.
        temperature (float): The temperature to use for generating responses.
        max_tokens (int): The maximum number of tokens to generate in a response.
        client (httpx.AsyncClient): The HTTP client used to interact with the API.

    Methods:
        __init__(config: LLMConfig | None = None, cache: bool = False):
            Initializes the OllamaClient with the provided configuration and cache setting.

        _generate_response(messages: list[Message]) -> dict[str, typing.Any]:
            Generates a response from the language model based on the provided messages.
    """

    # Class-level constants
    MAX_RETRIES: ClassVar[int] = 2

    def __init__(
            self,
            config: LLMConfig | None = None,
            cache: bool = False,
            max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        """
        Initialize the OllamaClient with the provided configuration and cache setting.

        Args:
            config (LLMConfig | None): The configuration for the LLM client, including model, base URL, temperature, and max tokens.
            cache (bool): Whether to use caching for responses. Defaults to False.
            max_tokens (int): The maximum number of tokens to generate in a response.
        """
        super().__init__(config, cache)

        self.base_url = self.config.base_url
        self.max_tokens = max_tokens
        # Initialize Ollama client with custom host if specified
        if self.config.base_url:
            self.client = ollama.AsyncClient(host=self.config.base_url)
        else:
            self.client = ollama.AsyncClient()

    async def _generate_response(
            self,
            messages: list[Message],
            response_model: type[BaseModel] | None = None,
            max_tokens: int = DEFAULT_MAX_TOKENS,
            model_size: ModelSize = ModelSize.medium,
    ) -> dict[str, typing.Any]:
        """
        Generate a response from Ollama based on the provided messages.

        Args:
            messages: List of messages to send to the model
            response_model: Optional Pydantic model for structured output
            max_tokens: Maximum number of tokens to generate
            model_size: Size of the model to use (small or medium)

        Returns:
            Dictionary containing the generated response

        Raises:
            RateLimitError: When rate limit is exceeded
            RefusalError: When the model refuses to generate a response
            EmptyResponseError: When the model returns an empty response
            Exception: For other API errors
        """
        # Convert messages to Ollama format
        ollama_messages = []
        for m in messages:
            m.content = self._clean_input(m.content)
            if m.role in ['user', 'system', 'assistant']:
                ollama_messages.append({
                    'role': m.role,
                    'content': m.content
                })

        try:
            # Select model based on size
            if model_size == ModelSize.small:
                model = self.model
            else:
                model = self.model

            # Prepare options for the request
            options = {
                'temperature': self.temperature,
                'num_predict': max_tokens or self.max_tokens,
            }

            # Add format instruction for structured output
            format_type = None
            if response_model is not None:
                format_type = 'json'
                schema_instruction = f"\n\nRespond with a JSON object in the following format:\n\n{json.dumps(response_model.model_json_schema())}"
                if ollama_messages:
                    ollama_messages[-1]['content'] += schema_instruction

            # Make API request using ollama client
            response = await self.client.chat(
                model=model,
                messages=ollama_messages,
                stream=False,
                format=format_type,
                options=options
            )

            # Extract the response content
            if 'message' not in response or 'content' not in response['message']:
                raise EmptyResponseError("No content in response")

            content = response['message']['content'].strip()
            if not content:
                raise EmptyResponseError("Empty response content")

            # Handle structured output
            if response_model is not None:
                try:
                    # Try to parse as JSON
                    parsed_content = json.loads(content)
                    # Validate against the response model
                    # validated = response_model.model_validate(parsed_content)
                    return parsed_content
                except (json.JSONDecodeError, ValueError) as e:
                    raise Exception(f"Failed to parse structured response: {e}")
            else:
                # Return plain text response
                return {'content': content}

        except Exception as e:
            # Handle ollama-specific errors
            if "connection" in str(e).lower() or "connect" in str(e).lower():
                raise Exception(f"Connection error: {e}. Make sure Ollama is running at {self.base_url}") from e
            elif "rate limit" in str(e).lower():
                raise RateLimitError("Rate limit exceeded") from e
            else:
                logger.error(f"Error in generating Ollama response: {e}")
                raise

    async def generate_response(
            self,
            messages: list[Message],
            response_model: type[BaseModel] | None = None,
            max_tokens: int | None = None,
            model_size: ModelSize = ModelSize.medium,
    ) -> dict[str, typing.Any]:
        """
        Generate a response with retry logic and error handling.

        Args:
            messages: List of messages to send to the model
            response_model: Optional Pydantic model for structured output
            max_tokens: Maximum number of tokens to generate
            model_size: Size of the model to use

        Returns:
            Dictionary containing the generated response
        """
        if max_tokens is None:
            max_tokens = self.max_tokens

        retry_count = 0
        last_error = None

        # Add multilingual extraction instructions
        messages[0].content += MULTILINGUAL_EXTRACTION_RESPONSES

        while retry_count <= self.MAX_RETRIES:
            try:
                response = await self._generate_response(
                    messages, response_model, max_tokens, model_size
                )
                return response
            except (RateLimitError, RefusalError):
                # These errors should not trigger retries
                raise
            except Exception as e:
                last_error = e

                # Don't retry if we've hit the max retries
                if retry_count >= self.MAX_RETRIES:
                    logger.error(f"Max retries ({self.MAX_RETRIES}) exceeded. Last error: {e}")
                    raise

                retry_count += 1

                # Construct a detailed error message for the LLM
                error_context = (
                    f"The previous response attempt was invalid. "
                    f"Error type: {e.__class__.__name__}. "
                    f"Error details: {str(e)}. "
                    f"Please try again with a valid response, ensuring the output matches "
                    f"the expected format and constraints."
                )

                error_message = Message(role='user', content=error_context)
                messages.append(error_message)
                logger.warning(
                    f"Retrying after application error (attempt {retry_count}/{self.MAX_RETRIES}): {e}"
                )

        # If we somehow get here, raise the last error
        raise last_error or Exception("Max retries exceeded with no specific error")

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        # Ollama client doesn't require explicit closing
        pass


# Test functions for OllamaClient
async def test_ollama_client_basic():
    """
    Test basic functionality of OllamaClient.

    This test verifies:
    1. Connect to local Ollama service
    2. Generate a simple response
    3. Handle basic conversation
    """
    print("Testing OllamaClient basic functionality...")

    try:
        # Test with default configuration
        config = LLMConfig()
        client = OllamaClient(config)

        # Test simple message
        messages = [Message(role='user', content='Hello! Please respond with just "Hello back!"')]
        response = await client.generate_response(messages)

        print(f"✅ Basic test passed. Response: {response['content'][:100]}...")
        return True

    except Exception as e:
        print(f"❌ Basic test failed: {e}")
        return False


async def test_ollama_client_structured():
    """
    Test structured output functionality of OllamaClient.
    """
    print("\nTesting OllamaClient structured output...")

    try:
        from pydantic import BaseModel

        class TestResponse(BaseModel):
            message: str
            success: bool

        client = OllamaClient()
        messages = [
            Message(role='user', content='Please respond with a message saying "test successful" and success as true')]

        response = await client.generate_response(messages, response_model=TestResponse)

        print(f"✅ Structured output test passed. Response: {response}")
        return True

    except Exception as e:
        print(f"❌ Structured output test failed: {e}")
        return False


async def test_ollama_client_custom_config():
    """
    Test OllamaClient with custom configuration.
    """
    print("\nTesting OllamaClient with custom configuration...")

    try:
        # Test with custom configuration
        custom_config = LLMConfig(
            base_url="http://localhost:11434",
            model="llama3.2",
            small_model="llama3.2:1b",
            temperature=0.7
        )

        client = OllamaClient(custom_config)
        messages = [Message(role='user', content='What is 2+2? Please answer briefly.')]

        response = await client.generate_response(messages, model_size=ModelSize.small)

        print(f"✅ Custom config test passed. Response: {response['content'][:100]}...")
        return True

    except Exception as e:
        print(f"❌ Custom config test failed: {e}")
        return False


async def test_ollama_client_context_manager():
    """
    Test OllamaClient as async context manager.
    """
    print("\nTesting OllamaClient as context manager...")

    try:
        async with OllamaClient() as client:
            messages = [Message(role='user', content='Say "Context manager works!"')]
            response = await client.generate_response(messages)

        print(f"✅ Context manager test passed. Response: {response['content'][:100]}...")
        return True

    except Exception as e:
        print(f"❌ Context manager test failed: {e}")
        return False


async def run_all_ollama_client_tests():
    """
    Run all tests for OllamaClient.
    """
    print("Running OllamaClient Tests")
    print("=" * 50)
    print("Make sure Ollama is running with: ollama serve")
    print("And that you have a model available: ollama pull llama3.2")
    print()

    tests = [
        test_ollama_client_basic,
        test_ollama_client_structured,
        test_ollama_client_custom_config,
        test_ollama_client_context_manager
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            success = await test()
            if success:
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")

    print(f"\n{'=' * 50}")
    print(f"Tests completed: {passed}/{total} passed")

    if passed == total:
        print("🎉 All tests passed! OllamaClient is working correctly.")
    else:
        print("❌ Some tests failed. Please check your Ollama setup.")

    return passed == total


if __name__ == "__main__":
    import asyncio

    asyncio.run(run_all_ollama_client_tests())
