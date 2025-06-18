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

import logging
from collections.abc import Iterable

import ollama
from pydantic import Field

from .client import EmbedderClient, EmbedderConfig

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = 'nomic-embed-text'
DEFAULT_BASE_URL = 'http://localhost:11434'


class OllamaEmbedderConfig(EmbedderConfig):
    """Configuration for Ollama Embedder."""
    embedding_model: str = Field(default=DEFAULT_EMBEDDING_MODEL)
    base_url: str = Field(default=DEFAULT_BASE_URL)
    timeout: float = Field(default=120.0)


class OllamaEmbedder(EmbedderClient):
    """
    Ollama Embedder Client

    This client provides embeddings using locally hosted Ollama models.
    It supports both single and batch embedding generation.
    """

    def __init__(self, config: OllamaEmbedderConfig | None = None):
        """
        Initialize the OllamaEmbedder with the provided configuration.

        Args:
            config (OllamaEmbedderConfig | None): Configuration for the embedder.
                If None, default configuration will be used.
        """
        if config is None:
            config = OllamaEmbedderConfig()
        self.config = config
        # Initialize Ollama client with custom host if specified
        if config.base_url != DEFAULT_BASE_URL:
            self.client = ollama.AsyncClient(host=config.base_url)
        else:
            self.client = ollama.AsyncClient()

    async def create(
            self, input_data: str | list[str] | Iterable[int] | Iterable[Iterable[int]]
    ) -> list[float]:
        """
        Create embeddings for the given input data.

        Args:
            input_data: Input data to embed. Can be a string, list of strings,
                       or iterables of integers.

        Returns:
            List of float values representing the embedding vector.

        Raises:
            Exception: If the API request fails or returns invalid data.
        """
        # Convert input to string format
        if isinstance(input_data, str):
            text_input = input_data
        elif isinstance(input_data, list) and all(isinstance(item, str) for item in input_data):
            # Join list of strings
            text_input = ' '.join(input_data)
        elif isinstance(input_data, Iterable):
            # Convert iterable to string
            text_input = ' '.join(str(item) for item in input_data if item is not None)
        else:
            text_input = str(input_data)

        try:
            # Use ollama client to generate embeddings
            response = await self.client.embeddings(
                model=self.config.embedding_model,
                prompt=text_input
            )

            # Extract embedding from response
            if 'embedding' not in response:
                raise Exception("No embedding found in response")

            embedding = response['embedding']
            if not isinstance(embedding, list):
                raise Exception("Invalid embedding format")

            # Truncate to configured dimension if necessary
            if len(embedding) > self.config.embedding_dim:
                embedding = embedding[:self.config.embedding_dim]
            elif len(embedding) < self.config.embedding_dim:
                # Pad with zeros if embedding is shorter than expected
                embedding.extend([0.0] * (self.config.embedding_dim - len(embedding)))

            return embedding

        except Exception as e:
            logger.error(f"Error in Ollama embedding: {e}")
            if "connection" in str(e).lower():
                raise Exception(f"Connection error: {e}. Make sure Ollama is running at {self.config.base_url}") from e
            raise

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        """
        Create embeddings for a batch of input strings.

        Args:
            input_data_list: List of strings to embed.

        Returns:
            List of embedding vectors, one for each input string.

        Raises:
            Exception: If any API request fails or returns invalid data.
        """
        if not input_data_list:
            return []

        embeddings = []

        # Process each input individually since Ollama API doesn't support batch requests
        for text in input_data_list:
            try:
                embedding = await self.create(text)
                embeddings.append(embedding)
            except Exception as e:
                logger.error(f"Failed to create embedding for text: {text[:50]}... Error: {e}")
                # You might want to handle this differently based on your needs
                # For now, we'll raise the exception to maintain consistency
                raise

        return embeddings

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if hasattr(self.client, 'close'):
            await self.client.close()

    async def close(self):
        """Close the Ollama client."""
        if hasattr(self.client, 'close'):
            await self.client.close()