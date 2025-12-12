"""
Embedding Service Module
Handles text embedding using OpenAI or other providers.
"""

import os
from typing import Any

from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()


class EmbeddingService:
    """Service for generating text embeddings."""
    
    def __init__(
        self,
        model: str | None = None,
        dimensions: int | None = None,
    ):
        """
        Initialize embedding service.
        
        Args:
            model: Embedding model name (default: text-embedding-3-small)
            dimensions: Embedding dimensions (default: 1536)
        """
        self.model = model or os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
        self.dimensions = dimensions or int(os.getenv("EMBEDDING_DIMENSION", "1024"))
        
        self._embeddings = OpenAIEmbeddings(
            model=self.model,
            dimensions=self.dimensions,
        )
    
    def embed_text(self, text: str) -> list[float]:
        """
        Embed a single text string.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as list of floats
        """
        return self._embeddings.embed_query(text)
    
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Embed multiple text strings.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        return self._embeddings.embed_documents(texts)
    
    def get_langchain_embeddings(self) -> OpenAIEmbeddings:
        """Return the underlying LangChain embeddings object."""
        return self._embeddings
    
    @property
    def info(self) -> dict[str, Any]:
        """Return embedding service info."""
        return {
            "model": self.model,
            "dimensions": self.dimensions,
        }

