"""
Pinecone Vector Store Module
Handles vector storage and retrieval with Pinecone.
"""

import os
import time
from typing import Any

from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from loguru import logger

from .embeddings import EmbeddingService

load_dotenv()


class PineconeStore:
    """Pinecone vector store for document embeddings."""
    
    def __init__(
        self,
        index_name: str | None = None,
        embedding_service: EmbeddingService | None = None,
    ):
        """
        Initialize Pinecone store.
        
        Args:
            index_name: Pinecone index name
            embedding_service: Embedding service instance
        """
        self.index_name = index_name or os.getenv("PINECONE_INDEX_NAME", "halter-suppor")
        self.embedding_service = embedding_service or EmbeddingService()
        
        # Initialize Pinecone client
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("PINECONE_API_KEY environment variable is required")
        
        self.pc = Pinecone(api_key=api_key)
        self.index = None
        
    def create_index_if_not_exists(
        self,
        dimension: int | None = None,
        metric: str = "cosine",
        cloud: str = "aws",
        region: str = "us-east-1",
    ) -> None:
        """
        Create Pinecone index if it doesn't exist.
        
        Args:
            dimension: Vector dimension (default: from embedding service)
            metric: Distance metric (cosine, euclidean, dotproduct)
            cloud: Cloud provider
            region: Cloud region
        """
        dimension = dimension or self.embedding_service.dimensions
        
        # Check if index exists
        existing_indexes = [idx.name for idx in self.pc.list_indexes()]
        
        if self.index_name not in existing_indexes:
            logger.info(f"Creating Pinecone index: {self.index_name}")
            
            self.pc.create_index(
                name=self.index_name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(
                    cloud=cloud,
                    region=region,
                ),
            )
            
            # Wait for index to be ready
            while not self.pc.describe_index(self.index_name).status["ready"]:
                logger.info("Waiting for index to be ready...")
                time.sleep(1)
            
            logger.info(f"Index {self.index_name} created successfully")
        else:
            logger.info(f"Index {self.index_name} already exists")
        
        # Connect to index
        self.index = self.pc.Index(self.index_name)
    
    def upsert_chunks(
        self,
        chunks: list[dict[str, Any]],
        namespace: str = "",
        batch_size: int = 100,
    ) -> dict[str, int]:
        """
        Upsert document chunks to Pinecone.
        
        Args:
            chunks: List of chunk dictionaries with 'id', 'content', 'metadata'
            namespace: Pinecone namespace for organizing vectors
            batch_size: Number of vectors to upsert per batch
            
        Returns:
            Statistics dictionary
        """
        if self.index is None:
            self.create_index_if_not_exists()
        
        # Prepare vectors
        vectors_to_upsert = []
        total_chunks = len(chunks)
        
        logger.info(f"Processing {total_chunks} chunks for embedding...")
        
        for i, chunk in enumerate(chunks):
            # Generate embedding
            embedding = self.embedding_service.embed_text(chunk["content"])
            
            # Prepare metadata (Pinecone has metadata size limits ~40KB)
            # Store full content up to 10000 chars for better retrieval
            raw_metadata = {
                **chunk["metadata"],
                "content": chunk["content"][:10000],  # Store more content
                "content_preview": chunk["content"][:300],
            }
            
            # Clean metadata: remove None values and convert types for Pinecone compatibility
            metadata = {}
            for key, value in raw_metadata.items():
                # Skip None values (Pinecone doesn't accept null)
                if value is None:
                    continue
                # Convert list fields appropriately
                if isinstance(value, list):
                    metadata[key] = value if all(isinstance(v, str) for v in value) else str(value)
                else:
                    metadata[key] = value
            
            vectors_to_upsert.append({
                "id": chunk["id"],
                "values": embedding,
                "metadata": metadata,
            })
            
            if (i + 1) % 10 == 0:
                logger.info(f"Embedded {i + 1}/{total_chunks} chunks")
        
        # Upsert in batches
        total_upserted = 0
        
        for i in range(0, len(vectors_to_upsert), batch_size):
            batch = vectors_to_upsert[i:i + batch_size]
            self.index.upsert(vectors=batch, namespace=namespace)
            total_upserted += len(batch)
            logger.info(f"Upserted batch {i // batch_size + 1}, total: {total_upserted}/{total_chunks}")
        
        logger.info(f"✅ Successfully upserted {total_upserted} vectors to Pinecone")
        
        return {
            "total_chunks": total_chunks,
            "total_upserted": total_upserted,
            "index_name": self.index_name,
            "namespace": namespace,
        }
    
    def query(
        self,
        query_text: str,
        top_k: int = 5,
        namespace: str = "",
        filter: dict | None = None,
        include_metadata: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Query similar documents from Pinecone.
        
        Args:
            query_text: Query text
            top_k: Number of results to return
            namespace: Pinecone namespace
            filter: Metadata filter dictionary
            include_metadata: Whether to include metadata in results
            
        Returns:
            List of matching documents with scores
        """
        if self.index is None:
            self.create_index_if_not_exists()
        
        # Generate query embedding
        query_embedding = self.embedding_service.embed_text(query_text)
        
        # Query Pinecone
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            namespace=namespace,
            filter=filter,
            include_metadata=include_metadata,
        )
        
        # Format results
        formatted_results = []
        for match in results.matches:
            formatted_results.append({
                "id": match.id,
                "score": match.score,
                "metadata": match.metadata if include_metadata else {},
            })
        
        return formatted_results
    
    def delete_namespace(self, namespace: str = "") -> None:
        """Delete all vectors in a namespace."""
        if self.index is None:
            self.create_index_if_not_exists()
        
        self.index.delete(delete_all=True, namespace=namespace)
        logger.info(f"Deleted all vectors in namespace: {namespace or 'default'}")
    
    def get_index_stats(self) -> dict[str, Any]:
        """Get index statistics."""
        if self.index is None:
            self.create_index_if_not_exists()
        
        stats = self.index.describe_index_stats()
        return {
            "total_vector_count": stats.total_vector_count,
            "dimension": stats.dimension,
            "namespaces": dict(stats.namespaces) if stats.namespaces else {},
        }

