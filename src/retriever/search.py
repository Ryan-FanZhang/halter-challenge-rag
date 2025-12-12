"""
RAG Search Module
Provides multiple search strategies for document retrieval.
"""

import os
from typing import Any
from dataclasses import dataclass

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

from src.vectorstore import PineconeStore, EmbeddingService


@dataclass
class SearchResult:
    """Search result with content and metadata."""
    id: str
    content: str
    score: float
    metadata: dict[str, Any]
    
    def __str__(self) -> str:
        return f"[{self.score:.4f}] {self.content[:100]}..."


class RAGSearcher:
    """
    RAG Searcher with multiple search strategies.
    
    Supports:
    - Semantic search (vector similarity)
    - Filtered search (with metadata filters)
    - Multi-query search (query expansion)
    """
    
    def __init__(
        self,
        index_name: str | None = None,
        namespace: str = "agents-doc",
    ):
        """
        Initialize RAG Searcher.
        
        Args:
            index_name: Pinecone index name
            namespace: Default namespace for queries
        """
        self.namespace = namespace
        self.store = PineconeStore(index_name=index_name)
        self.store.create_index_if_not_exists()
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        filter: dict | None = None,
        include_content: bool = True,
    ) -> list[SearchResult]:
        """
        Basic semantic search.
        
        Args:
            query: Search query text
            top_k: Number of results to return
            filter: Optional metadata filter
            include_content: Whether to include full content
            
        Returns:
            List of SearchResult objects
        """
        results = self.store.query(
            query_text=query,
            top_k=top_k,
            namespace=self.namespace,
            filter=filter,
            include_metadata=True,
        )
        
        return self._format_results(results, include_content)
    
    def search_by_content_type(
        self,
        query: str,
        content_type: str,
        top_k: int = 5,
    ) -> list[SearchResult]:
        """
        Search filtered by content type.
        
        Args:
            query: Search query text
            content_type: Content type filter (definition, example, workflow, etc.)
            top_k: Number of results
            
        Returns:
            List of SearchResult objects
        """
        return self.search(
            query=query,
            top_k=top_k,
            filter={"content_type": content_type},
        )
    
    def search_by_workflow(
        self,
        query: str,
        workflow_type: str,
        top_k: int = 5,
    ) -> list[SearchResult]:
        """
        Search filtered by workflow type.
        
        Args:
            query: Search query text
            workflow_type: Workflow type (routing, parallelization, agents, etc.)
            top_k: Number of results
            
        Returns:
            List of SearchResult objects
        """
        return self.search(
            query=query,
            top_k=top_k,
            filter={"workflow_type": workflow_type},
        )
    
    def search_by_topic(
        self,
        query: str,
        topic: str,
        top_k: int = 5,
    ) -> list[SearchResult]:
        """
        Search filtered by topic.
        
        Args:
            query: Search query text
            topic: Topic to filter by
            top_k: Number of results
            
        Returns:
            List of SearchResult objects
        """
        # Pinecone supports $in operator for array fields
        return self.search(
            query=query,
            top_k=top_k,
            filter={"topics": {"$in": [topic]}},
        )
    
    def search_with_diagrams(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[SearchResult]:
        """
        Search only chunks that contain diagrams.
        
        Args:
            query: Search query text
            top_k: Number of results
            
        Returns:
            List of SearchResult objects
        """
        return self.search(
            query=query,
            top_k=top_k,
            filter={"has_diagram": True},
        )
    
    def search_advanced(
        self,
        query: str,
        top_k: int = 5,
        content_type: str | None = None,
        workflow_type: str | None = None,
        topics: list[str] | None = None,
        has_diagram: bool | None = None,
        has_code: bool | None = None,
        section_level: int | None = None,
    ) -> list[SearchResult]:
        """
        Advanced search with multiple filters.
        
        Args:
            query: Search query text
            top_k: Number of results
            content_type: Filter by content type
            workflow_type: Filter by workflow type
            topics: Filter by topics (any match)
            has_diagram: Filter by diagram presence
            has_code: Filter by code presence
            section_level: Filter by section level
            
        Returns:
            List of SearchResult objects
        """
        # Build filter dynamically
        filter_conditions = {}
        
        if content_type:
            filter_conditions["content_type"] = content_type
        if workflow_type:
            filter_conditions["workflow_type"] = workflow_type
        if topics:
            filter_conditions["topics"] = {"$in": topics}
        if has_diagram is not None:
            filter_conditions["has_diagram"] = has_diagram
        if has_code is not None:
            filter_conditions["has_code"] = has_code
        if section_level is not None:
            filter_conditions["section_level"] = section_level
        
        return self.search(
            query=query,
            top_k=top_k,
            filter=filter_conditions if filter_conditions else None,
        )
    
    def _format_results(
        self,
        results: list[dict],
        include_content: bool = True,
    ) -> list[SearchResult]:
        """Format raw results into SearchResult objects."""
        formatted = []
        
        for result in results:
            metadata = result.get("metadata", {})
            content = metadata.get("content", "") if include_content else ""
            
            formatted.append(SearchResult(
                id=result["id"],
                content=content,
                score=result["score"],
                metadata=metadata,
            ))
        
        return formatted
    
    def get_stats(self) -> dict[str, Any]:
        """Get index statistics."""
        return self.store.get_index_stats()

