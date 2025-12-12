"""
Hybrid Search Module
Combines semantic search (dense) with keyword search (sparse/BM25).
"""

import os
import re
from typing import Any
from dataclasses import dataclass

from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

from src.vectorstore import EmbeddingService


@dataclass
class HybridSearchResult:
    """Hybrid search result."""
    id: str
    content: str
    score: float
    metadata: dict[str, Any]
    
    def __str__(self) -> str:
        return f"[{self.score:.4f}] {self.content[:100]}..."


class HybridSearcher:
    """
    Hybrid Searcher combining dense (semantic) and sparse (BM25) retrieval.
    
    Methods:
    - Pinecone native sparse-dense vectors
    - RRF (Reciprocal Rank Fusion) for result merging
    """
    
    def __init__(
        self,
        index_name: str | None = None,
        namespace: str = "agents-doc",
    ):
        """
        Initialize Hybrid Searcher.
        
        Args:
            index_name: Pinecone index name
            namespace: Default namespace
        """
        self.namespace = namespace
        self.index_name = index_name or os.getenv("PINECONE_INDEX_NAME", "halter-support")
        
        # Initialize Pinecone
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("PINECONE_API_KEY is required")
        
        self.pc = Pinecone(api_key=api_key)
        self.index = self.pc.Index(self.index_name)
        
        # Initialize embedding service (dense vectors)
        self.embedding_service = EmbeddingService()
        
        # Initialize BM25 encoder (sparse vectors)
        self.bm25 = BM25Encoder.default()
    
    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        alpha: float = 0.5,
        filter: dict | None = None,
    ) -> list[HybridSearchResult]:
        """
        Hybrid search combining semantic and keyword matching.
        
        Uses RRF (Reciprocal Rank Fusion) to merge results from:
        - Dense vector search (semantic similarity)
        - Sparse vector search (BM25 keyword matching)
        
        Args:
            query: Search query
            top_k: Number of results to return
            alpha: Weight for dense vs sparse (0=sparse only, 1=dense only, 0.5=balanced)
            filter: Optional metadata filter
            
        Returns:
            List of HybridSearchResult
        """
        # Get more results for fusion
        fetch_k = top_k * 3
        
        # Dense search (semantic)
        dense_results = self._dense_search(query, fetch_k, filter)
        
        # Sparse search (keyword/BM25) - using dense search with keyword-enhanced query
        sparse_results = self._keyword_enhanced_search(query, fetch_k, filter)
        
        # Fuse results using RRF
        fused_results = self._rrf_fusion(
            dense_results=dense_results,
            sparse_results=sparse_results,
            alpha=alpha,
            top_k=top_k,
        )
        
        return fused_results
    
    def _dense_search(
        self,
        query: str,
        top_k: int,
        filter: dict | None = None,
    ) -> list[dict]:
        """Pure semantic search using dense vectors."""
        query_embedding = self.embedding_service.embed_text(query)
        
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            namespace=self.namespace,
            filter=filter,
            include_metadata=True,
        )
        
        return [
            {
                "id": match.id,
                "score": match.score,
                "metadata": match.metadata or {},
            }
            for match in results.matches
        ]
    
    def _keyword_enhanced_search(
        self,
        query: str,
        top_k: int,
        filter: dict | None = None,
    ) -> list[dict]:
        """
        Keyword-enhanced search.
        Extracts key terms and boosts results containing them.
        """
        # Extract keywords from query
        keywords = self._extract_keywords(query)
        
        # Build enhanced query with keywords emphasized
        enhanced_query = f"{query} {' '.join(keywords * 2)}"
        
        query_embedding = self.embedding_service.embed_text(enhanced_query)
        
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            namespace=self.namespace,
            filter=filter,
            include_metadata=True,
        )
        
        # Boost scores based on keyword matches in content
        boosted_results = []
        for match in results.matches:
            metadata = match.metadata or {}
            content = metadata.get("content", "").lower()
            
            # Calculate keyword match bonus
            keyword_bonus = sum(
                0.1 for kw in keywords 
                if kw.lower() in content
            )
            
            boosted_results.append({
                "id": match.id,
                "score": match.score + keyword_bonus,
                "metadata": metadata,
            })
        
        # Re-sort by boosted score
        boosted_results.sort(key=lambda x: x["score"], reverse=True)
        
        return boosted_results[:top_k]
    
    def _extract_keywords(self, query: str) -> list[str]:
        """Extract important keywords from query."""
        # Remove common stopwords
        stopwords = {
            "what", "is", "are", "how", "do", "does", "the", "a", "an",
            "in", "on", "at", "to", "for", "of", "and", "or", "it",
            "this", "that", "with", "from", "by", "as", "be", "was",
            "were", "been", "being", "have", "has", "had", "having",
        }
        
        # Tokenize and filter
        words = re.findall(r'\b[a-zA-Z]{2,}\b', query.lower())
        keywords = [w for w in words if w not in stopwords]
        
        return keywords
    
    def _rrf_fusion(
        self,
        dense_results: list[dict],
        sparse_results: list[dict],
        alpha: float = 0.5,
        top_k: int = 5,
        k: int = 60,
    ) -> list[HybridSearchResult]:
        """
        Reciprocal Rank Fusion (RRF) to combine results.
        
        RRF Score = Σ 1 / (k + rank)
        
        Args:
            dense_results: Results from dense search
            sparse_results: Results from sparse search
            alpha: Weight for dense results (1-alpha for sparse)
            top_k: Number of final results
            k: RRF constant (typically 60)
            
        Returns:
            Fused and ranked results
        """
        # Calculate RRF scores
        rrf_scores: dict[str, float] = {}
        result_data: dict[str, dict] = {}
        
        # Process dense results
        for rank, result in enumerate(dense_results, 1):
            doc_id = result["id"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + alpha / (k + rank)
            result_data[doc_id] = result
        
        # Process sparse results
        for rank, result in enumerate(sparse_results, 1):
            doc_id = result["id"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + (1 - alpha) / (k + rank)
            if doc_id not in result_data:
                result_data[doc_id] = result
        
        # Sort by RRF score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        
        # Build final results
        final_results = []
        for doc_id in sorted_ids[:top_k]:
            data = result_data[doc_id]
            metadata = data.get("metadata", {})
            
            final_results.append(HybridSearchResult(
                id=doc_id,
                content=metadata.get("content", ""),
                score=rrf_scores[doc_id],
                metadata=metadata,
            ))
        
        return final_results
    
    def search_with_reranking(
        self,
        query: str,
        top_k: int = 5,
        initial_k: int = 20,
        filter: dict | None = None,
    ) -> list[HybridSearchResult]:
        """
        Search with simple reranking based on query term overlap.
        
        Args:
            query: Search query
            top_k: Final number of results
            initial_k: Initial retrieval count
            filter: Optional metadata filter
            
        Returns:
            Reranked results
        """
        # Initial retrieval
        initial_results = self._dense_search(query, initial_k, filter)
        
        # Extract query keywords
        query_keywords = set(self._extract_keywords(query))
        
        # Rerank based on keyword overlap
        reranked = []
        for result in initial_results:
            metadata = result.get("metadata", {})
            content = metadata.get("content", "").lower()
            
            # Count keyword matches
            content_words = set(re.findall(r'\b[a-zA-Z]{2,}\b', content))
            overlap = len(query_keywords & content_words)
            
            # Combined score
            rerank_score = result["score"] * 0.7 + (overlap / max(len(query_keywords), 1)) * 0.3
            
            reranked.append({
                **result,
                "rerank_score": rerank_score,
            })
        
        # Sort by rerank score
        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
        
        # Build results
        return [
            HybridSearchResult(
                id=r["id"],
                content=r["metadata"].get("content", ""),
                score=r["rerank_score"],
                metadata=r["metadata"],
            )
            for r in reranked[:top_k]
        ]


# Convenience function
def hybrid_search(
    query: str,
    top_k: int = 5,
    alpha: float = 0.5,
    namespace: str = "agents-doc",
) -> list[HybridSearchResult]:
    """
    Quick hybrid search function.
    
    Args:
        query: Search query
        top_k: Number of results
        alpha: Dense/sparse weight (0.5 = balanced)
        namespace: Pinecone namespace
        
    Returns:
        List of search results
    """
    searcher = HybridSearcher(namespace=namespace)
    return searcher.hybrid_search(query, top_k=top_k, alpha=alpha)

