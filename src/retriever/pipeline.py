"""
RAG Search Pipeline
Complete pipeline with retrieval + reranking.
"""

import os
from typing import Any
from dataclasses import dataclass

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

from .search import RAGSearcher
from .hybrid_search import HybridSearcher
from .reranker import LLMReranker, CrossEncoderReranker, RerankResult


@dataclass
class PipelineConfig:
    """Configuration for RAG pipeline."""
    
    # Retrieval settings
    namespace: str = "agents-doc"
    initial_k: int = 20  # Retrieve more for reranking
    final_k: int = 5     # Final results after reranking
    
    # Hybrid search settings
    use_hybrid: bool = True
    alpha: float = 0.5
    
    # Reranking settings
    use_reranking: bool = True
    reranker_type: str = "llm"  # "llm" or "cross_encoder"
    vector_weight: float = 0.3
    rerank_weight: float = 0.7
    batch_size: int = 3


class RAGPipeline:
    """
    Complete RAG Search Pipeline.
    
    Pipeline stages:
    1. Initial retrieval (vector search or hybrid search)
    2. Reranking (LLM or cross-encoder)
    3. Final ranking and filtering
    """
    
    def __init__(self, config: PipelineConfig | None = None):
        """
        Initialize RAG Pipeline.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config or PipelineConfig()
        
        # Initialize retrievers
        self.basic_searcher = RAGSearcher(namespace=self.config.namespace)
        self.hybrid_searcher = HybridSearcher(namespace=self.config.namespace)
        
        # Initialize reranker
        if self.config.reranker_type == "cross_encoder":
            self.reranker = CrossEncoderReranker(
                vector_weight=self.config.vector_weight,
                rerank_weight=self.config.rerank_weight,
            )
        else:
            self.reranker = LLMReranker(
                vector_weight=self.config.vector_weight,
                llm_weight=self.config.rerank_weight,
            )
    
    def search(
        self,
        query: str,
        top_k: int | None = None,
        filter: dict | None = None,
        use_hybrid: bool | None = None,
        use_reranking: bool | None = None,
    ) -> list[RerankResult]:
        """
        Execute full search pipeline.
        
        Args:
            query: Search query
            top_k: Number of final results (default: config.final_k)
            filter: Optional metadata filter
            use_hybrid: Override hybrid setting
            use_reranking: Override reranking setting
            
        Returns:
            List of reranked results
        """
        top_k = top_k or self.config.final_k
        use_hybrid = use_hybrid if use_hybrid is not None else self.config.use_hybrid
        use_reranking = use_reranking if use_reranking is not None else self.config.use_reranking
        
        # Stage 1: Initial retrieval
        initial_k = self.config.initial_k if use_reranking else top_k
        
        logger.info(f"Stage 1: Retrieving top {initial_k} results...")
        
        if use_hybrid:
            initial_results = self.hybrid_searcher.hybrid_search(
                query=query,
                top_k=initial_k,
                alpha=self.config.alpha,
                filter=filter,
            )
            # Convert to dict format for reranker
            results_for_rerank = [
                {
                    "id": r.id,
                    "content": r.content,
                    "score": r.score,
                    "metadata": r.metadata,
                }
                for r in initial_results
            ]
        else:
            initial_results = self.basic_searcher.search(
                query=query,
                top_k=initial_k,
                filter=filter,
            )
            results_for_rerank = [
                {
                    "id": r.id,
                    "content": r.content,
                    "score": r.score,
                    "metadata": r.metadata,
                }
                for r in initial_results
            ]
        
        logger.info(f"Retrieved {len(results_for_rerank)} results")
        
        # Stage 2: Reranking
        if use_reranking and results_for_rerank:
            logger.info(f"Stage 2: Reranking with {self.config.reranker_type}...")
            
            reranked_results = self.reranker.rerank(
                query=query,
                results=results_for_rerank,
                top_k=top_k,
            )
            
            logger.info(f"Reranking complete. Top {len(reranked_results)} results.")
            return reranked_results
        else:
            # No reranking, convert to RerankResult format
            return [
                RerankResult(
                    id=r["id"],
                    content=r["content"],
                    original_score=r["score"],
                    rerank_score=r["score"],
                    final_score=r["score"],
                    reasoning="No reranking applied",
                    metadata=r["metadata"],
                )
                for r in results_for_rerank[:top_k]
            ]
    
    def search_simple(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RerankResult]:
        """
        Simple search with default settings.
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            Reranked results
        """
        return self.search(query, top_k=top_k)


# Convenience function
def search_with_reranking(
    query: str,
    top_k: int = 5,
    initial_k: int = 20,
    namespace: str = "agents-doc",
) -> list[RerankResult]:
    """
    Quick search with LLM reranking.
    
    Args:
        query: Search query
        top_k: Final number of results
        initial_k: Initial retrieval count
        namespace: Pinecone namespace
        
    Returns:
        Reranked search results
    """
    config = PipelineConfig(
        namespace=namespace,
        initial_k=initial_k,
        final_k=top_k,
        use_hybrid=True,
        use_reranking=True,
    )
    pipeline = RAGPipeline(config)
    return pipeline.search(query)

