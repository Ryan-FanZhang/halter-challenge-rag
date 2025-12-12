"""
Reranker Module
Supports LLM Reranking (recommended) and Cross-encoder Reranking.
"""

import os
import json
from typing import Any
from dataclasses import dataclass

from openai import OpenAI
from dotenv import load_dotenv
from loguru import logger

from src.prompts import LLM_RERANK_SYSTEM_PROMPT, format_rerank_user_prompt

load_dotenv()


@dataclass
class RerankResult:
    """Reranked search result."""
    id: str
    content: str
    original_score: float
    rerank_score: float
    final_score: float
    reasoning: str
    metadata: dict[str, Any]


class LLMReranker:
    """
    LLM-based Reranker using GPT-4o-mini.
    
    Advantages:
    - High quality relevance scoring
    - Low cost (~$0.01 per question)
    - Fast with batch processing
    """
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        vector_weight: float = 0.3,
        llm_weight: float = 0.7,
    ):
        """
        Initialize LLM Reranker.
        
        Args:
            model: OpenAI model to use
            vector_weight: Weight for original vector score
            llm_weight: Weight for LLM relevance score
        """
        self.model = model
        self.vector_weight = vector_weight
        self.llm_weight = llm_weight
        self.client = OpenAI()
    
    def rerank(
        self,
        query: str,
        results: list[dict],
        top_k: int | None = None,
        batch_size: int = 3,
    ) -> list[RerankResult]:
        """
        Rerank search results using LLM.
        
        Args:
            query: Original search query
            results: List of search results with 'id', 'content', 'score', 'metadata'
            top_k: Number of results to return (None = all)
            batch_size: Number of chunks to score per LLM call
            
        Returns:
            Reranked results sorted by final score
        """
        if not results:
            return []
        
        reranked = []
        
        # Process in batches for efficiency
        for i in range(0, len(results), batch_size):
            batch = results[i:i + batch_size]
            batch_scores = self._score_batch(query, batch)
            
            for result, (score, reasoning) in zip(batch, batch_scores):
                original_score = result.get("score", 0)
                
                # Calculate final score with weighted average
                final_score = (
                    self.vector_weight * original_score +
                    self.llm_weight * score
                )
                
                reranked.append(RerankResult(
                    id=result.get("id", ""),
                    content=result.get("content", result.get("metadata", {}).get("content", "")),
                    original_score=original_score,
                    rerank_score=score,
                    final_score=final_score,
                    reasoning=reasoning,
                    metadata=result.get("metadata", {}),
                ))
        
        # Sort by final score
        reranked.sort(key=lambda x: x.final_score, reverse=True)
        
        if top_k:
            reranked = reranked[:top_k]
        
        return reranked
    
    def _score_batch(
        self,
        query: str,
        batch: list[dict],
    ) -> list[tuple[float, str]]:
        """Score a batch of chunks using LLM."""
        
        # Build chunks text
        chunks_text = ""
        for i, result in enumerate(batch, 1):
            content = result.get("content", result.get("metadata", {}).get("content", ""))
            chunks_text += f"\n--- Chunk {i} ---\n{content[:2000]}\n"
        
        # Build response format
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "relevance_scores",
                "schema": {
                    "type": "object",
                    "properties": {
                        "scores": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "chunk_number": {"type": "integer"},
                                    "reasoning": {"type": "string"},
                                    "relevance_score": {"type": "number"}
                                },
                                "required": ["chunk_number", "reasoning", "relevance_score"]
                            }
                        }
                    },
                    "required": ["scores"]
                }
            }
        }
        
        # Use centralized prompt
        user_message = format_rerank_user_prompt(query, chunks_text, len(batch))

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": LLM_RERANK_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message}
                ],
                response_format=response_format,
                temperature=0,
            )
            
            result = json.loads(response.choices[0].message.content)
            scores = result.get("scores", [])
            
            # Map scores to batch order
            score_map = {s["chunk_number"]: (s["relevance_score"], s["reasoning"]) for s in scores}
            
            return [
                score_map.get(i, (0.5, "No score available"))
                for i in range(1, len(batch) + 1)
            ]
            
        except Exception as e:
            logger.error(f"LLM reranking error: {e}")
            # Return neutral scores on error
            return [(0.5, f"Error: {e}")] * len(batch)


class CrossEncoderReranker:
    """
    Cross-encoder Reranker using Jina Reranker API.
    
    Requires: JINA_API_KEY environment variable
    Get API key at: https://jina.ai/reranker/
    """
    
    def __init__(
        self,
        model: str = "jina-reranker-v2-base-multilingual",
        vector_weight: float = 0.3,
        rerank_weight: float = 0.7,
    ):
        """
        Initialize Cross-encoder Reranker.
        
        Args:
            model: Jina reranker model
            vector_weight: Weight for original vector score
            rerank_weight: Weight for cross-encoder score
        """
        self.model = model
        self.vector_weight = vector_weight
        self.rerank_weight = rerank_weight
        self.api_key = os.getenv("JINA_API_KEY")
        
        if not self.api_key:
            logger.warning("JINA_API_KEY not set. Cross-encoder reranking will not work.")
    
    def rerank(
        self,
        query: str,
        results: list[dict],
        top_k: int | None = None,
    ) -> list[RerankResult]:
        """
        Rerank search results using Jina Cross-encoder.
        
        Args:
            query: Original search query
            results: List of search results
            top_k: Number of results to return
            
        Returns:
            Reranked results
        """
        if not self.api_key:
            raise ValueError("JINA_API_KEY is required for cross-encoder reranking")
        
        if not results:
            return []
        
        import requests
        
        # Prepare documents
        documents = []
        for result in results:
            content = result.get("content", result.get("metadata", {}).get("content", ""))
            documents.append(content[:4000])  # Jina has input limits
        
        # Call Jina API
        try:
            response = requests.post(
                "https://api.jina.ai/v1/rerank",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "query": query,
                    "documents": documents,
                    "top_n": top_k or len(results),
                }
            )
            response.raise_for_status()
            data = response.json()
            
            # Build reranked results
            reranked = []
            for item in data.get("results", []):
                idx = item["index"]
                original_result = results[idx]
                original_score = original_result.get("score", 0)
                rerank_score = item["relevance_score"]
                
                final_score = (
                    self.vector_weight * original_score +
                    self.rerank_weight * rerank_score
                )
                
                reranked.append(RerankResult(
                    id=original_result.get("id", ""),
                    content=documents[idx],
                    original_score=original_score,
                    rerank_score=rerank_score,
                    final_score=final_score,
                    reasoning=f"Cross-encoder score: {rerank_score:.4f}",
                    metadata=original_result.get("metadata", {}),
                ))
            
            return reranked
            
        except Exception as e:
            logger.error(f"Jina reranking error: {e}")
            raise


class HybridReranker:
    """
    Combined reranker with fallback support.
    Uses LLM reranking by default, with optional cross-encoder.
    """
    
    def __init__(
        self,
        use_cross_encoder: bool = False,
        vector_weight: float = 0.3,
        rerank_weight: float = 0.7,
    ):
        """
        Initialize Hybrid Reranker.
        
        Args:
            use_cross_encoder: Use Jina cross-encoder instead of LLM
            vector_weight: Weight for original vector score
            rerank_weight: Weight for reranker score
        """
        if use_cross_encoder:
            self.reranker = CrossEncoderReranker(
                vector_weight=vector_weight,
                rerank_weight=rerank_weight,
            )
        else:
            self.reranker = LLMReranker(
                vector_weight=vector_weight,
                llm_weight=rerank_weight,
            )
    
    def rerank(
        self,
        query: str,
        results: list[dict],
        top_k: int | None = None,
    ) -> list[RerankResult]:
        """Rerank results using configured reranker."""
        return self.reranker.rerank(query, results, top_k)

