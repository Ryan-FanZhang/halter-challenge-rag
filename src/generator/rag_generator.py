"""
RAG Generator Module
Augmentation + Generation: Combines retrieved context with query to generate answers.
"""

import os
import json
import time
from typing import Any, Literal
from dataclasses import dataclass, field

from openai import OpenAI
from pydantic import BaseModel
from dotenv import load_dotenv
from loguru import logger

from src.prompts import (
    RAGAnswerPrompt,
    QueryExpansionPrompt,
    ContextCompressionPrompt,
    format_rag_answer_prompt,
    format_query_expansion_prompt,
    format_context_compression_prompt,
)

load_dotenv()


@dataclass
class GeneratorConfig:
    """Configuration for RAG Generator."""
    
    # Model settings
    model: str = "gpt-4o-mini"
    temperature: float = 0.1
    max_tokens: int = 2000
    
    # Context settings
    max_context_length: int = 8000  # chars
    compress_context: bool = False
    
    # Response settings
    use_structured_output: bool = True
    include_reasoning: bool = True


@dataclass
class Citation:
    """Citation with source content."""
    source_id: str
    section: str
    content_excerpt: str  # Actual quoted text
    relevance_note: str = ""  # Why this was cited


@dataclass
class TimingInfo:
    """Timing information for each pipeline step."""
    retrieval_ms: float = 0
    reranking_ms: float = 0
    augmentation_ms: float = 0
    generation_ms: float = 0
    total_ms: float = 0
    
    def __str__(self) -> str:
        return (
            f"Retrieval: {self.retrieval_ms:.0f}ms | "
            f"Reranking: {self.reranking_ms:.0f}ms | "
            f"Generation: {self.generation_ms:.0f}ms | "
            f"Total: {self.total_ms:.0f}ms"
        )


@dataclass
class RAGAnswer:
    """Structured RAG answer with citations and timing."""
    question: str
    answer: str
    reasoning: str = ""
    confidence: str = "medium"
    relevant_sections: list[str] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)
    sources: list[dict] = field(default_factory=list)
    timing: TimingInfo = field(default_factory=TimingInfo)
    
    def __str__(self) -> str:
        return f"[{self.confidence}] {self.answer[:200]}..."
    
    def format_citations(self) -> str:
        """Format citations for display."""
        if not self.citations:
            return "No citations"
        
        lines = []
        for i, c in enumerate(self.citations, 1):
            lines.append(f"\n[{i}] {c.section}")
            lines.append(f"    \"{c.content_excerpt[:200]}{'...' if len(c.content_excerpt) > 200 else ''}\"")
            if c.relevance_note:
                lines.append(f"    → {c.relevance_note}")
        
        return "\n".join(lines)


class RAGGenerator:
    """
    RAG Generator for answer generation.
    
    Pipeline:
    1. (Optional) Query expansion
    2. Context augmentation (formatting)
    3. (Optional) Context compression
    4. LLM answer generation
    """
    
    def __init__(self, config: GeneratorConfig | None = None):
        """
        Initialize RAG Generator.
        
        Args:
            config: Generator configuration
        """
        self.config = config or GeneratorConfig()
        self.client = OpenAI()
    
    def generate(
        self,
        question: str,
        contexts: list[dict[str, Any]],
        use_structured: bool | None = None,
    ) -> RAGAnswer:
        """
        Generate answer from question and retrieved contexts.
        
        Args:
            question: User question
            contexts: List of retrieved context dicts with 'content' and 'metadata'
            use_structured: Override structured output setting
            
        Returns:
            RAGAnswer object
        """
        use_structured = use_structured if use_structured is not None else self.config.use_structured_output
        
        # Step 1: Format context
        formatted_context = self._format_contexts(contexts)
        
        # Step 2: Compress if needed
        if self.config.compress_context and len(formatted_context) > self.config.max_context_length:
            logger.info("Compressing context...")
            formatted_context = self._compress_context(question, formatted_context)
        
        # Step 3: Generate answer
        if use_structured:
            return self._generate_structured(question, formatted_context, contexts)
        else:
            return self._generate_simple(question, formatted_context, contexts)
    
    def generate_with_expansion(
        self,
        question: str,
        retriever_func,
        top_k: int = 5,
    ) -> RAGAnswer:
        """
        Generate answer with query expansion for better retrieval.
        
        Args:
            question: User question
            retriever_func: Function that takes query and returns contexts
            top_k: Number of results per query
            
        Returns:
            RAGAnswer object
        """
        # Expand query
        expanded_queries = self._expand_query(question)
        logger.info(f"Expanded to {len(expanded_queries)} queries")
        
        # Retrieve for each query
        all_contexts = []
        seen_ids = set()
        
        for query in [question] + expanded_queries:
            results = retriever_func(query, top_k=top_k)
            for r in results:
                if r.get("id") not in seen_ids:
                    seen_ids.add(r.get("id"))
                    all_contexts.append(r)
        
        logger.info(f"Retrieved {len(all_contexts)} unique contexts")
        
        # Generate answer
        return self.generate(question, all_contexts)
    
    def _format_contexts(self, contexts: list[dict]) -> str:
        """Format retrieved contexts into a single string."""
        formatted_parts = []
        
        for i, ctx in enumerate(contexts, 1):
            content = ctx.get("content", "")
            metadata = ctx.get("metadata", {})
            
            # Build context header
            section = " > ".join(metadata.get("section_hierarchy", []))
            header = f"[Source {i}]"
            if section:
                header += f" {section}"
            
            # Add metadata tags
            tags = []
            if metadata.get("content_type"):
                tags.append(f"type:{metadata['content_type']}")
            if metadata.get("workflow_type"):
                tags.append(f"workflow:{metadata['workflow_type']}")
            
            if tags:
                header += f" ({', '.join(tags)})"
            
            formatted_parts.append(f"{header}\n{content}")
        
        return "\n\n---\n\n".join(formatted_parts)
    
    def _compress_context(self, question: str, context: str) -> str:
        """Compress context using LLM."""
        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": ContextCompressionPrompt.system_prompt},
                    {"role": "user", "content": format_context_compression_prompt(question, context)}
                ],
                temperature=0,
                max_tokens=self.config.max_tokens,
            )
            
            result = response.choices[0].message.content
            
            # Try to parse structured output
            try:
                data = json.loads(result)
                return data.get("compressed_context", result)
            except:
                return result
                
        except Exception as e:
            logger.warning(f"Context compression failed: {e}")
            # Fallback: simple truncation
            return context[:self.config.max_context_length]
    
    def _expand_query(self, question: str) -> list[str]:
        """Expand query into multiple search queries."""
        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": QueryExpansionPrompt.system_prompt_with_schema},
                    {"role": "user", "content": format_query_expansion_prompt(question)}
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
            )
            
            result = json.loads(response.choices[0].message.content)
            return result.get("queries", [])
            
        except Exception as e:
            logger.warning(f"Query expansion failed: {e}")
            return []
    
    def _generate_structured(
        self,
        question: str,
        context: str,
        original_contexts: list[dict],
    ) -> RAGAnswer:
        """Generate structured answer using JSON schema with citations."""
        
        # Build response format for structured output with citations
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "rag_answer",
                "schema": {
                    "type": "object",
                    "properties": {
                        "step_by_step_analysis": {"type": "string"},
                        "reasoning_summary": {"type": "string"},
                        "relevant_sections": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "citations": {
                            "type": "array",
                            "description": "Direct quotes from the context that support the answer",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "source_number": {
                                        "type": "integer",
                                        "description": "Source number (1, 2, 3...) from the context"
                                    },
                                    "quote": {
                                        "type": "string",
                                        "description": "Exact quote from the source (50-150 chars)"
                                    },
                                    "relevance": {
                                        "type": "string",
                                        "description": "Brief note on why this quote is relevant"
                                    }
                                },
                                "required": ["source_number", "quote", "relevance"]
                            }
                        },
                        "confidence": {
                            "type": "string",
                            "enum": ["high", "medium", "low"]
                        },
                        "final_answer": {"type": "string"}
                    },
                    "required": ["step_by_step_analysis", "reasoning_summary", "relevant_sections", "citations", "confidence", "final_answer"]
                }
            }
        }
        
        start_time = time.time()
        
        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": RAGAnswerPrompt.system_prompt_with_schema},
                    {"role": "user", "content": format_rag_answer_prompt(context, question)}
                ],
                response_format=response_format,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            
            generation_time = (time.time() - start_time) * 1000
            
            result = json.loads(response.choices[0].message.content)
            
            # Debug: log what we got
            logger.debug(f"LLM response keys: {result.keys()}")
            logger.debug(f"Citations received: {result.get('citations', 'NONE')}")
            
            # Build citations with actual content
            citations = []
            raw_citations = result.get("citations", [])
            if not raw_citations:
                logger.warning("No citations in LLM response")
            
            for cite in raw_citations:
                source_num = cite.get("source_number", 1) - 1  # Convert to 0-indexed
                if 0 <= source_num < len(original_contexts):
                    ctx = original_contexts[source_num]
                    section_hierarchy = ctx.get("metadata", {}).get("section_hierarchy", [])
                    section = " > ".join(section_hierarchy) if section_hierarchy else f"Source {source_num + 1}"
                    
                    citations.append(Citation(
                        source_id=ctx.get("id", ""),
                        section=section,
                        content_excerpt=cite.get("quote", ""),
                        relevance_note=cite.get("relevance", ""),
                    ))
            
            return RAGAnswer(
                question=question,
                answer=result.get("final_answer", ""),
                reasoning=result.get("reasoning_summary", ""),
                confidence=result.get("confidence", "medium"),
                relevant_sections=result.get("relevant_sections", []),
                citations=citations,
                sources=[
                    {
                        "id": c.get("id"), 
                        "section": c.get("metadata", {}).get("section_hierarchy", []),
                        "content_preview": c.get("content", "")[:200],
                    } 
                    for c in original_contexts
                ],
                timing=TimingInfo(generation_ms=generation_time),
            )
            
        except Exception as e:
            logger.error(f"Structured generation failed: {e}")
            # Fallback to simple generation
            return self._generate_simple(question, context, original_contexts)
    
    def _generate_simple(
        self,
        question: str,
        context: str,
        original_contexts: list[dict],
    ) -> RAGAnswer:
        """Generate simple text answer."""
        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": RAGAnswerPrompt.system_prompt},
                    {"role": "user", "content": format_rag_answer_prompt(context, question)}
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            
            answer = response.choices[0].message.content
            
            return RAGAnswer(
                question=question,
                answer=answer,
                sources=[{"id": c.get("id")} for c in original_contexts],
            )
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return RAGAnswer(
                question=question,
                answer=f"Error generating answer: {e}",
            )


# Convenience function
def generate_answer(
    question: str,
    contexts: list[dict],
    model: str = "gpt-4o-mini",
) -> RAGAnswer:
    """
    Quick answer generation.
    
    Args:
        question: User question
        contexts: Retrieved contexts
        model: LLM model to use
        
    Returns:
        RAGAnswer object
    """
    config = GeneratorConfig(model=model)
    generator = RAGGenerator(config)
    return generator.generate(question, contexts)

