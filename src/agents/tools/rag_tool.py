"""
RAG Tool
Knowledge base retrieval tool using RAG pipeline.
"""

from typing import Any, Type
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool

from src.retriever.pipeline import RAGPipeline, PipelineConfig
from src.generator import RAGGenerator, GeneratorConfig


class RAGToolInput(BaseModel):
    """Input schema for RAG tool."""
    question: str = Field(description="The question to search in the knowledge base")


class RAGTool(BaseTool):
    """
    RAG Tool for knowledge base queries.
    
    Use this tool when user asks:
    - Technical questions about agents, workflows, LLMs
    - Conceptual questions (what is X, how does Y work)
    - Best practices and recommendations
    - Documentation-related queries
    """
    
    name: str = "rag_knowledge_base"
    description: str = """Search the knowledge base for technical documentation and concepts.
Use this tool for questions about:
- Agent definitions and architectures
- Workflow patterns (routing, parallelization, orchestrator-workers, etc.)
- Best practices for building AI agents
- LLM concepts and implementations
- Technical documentation and guides

Input should be a clear question about the topic you want to learn about."""
    
    args_schema: Type[BaseModel] = RAGToolInput
    
    # Internal components
    retrieval_pipeline: RAGPipeline = None
    generator: RAGGenerator = None
    confidence_threshold: float = 0.6
    
    def __init__(self, confidence_threshold: float = 0.6, **kwargs):
        super().__init__(**kwargs)
        
        # Initialize retrieval pipeline
        retrieval_config = PipelineConfig(
            namespace="agents-doc",
            initial_k=15,
            final_k=5,
            use_hybrid=True,
            use_reranking=True,
            vector_weight=0.3,
            rerank_weight=0.7,
        )
        self.retrieval_pipeline = RAGPipeline(retrieval_config)
        
        # Initialize generator
        generator_config = GeneratorConfig(
            model="gpt-4o-mini",
            use_structured_output=True,
        )
        self.generator = RAGGenerator(generator_config)
        
        self.confidence_threshold = confidence_threshold
    
    def _run(self, question: str) -> dict[str, Any]:
        """Execute RAG query."""
        try:
            # Step 1: Retrieve and rerank
            reranked_results = self.retrieval_pipeline.search(question)
            
            if not reranked_results:
                return {
                    "success": False,
                    "source": "rag",
                    "answer": "No relevant information found in the knowledge base.",
                    "confidence": "low",
                    "should_escalate": True,
                    "escalate_reason": "no_results",
                }
            
            # Step 2: Format contexts
            contexts = [
                {
                    "id": r.id,
                    "content": r.content,
                    "metadata": r.metadata,
                    "score": r.final_score,
                }
                for r in reranked_results
            ]
            
            # Step 3: Generate answer
            answer = self.generator.generate(question, contexts)
            
            # Step 4: Check confidence and determine if escalation needed
            confidence_map = {"high": 0.9, "medium": 0.7, "low": 0.4}
            confidence_score = confidence_map.get(answer.confidence, 0.5)
            should_escalate = confidence_score < self.confidence_threshold
            
            return {
                "success": True,
                "source": "rag",
                "answer": answer.answer,
                "confidence": answer.confidence,
                "confidence_score": confidence_score,
                "reasoning": answer.reasoning,
                "citations": [
                    {
                        "section": c.section,
                        "quote": c.content_excerpt,
                        "relevance": c.relevance_note,
                    }
                    for c in answer.citations
                ],
                "relevant_sections": answer.relevant_sections,
                "should_escalate": should_escalate,
                "escalate_reason": "low_confidence" if should_escalate else None,
            }
            
        except Exception as e:
            return {
                "success": False,
                "source": "rag",
                "answer": f"Error querying knowledge base: {str(e)}",
                "confidence": "low",
                "should_escalate": True,
                "escalate_reason": "error",
                "error": str(e),
            }
    
    async def _arun(self, question: str) -> dict[str, Any]:
        """Async version - just calls sync for now."""
        return self._run(question)


# Create singleton instance
rag_tool = RAGTool()

