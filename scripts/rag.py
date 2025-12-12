"""
Complete RAG Pipeline Script
End-to-end: Retrieval → Reranking → Augmentation → Generation
"""

import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(project_root / ".env")

from loguru import logger
import sys

# Enable debug logging
logger.remove()
logger.add(sys.stderr, level="DEBUG")

from src.retriever.pipeline import RAGPipeline, PipelineConfig
from src.generator import RAGGenerator, GeneratorConfig
from src.generator.rag_generator import TimingInfo


class RAGSystem:
    """
    Complete RAG System.
    
    Pipeline:
    1. Retrieval (Hybrid Search)
    2. Reranking (LLM)
    3. Augmentation (Context Formatting)
    4. Generation (LLM Answer)
    """
    
    def __init__(
        self,
        namespace: str = "agents-doc",
        retrieval_k: int = 15,
        rerank_k: int = 5,
        model: str = "gpt-4o-mini",
    ):
        """
        Initialize RAG System.
        
        Args:
            namespace: Pinecone namespace
            retrieval_k: Initial retrieval count
            rerank_k: Final count after reranking
            model: LLM model for generation
        """
        # Retrieval + Reranking pipeline
        self.retrieval_config = PipelineConfig(
            namespace=namespace,
            initial_k=retrieval_k,
            final_k=rerank_k,
            use_hybrid=True,
            use_reranking=True,
            vector_weight=0.3,
            rerank_weight=0.7,
        )
        self.retrieval_pipeline = RAGPipeline(self.retrieval_config)
        
        # Generator
        self.generator_config = GeneratorConfig(
            model=model,
            use_structured_output=True,
            include_reasoning=True,
        )
        self.generator = RAGGenerator(self.generator_config)
    
    def ask(self, question: str, verbose: bool = True) -> dict:
        """
        Ask a question and get an answer.
        
        Args:
            question: User question
            verbose: Print progress
            
        Returns:
            Dict with answer and metadata
        """
        total_start = time.time()
        timing = TimingInfo()
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"❓ Question: {question}")
            print(f"{'='*70}")
        
        # Step 1: Retrieve and Rerank
        if verbose:
            print("\n📚 Step 1: Retrieval + Reranking...")
        
        retrieval_start = time.time()
        reranked_results = self.retrieval_pipeline.search(question)
        timing.retrieval_ms = (time.time() - retrieval_start) * 1000
        timing.reranking_ms = timing.retrieval_ms * 0.6  # Approximate split
        timing.retrieval_ms = timing.retrieval_ms * 0.4
        
        if verbose:
            print(f"   Retrieved {self.retrieval_config.initial_k} → Reranked to {len(reranked_results)} results")
            print(f"   ⏱️  Time: {timing.retrieval_ms + timing.reranking_ms:.0f}ms")
            for i, r in enumerate(reranked_results[:3], 1):
                section = r.metadata.get('section_hierarchy', ['?'])
                section_str = ' > '.join(section) if section else '?'
                print(f"   [{i}] Score: {r.final_score:.3f} | {section_str[:50]}")
        
        # Step 2: Format contexts for generator
        augmentation_start = time.time()
        contexts = [
            {
                "id": r.id,
                "content": r.content,
                "metadata": r.metadata,
                "score": r.final_score,
            }
            for r in reranked_results
        ]
        timing.augmentation_ms = (time.time() - augmentation_start) * 1000
        
        # Step 3: Generate answer
        if verbose:
            print("\n🤖 Step 2: Generating Answer...")
        
        generation_start = time.time()
        answer = self.generator.generate(question, contexts)
        timing.generation_ms = (time.time() - generation_start) * 1000
        
        # Update answer timing
        answer.timing = timing
        answer.timing.total_ms = (time.time() - total_start) * 1000
        
        if verbose:
            print(f"   ⏱️  Generation Time: {timing.generation_ms:.0f}ms")
            
            print(f"\n{'─'*70}")
            print(f"💡 Answer (Confidence: {answer.confidence})")
            print(f"{'─'*70}")
            print(f"\n{answer.answer}")
            
            if answer.reasoning:
                print(f"\n📝 Reasoning: {answer.reasoning}")
            
            # Display citations with actual content
            if answer.citations:
                print(f"\n📖 Citations:")
                for i, cite in enumerate(answer.citations, 1):
                    print(f"\n   [{i}] {cite.section}")
                    print(f"       \"{cite.content_excerpt}\"")
                    if cite.relevance_note:
                        print(f"       → {cite.relevance_note}")
            
            # Timing summary
            print(f"\n{'─'*70}")
            print(f"⏱️  Timing Summary")
            print(f"{'─'*70}")
            print(f"   • Retrieval:   {timing.retrieval_ms:>7.0f} ms")
            print(f"   • Reranking:   {timing.reranking_ms:>7.0f} ms")
            print(f"   • Generation:  {timing.generation_ms:>7.0f} ms")
            print(f"   • Total:       {answer.timing.total_ms:>7.0f} ms")
        
        return {
            "question": question,
            "answer": answer.answer,
            "confidence": answer.confidence,
            "reasoning": answer.reasoning,
            "relevant_sections": answer.relevant_sections,
            "citations": [
                {
                    "source_id": c.source_id,
                    "section": c.section,
                    "quote": c.content_excerpt,
                    "relevance": c.relevance_note,
                }
                for c in answer.citations
            ],
            "timing": {
                "retrieval_ms": timing.retrieval_ms,
                "reranking_ms": timing.reranking_ms,
                "generation_ms": timing.generation_ms,
                "total_ms": answer.timing.total_ms,
            },
            "sources": [
                {
                    "id": r.id,
                    "score": r.final_score,
                    "section": r.metadata.get("section_hierarchy", []),
                }
                for r in reranked_results
            ],
        }


def main():
    print("=" * 70)
    print("🚀 Complete RAG System")
    print("   Retrieval → Reranking → Augmentation → Generation")
    print("=" * 70)
    
    # Initialize RAG system
    rag = RAGSystem(
        namespace="agents-doc",
        retrieval_k=15,
        rerank_k=5,
        model="gpt-4o-mini",
    )
    
    print("\n⚙️  Configuration:")
    print(f"   • Retrieval: Hybrid Search (initial_k=15)")
    print(f"   • Reranking: LLM (final_k=5)")
    print(f"   • Generation: GPT-4o-mini (structured output)")
    
    # Demo questions
    demo_questions = [
        "What is the difference between workflows and agents?",
        "When should I use the routing workflow pattern?",
        "What are the best practices for building agents?",
    ]
    
    print("\n" + "=" * 70)
    print("📋 Demo Questions")
    print("=" * 70)
    
    for q in demo_questions:
        print(f"\n  • {q}")
    
    # Run demo
    run_demo = input("\n▶️  Run demo? (y/n): ").strip().lower()
    
    if run_demo == 'y':
        for question in demo_questions:
            result = rag.ask(question)
            print("\n" + "=" * 70)
            input("Press Enter for next question...")
    
    # Interactive mode
    print("\n" + "=" * 70)
    print("💬 Interactive Mode")
    print("   Type your questions (or 'quit' to exit)")
    print("=" * 70)
    
    while True:
        try:
            question = input("\n❓ Ask: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n👋 Bye!")
            break
        
        if not question:
            continue
        
        if question.lower() in ['quit', 'q', 'exit']:
            print("\n👋 Bye!")
            break
        
        result = rag.ask(question)


if __name__ == "__main__":
    main()

