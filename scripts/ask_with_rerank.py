"""
Interactive RAG Search with Reranking
Ask questions and get reranked answers.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(project_root / ".env")

from src.retriever.pipeline import RAGPipeline, PipelineConfig
from src.retriever import RerankResult


def print_results(results: list[RerankResult], show_reasoning: bool = True):
    """Print reranked results."""
    if not results:
        print("\n❌ No results found.\n")
        return
    
    print(f"\n📚 Found {len(results)} results (reranked):\n")
    
    for i, r in enumerate(results, 1):
        print(f"{'─'*70}")
        print(f"📄 [{i}] Final: {r.final_score:.4f} (Vector: {r.original_score:.4f} → LLM: {r.rerank_score:.2f})")
        
        meta = r.metadata
        if meta.get("section_hierarchy"):
            print(f"   📂 {' > '.join(meta.get('section_hierarchy', []))}")
        
        tags = []
        if meta.get("content_type"):
            tags.append(f"type:{meta['content_type']}")
        if meta.get("workflow_type"):
            tags.append(f"workflow:{meta['workflow_type']}")
        if tags:
            print(f"   🏷️  {' | '.join(tags)}")
        
        # Reasoning from reranker
        if show_reasoning and r.reasoning:
            print(f"\n   💭 Reasoning: {r.reasoning[:200]}{'...' if len(r.reasoning) > 200 else ''}")
        
        # Content preview
        if r.content:
            preview = r.content[:350].replace('\n', ' ')
            print(f"\n   📝 {preview}...")
    
    print(f"{'─'*70}\n")


def main():
    print("=" * 70)
    print("🚀 RAG Search with LLM Reranking")
    print("=" * 70)
    print("\nPipeline: Vector Search → LLM Reranking (GPT-4o-mini)")
    print("\nCommands:")
    print("  • Type your question to search")
    print("  • 'r' - Toggle reasoning display")
    print("  • 'h' - Toggle hybrid mode")
    print("  • 'k <num>' - Set final top_k")
    print("  • 'i <num>' - Set initial retrieval count")
    print("  • 'w <v> <r>' - Set weights (e.g., 'w 0.3 0.7')")
    print("  • 'q' - Quit")
    print("=" * 70)
    
    # Initialize pipeline
    config = PipelineConfig(
        namespace="agents-doc",
        initial_k=15,
        final_k=5,
        use_hybrid=True,
        use_reranking=True,
        vector_weight=0.3,
        rerank_weight=0.7,
    )
    pipeline = RAGPipeline(config)
    
    # Settings
    show_reasoning = True
    
    print(f"\n⚙️  Settings: initial_k={config.initial_k}, final_k={config.final_k}")
    print(f"   Weights: vector={config.vector_weight}, rerank={config.rerank_weight}")
    print(f"   Hybrid: {config.use_hybrid}, Reranking: {config.use_reranking}")
    
    while True:
        try:
            query = input("\n🔍 Ask: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n👋 Bye!")
            break
        
        if not query:
            continue
        
        # Commands
        if query.lower() in ['quit', 'q', 'exit']:
            print("\n👋 Bye!")
            break
        
        if query.lower() == 'r':
            show_reasoning = not show_reasoning
            print(f"⚙️  Show reasoning: {'ON' if show_reasoning else 'OFF'}")
            continue
        
        if query.lower() == 'h':
            config.use_hybrid = not config.use_hybrid
            print(f"⚙️  Hybrid mode: {'ON' if config.use_hybrid else 'OFF'}")
            continue
        
        if query.lower().startswith('k '):
            try:
                config.final_k = int(query.split()[1])
                print(f"⚙️  final_k set to: {config.final_k}")
            except:
                print("❌ Invalid format. Use: k <number>")
            continue
        
        if query.lower().startswith('i '):
            try:
                config.initial_k = int(query.split()[1])
                print(f"⚙️  initial_k set to: {config.initial_k}")
            except:
                print("❌ Invalid format. Use: i <number>")
            continue
        
        if query.lower().startswith('w '):
            try:
                parts = query.split()
                config.vector_weight = float(parts[1])
                config.rerank_weight = float(parts[2])
                # Reinitialize reranker with new weights
                from src.retriever.reranker import LLMReranker
                pipeline.reranker = LLMReranker(
                    vector_weight=config.vector_weight,
                    llm_weight=config.rerank_weight,
                )
                print(f"⚙️  Weights: vector={config.vector_weight}, rerank={config.rerank_weight}")
            except:
                print("❌ Invalid format. Use: w <vector_weight> <rerank_weight>")
            continue
        
        # Search with reranking
        print(f"\n🔎 Searching (initial={config.initial_k}, final={config.final_k})...")
        print(f"   Stage 1: Vector/Hybrid retrieval...")
        print(f"   Stage 2: LLM Reranking (GPT-4o-mini)...")
        
        try:
            results = pipeline.search(
                query=query,
                top_k=config.final_k,
                use_hybrid=config.use_hybrid,
                use_reranking=True,
            )
            print_results(results, show_reasoning)
        except Exception as e:
            print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()

