"""
Hybrid Search Demo Script
Demonstrates hybrid search combining semantic and keyword matching.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(project_root / ".env")

from src.retriever import RAGSearcher, HybridSearcher


def print_results(title: str, results: list):
    """Pretty print search results."""
    print(f"\n{'='*80}")
    print(f"🔍 {title}")
    print(f"{'='*80}")
    print(f"Found {len(results)} results\n")
    
    for i, result in enumerate(results, 1):
        print(f"{'─'*60}")
        print(f"📄 Result {i} | Score: {result.score:.4f}")
        print(f"{'─'*60}")
        
        meta = result.metadata
        if meta.get("section_hierarchy"):
            print(f"  📂 {' > '.join(meta.get('section_hierarchy', []))}")
        if meta.get("content_type"):
            print(f"  📝 Type: {meta.get('content_type')}")
        if meta.get("workflow_type"):
            print(f"  🔄 Workflow: {meta.get('workflow_type')}")
        
        # Content preview
        if result.content:
            preview = result.content[:250].replace('\n', ' ')
            print(f"\n  {preview}...")
    
    print()


def compare_search_methods(query: str):
    """Compare different search methods for the same query."""
    print(f"\n{'#'*80}")
    print(f"🔬 COMPARING SEARCH METHODS")
    print(f"   Query: \"{query}\"")
    print(f"{'#'*80}")
    
    # Initialize searchers
    basic_searcher = RAGSearcher(namespace="agents-doc")
    hybrid_searcher = HybridSearcher(namespace="agents-doc")
    
    # 1. Basic Semantic Search
    print_results(
        "Method 1: Basic Semantic Search",
        basic_searcher.search(query, top_k=3)
    )
    
    # 2. Hybrid Search (balanced)
    print_results(
        "Method 2: Hybrid Search (alpha=0.5, balanced)",
        hybrid_searcher.hybrid_search(query, top_k=3, alpha=0.5)
    )
    
    # 3. Hybrid Search (favor semantic)
    print_results(
        "Method 3: Hybrid Search (alpha=0.7, favor semantic)",
        hybrid_searcher.hybrid_search(query, top_k=3, alpha=0.7)
    )
    
    # 4. Hybrid Search (favor keyword)
    print_results(
        "Method 4: Hybrid Search (alpha=0.3, favor keyword)",
        hybrid_searcher.hybrid_search(query, top_k=3, alpha=0.3)
    )
    
    # 5. Search with Reranking
    print_results(
        "Method 5: Search with Reranking",
        hybrid_searcher.search_with_reranking(query, top_k=3, initial_k=15)
    )


def main():
    print("=" * 80)
    print("🚀 Hybrid Search Demo")
    print("=" * 80)
    
    # Initialize hybrid searcher
    searcher = HybridSearcher(namespace="agents-doc")
    
    # Demo 1: Basic hybrid search
    print_results(
        "Hybrid Search: 'What is an agent and how does it work?'",
        searcher.hybrid_search(
            "What is an agent and how does it work?",
            top_k=5,
            alpha=0.5  # Balanced semantic + keyword
        )
    )
    
    # Demo 2: Keyword-heavy search (good for specific terms)
    print_results(
        "Keyword-Heavy (alpha=0.3): 'orchestrator workers workflow'",
        searcher.hybrid_search(
            "orchestrator workers workflow",
            top_k=3,
            alpha=0.3  # Favor keyword matching
        )
    )
    
    # Demo 3: Semantic-heavy search (good for conceptual queries)
    print_results(
        "Semantic-Heavy (alpha=0.8): 'best way to handle complex AI tasks'",
        searcher.hybrid_search(
            "best way to handle complex AI tasks",
            top_k=3,
            alpha=0.8  # Favor semantic understanding
        )
    )
    
    # Demo 4: With metadata filter
    print_results(
        "Hybrid + Filter (workflow_type='routing'): 'classification'",
        searcher.hybrid_search(
            "classification",
            top_k=3,
            filter={"workflow_type": "routing"}
        )
    )
    
    # Demo 5: Search with reranking
    print_results(
        "Search with Reranking: 'tools and APIs for agents'",
        searcher.search_with_reranking(
            "tools and APIs for agents",
            top_k=5,
            initial_k=20
        )
    )
    
    # Compare methods
    compare_search_methods("prompt chaining workflow example")
    
    print("=" * 80)
    print("✅ Hybrid Search Demo Completed!")
    print("=" * 80)
    
    # Tips
    print("\n💡 Tips for choosing alpha value:")
    print("   • alpha = 0.5: Balanced (default, good for most cases)")
    print("   • alpha > 0.5: Favor semantic (conceptual/vague queries)")
    print("   • alpha < 0.5: Favor keyword (specific terms/names)")
    print("   • alpha = 1.0: Pure semantic search")
    print("   • alpha = 0.0: Pure keyword-enhanced search")


if __name__ == "__main__":
    main()

