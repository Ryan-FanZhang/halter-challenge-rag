"""
Search Demo Script
Demonstrates various search capabilities of the RAG system.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(project_root / ".env")

from src.retriever import RAGSearcher


def print_results(title: str, results: list, show_content: bool = True):
    """Pretty print search results."""
    print(f"\n{'='*80}")
    print(f"🔍 {title}")
    print(f"{'='*80}")
    print(f"Found {len(results)} results\n")
    
    for i, result in enumerate(results, 1):
        print(f"{'─'*60}")
        print(f"📄 Result {i}")
        print(f"{'─'*60}")
        print(f"  ID:    {result.id}")
        print(f"  Score: {result.score:.4f}")
        
        # Metadata
        meta = result.metadata
        if meta.get("section_hierarchy"):
            print(f"  Path:  {' > '.join(meta.get('section_hierarchy', []))}")
        if meta.get("content_type"):
            print(f"  Type:  {meta.get('content_type')}")
        if meta.get("workflow_type"):
            print(f"  Workflow: {meta.get('workflow_type')}")
        if meta.get("topics"):
            print(f"  Topics: {', '.join(meta.get('topics', [])[:5])}")
        
        # Content preview
        if show_content and result.content:
            preview = result.content[:300].replace('\n', ' ')
            print(f"\n  📝 Content Preview:")
            print(f"     {preview}...")
    
    print()


def main():
    print("=" * 80)
    print("🚀 RAG Search Demo")
    print("=" * 80)
    
    # Initialize searcher
    searcher = RAGSearcher(namespace="agents-doc")
    
    # Get stats
    stats = searcher.get_stats()
    print(f"\n📊 Index Stats:")
    print(f"   Total Vectors: {stats['total_vector_count']}")
    
    # Demo 1: Basic semantic search
    print_results(
        "Basic Search: 'What is an agent?'",
        searcher.search("What is an agent?", top_k=3)
    )
    
    # Demo 2: Search by content type
    print_results(
        "Search Examples: 'How to use routing?'",
        searcher.search_by_content_type(
            query="How to use routing?",
            content_type="example",
            top_k=3
        )
    )
    
    # Demo 3: Search by workflow type
    print_results(
        "Search Routing Workflow: 'classification'",
        searcher.search_by_workflow(
            query="classification",
            workflow_type="routing",
            top_k=3
        )
    )
    
    # Demo 4: Search with diagrams
    print_results(
        "Search with Diagrams: 'workflow architecture'",
        searcher.search_with_diagrams(
            query="workflow architecture",
            top_k=3
        )
    )
    
    # Demo 5: Advanced search with multiple filters
    print_results(
        "Advanced Search: 'best practices' (workflow content, with lists)",
        searcher.search_advanced(
            query="best practices",
            top_k=3,
            content_type="workflow",
            has_code=False,
        )
    )
    
    # Demo 6: Search by topic
    print_results(
        "Search by Topic: 'implementation' (topic: tool)",
        searcher.search_by_topic(
            query="implementation",
            topic="tool",
            top_k=3
        )
    )
    
    print("=" * 80)
    print("✅ Search Demo Completed!")
    print("=" * 80)
    
    # Interactive mode
    print("\n💡 Interactive Mode")
    print("Enter your queries (type 'quit' to exit):\n")
    
    while True:
        query = input("🔍 Query: ").strip()
        if query.lower() in ['quit', 'exit', 'q']:
            break
        if not query:
            continue
        
        results = searcher.search(query, top_k=3)
        print_results(f"Results for: '{query}'", results)


if __name__ == "__main__":
    main()

