"""
Interactive RAG Search
Ask questions and get answers from your documents.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(project_root / ".env")

from src.retriever import RAGSearcher, HybridSearcher


def print_results(results: list, show_full: bool = False):
    """Print search results."""
    if not results:
        print("\n❌ No results found.\n")
        return
    
    print(f"\n📚 Found {len(results)} results:\n")
    
    for i, r in enumerate(results, 1):
        print(f"{'─'*70}")
        print(f"📄 [{i}] Score: {r.score:.4f}")
        
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
        
        # Content
        content = r.content if show_full else r.content[:400]
        print(f"\n   {content.replace(chr(10), chr(10) + '   ')}{'...' if not show_full and len(r.content) > 400 else ''}")
    
    print(f"{'─'*70}\n")


def main():
    print("=" * 70)
    print("🔍 RAG Interactive Search")
    print("=" * 70)
    print("\nCommands:")
    print("  • Just type your question to search")
    print("  • 'h' or 'hybrid' - Toggle hybrid search mode")
    print("  • 'f' or 'full'   - Toggle full content display")
    print("  • 'k <num>'       - Set top_k (e.g., 'k 10')")
    print("  • 'a <num>'       - Set alpha for hybrid (e.g., 'a 0.7')")
    print("  • 'quit' or 'q'   - Exit")
    print("=" * 70)
    
    # Initialize searchers
    basic = RAGSearcher(namespace="agents-doc")
    hybrid = HybridSearcher(namespace="agents-doc")
    
    # Settings
    use_hybrid = False
    show_full = False
    top_k = 5
    alpha = 0.5
    
    print(f"\n⚙️  Settings: top_k={top_k}, hybrid={use_hybrid}, alpha={alpha}")
    
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
        
        if query.lower() in ['h', 'hybrid']:
            use_hybrid = not use_hybrid
            print(f"⚙️  Hybrid mode: {'ON' if use_hybrid else 'OFF'}")
            continue
        
        if query.lower() in ['f', 'full']:
            show_full = not show_full
            print(f"⚙️  Full content: {'ON' if show_full else 'OFF'}")
            continue
        
        if query.lower().startswith('k '):
            try:
                top_k = int(query.split()[1])
                print(f"⚙️  top_k set to: {top_k}")
            except:
                print("❌ Invalid format. Use: k <number>")
            continue
        
        if query.lower().startswith('a '):
            try:
                alpha = float(query.split()[1])
                alpha = max(0, min(1, alpha))
                print(f"⚙️  alpha set to: {alpha}")
            except:
                print("❌ Invalid format. Use: a <0.0-1.0>")
            continue
        
        # Search
        mode = "Hybrid" if use_hybrid else "Semantic"
        print(f"\n🔎 Searching ({mode}, top_k={top_k})...")
        
        if use_hybrid:
            results = hybrid.hybrid_search(query, top_k=top_k, alpha=alpha)
        else:
            results = basic.search(query, top_k=top_k)
        
        print_results(results, show_full)


if __name__ == "__main__":
    main()

