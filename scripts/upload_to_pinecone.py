"""
Upload chunks to Pinecone vector database.
"""

import sys
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

from src.vectorstore import PineconeStore


def main():
    # Paths
    chunks_file = project_root / "processed_data" / "chunks.json"
    
    if not chunks_file.exists():
        logger.error(f"❌ Chunks file not found: {chunks_file}")
        logger.info("Please run 'python scripts/chunking.py' first")
        return
    
    print("=" * 80)
    print("🚀 Upload to Pinecone")
    print("=" * 80)
    
    # Load chunks
    logger.info(f"Loading chunks from: {chunks_file}")
    with open(chunks_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    chunks = data["chunks"]
    logger.info(f"Loaded {len(chunks)} chunks")
    
    # Initialize Pinecone store
    logger.info("Initializing Pinecone store...")
    store = PineconeStore()
    
    # Create index if needed
    store.create_index_if_not_exists()
    
    # Upsert chunks
    logger.info("Uploading chunks to Pinecone...")
    stats = store.upsert_chunks(
        chunks=chunks,
        namespace="agents-doc",  # You can customize this
        batch_size=50,
    )
    
    print("\n" + "=" * 80)
    print("📊 Upload Statistics")
    print("=" * 80)
    print(f"  • Index Name:     {stats['index_name']}")
    print(f"  • Namespace:      {stats['namespace']}")
    print(f"  • Total Chunks:   {stats['total_chunks']}")
    print(f"  • Total Upserted: {stats['total_upserted']}")
    
    # Get index stats
    index_stats = store.get_index_stats()
    print(f"\n📈 Index Stats:")
    print(f"  • Total Vectors:  {index_stats['total_vector_count']}")
    print(f"  • Dimension:      {index_stats['dimension']}")
    
    print("\n" + "=" * 80)
    print("✅ Upload completed successfully!")
    print("=" * 80)
    
    # Test query
    print("\n🔍 Testing query...")
    results = store.query(
        query_text="What is an agent?",
        top_k=3,
        namespace="agents-doc",
    )
    
    print(f"\nTop 3 results for 'What is an agent?':")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Score: {result['score']:.4f}")
        print(f"   ID: {result['id']}")
        if result.get('metadata'):
            print(f"   Section: {' > '.join(result['metadata'].get('section_hierarchy', []))}")
            print(f"   Preview: {result['metadata'].get('content_preview', '')[:100]}...")


if __name__ == "__main__":
    main()

