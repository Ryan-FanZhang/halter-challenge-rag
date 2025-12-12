"""
Document chunking script.
Processes markdown files and saves chunks with rich metadata to JSON.
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.document_processing import EnhancedMarkdownChunker
from src.document_processing.chunker import ChunkConfig


def main():
    # Configure chunker
    config = ChunkConfig(
        chunk_size=800,
        chunk_overlap=100,
    )
    
    chunker = EnhancedMarkdownChunker(config)
    
    # Process the sample file
    input_file = project_root / "raw_data" / "build_effective_ai_agents.md"
    output_file = project_root / "processed_data" / "chunks.json"
    
    if not input_file.exists():
        print(f"❌ File not found: {input_file}")
        return
    
    print("=" * 80)
    print("🔧 Document Chunking Pipeline")
    print("=" * 80)
    print(f"\n📄 Input:  {input_file}")
    print(f"📁 Output: {output_file}")
    print(f"📏 Chunk Size: {config.chunk_size} chars")
    print(f"🔗 Overlap: {config.chunk_overlap} chars")
    print()
    
    # Chunk the file
    documents = chunker.chunk_file(input_file)
    
    print(f"📊 Total Chunks Generated: {len(documents)}")
    
    # Metadata fields to keep
    METADATA_FIELDS = [
        "doc_title",
        "doc_source",
        "section_hierarchy",
        "section_level",
        "content_type",
        "workflow_type",
        "has_diagram",
        "has_code",
        "has_list",
        "has_table",
        "topics",
    ]
    
    # Convert to JSON-serializable format
    chunks_data = {
        "metadata": {
            "source_file": str(input_file),
            "processed_at": datetime.now().isoformat(),
            "chunk_config": {
                "chunk_size": config.chunk_size,
                "chunk_overlap": config.chunk_overlap,
            },
            "total_chunks": len(documents),
        },
        "chunks": []
    }
    
    for i, doc in enumerate(documents):
        # Filter metadata to keep only specified fields
        filtered_metadata = {
            k: v for k, v in doc.metadata.items() 
            if k in METADATA_FIELDS
        }
        
        chunk_entry = {
            "id": f"chunk_{i:04d}",
            "content": doc.page_content,
            "metadata": filtered_metadata,
        }
        chunks_data["chunks"].append(chunk_entry)
    
    # Write to JSON file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(chunks_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Chunks saved to: {output_file}")
    
    # Print summary statistics
    print("\n" + "=" * 80)
    print("📈 SUMMARY STATISTICS")
    print("=" * 80)
    
    # Content type distribution
    content_types = {}
    workflow_types = {}
    topics_all = []
    
    for doc in documents:
        ct = doc.metadata.get('content_type', 'unknown')
        content_types[ct] = content_types.get(ct, 0) + 1
        
        wt = doc.metadata.get('workflow_type')
        if wt:
            workflow_types[wt] = workflow_types.get(wt, 0) + 1
        
        topics_all.extend(doc.metadata.get('topics', []))
    
    print("\n📊 Content Type Distribution:")
    for ct, count in sorted(content_types.items(), key=lambda x: -x[1]):
        print(f"   • {ct}: {count} chunks")
    
    print("\n🔄 Workflow Type Distribution:")
    for wt, count in sorted(workflow_types.items(), key=lambda x: -x[1]):
        print(f"   • {wt}: {count} chunks")
    
    print("\n🏷️  Top Topics:")
    topic_counts = {}
    for t in topics_all:
        topic_counts[t] = topic_counts.get(t, 0) + 1
    for topic, count in sorted(topic_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"   • {topic}: {count} occurrences")
    
    # Show sample chunk
    print("\n" + "=" * 80)
    print("📦 SAMPLE CHUNK (First Chunk)")
    print("=" * 80)
    
    if documents:
        sample = chunks_data["chunks"][0]
        print(f"\nID: {sample['id']}")
        print(f"\nMetadata:")
        for key, value in sample['metadata'].items():
            if isinstance(value, list):
                print(f"  • {key}: {value[:5]}{'...' if len(value) > 5 else ''}")
            else:
                print(f"  • {key}: {value}")
        print(f"\nContent Preview:")
        print(f"  {sample['content'][:300]}...")
    
    print("\n" + "=" * 80)
    print("✅ Chunking completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
