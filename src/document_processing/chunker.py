"""
Enhanced Markdown Chunker Module
Two-stage chunking strategy with rich metadata extraction.
"""

import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Generator

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_core.documents import Document

from .metadata_extractor import MetadataExtractor, ChunkMetadata


@dataclass
class ChunkConfig:
    """Configuration for chunking behavior."""
    
    chunk_size: int = 800
    chunk_overlap: int = 100
    
    # Headers to split on (markdown level -> metadata key)
    headers_to_split_on: list[tuple[str, str]] = field(default_factory=lambda: [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
        ("####", "h4"),
    ])
    
    # Separators for recursive splitting (in order of priority)
    separators: list[str] = field(default_factory=lambda: [
        "\n## ",      # H2 headers
        "\n### ",     # H3 headers
        "\n#### ",    # H4 headers
        "\n\n",       # Paragraphs
        "\n",         # Lines
        ". ",         # Sentences
        " ",          # Words
    ])
    
    # Minimum chunk size (skip if smaller)
    min_chunk_size: int = 50
    
    # Whether to strip whitespace
    strip_whitespace: bool = True


class EnhancedMarkdownChunker:
    """
    Two-stage markdown chunker with rich metadata extraction.
    
    Stage 1: Split by markdown headers (semantic boundaries)
    Stage 2: Further split large chunks using recursive character splitter
    """
    
    def __init__(self, config: ChunkConfig | None = None):
        self.config = config or ChunkConfig()
        
        # Stage 1: Markdown header splitter
        self.header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.config.headers_to_split_on,
            strip_headers=False,  # Keep headers for context
        )
        
        # Stage 2: Recursive character splitter for oversized chunks
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=self.config.separators,
            length_function=len,
        )
    
    def chunk_file(self, file_path: str | Path) -> list[Document]:
        """
        Chunk a markdown file with rich metadata.
        
        Args:
            file_path: Path to the markdown file
            
        Returns:
            List of LangChain Document objects with rich metadata
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        content = file_path.read_text(encoding="utf-8")
        
        # Extract document title from first H1
        doc_title = self._extract_doc_title(content)
        
        return self.chunk_text(
            content=content,
            doc_title=doc_title,
            doc_source=str(file_path),
        )
    
    def chunk_text(
        self,
        content: str,
        doc_title: str = "",
        doc_source: str = "",
    ) -> list[Document]:
        """
        Chunk markdown text with rich metadata.
        
        Args:
            content: Markdown content to chunk
            doc_title: Title of the document
            doc_source: Source path or identifier
            
        Returns:
            List of LangChain Document objects with rich metadata
        """
        # Initialize metadata extractor
        extractor = MetadataExtractor(doc_title=doc_title, doc_source=doc_source)
        
        # Build line index for tracking positions
        line_index = self._build_line_index(content)
        
        # Stage 1: Split by headers
        header_chunks = self.header_splitter.split_text(content)
        
        # Stage 2: Further split if needed and extract metadata
        final_documents: list[Document] = []
        chunk_counter = 0
        
        for header_chunk in header_chunks:
            chunk_content = header_chunk.page_content
            header_metadata = header_chunk.metadata
            
            # Build section hierarchy from header metadata
            section_hierarchy = self._build_section_hierarchy(header_metadata)
            
            # Find line range in original content
            line_start, line_end = self._find_line_range(
                chunk_content, content, line_index
            )
            
            # Check if chunk needs further splitting
            if len(chunk_content) > self.config.chunk_size:
                # Split into smaller chunks
                sub_chunks = self.text_splitter.split_text(chunk_content)
                
                for i, sub_chunk in enumerate(sub_chunks):
                    if len(sub_chunk.strip()) < self.config.min_chunk_size:
                        continue
                    
                    # Extract metadata for sub-chunk
                    metadata = extractor.extract(
                        content=sub_chunk,
                        section_hierarchy=section_hierarchy,
                        chunk_index=chunk_counter,
                        line_start=line_start,
                        line_end=line_end,
                    )
                    
                    # Add sub-chunk indicator
                    metadata_dict = metadata.to_dict()
                    metadata_dict["is_sub_chunk"] = True
                    metadata_dict["sub_chunk_index"] = i
                    metadata_dict["sub_chunk_total"] = len(sub_chunks)
                    
                    final_documents.append(Document(
                        page_content=sub_chunk.strip() if self.config.strip_whitespace else sub_chunk,
                        metadata=metadata_dict,
                    ))
                    chunk_counter += 1
            else:
                if len(chunk_content.strip()) < self.config.min_chunk_size:
                    continue
                
                # Extract metadata for whole chunk
                metadata = extractor.extract(
                    content=chunk_content,
                    section_hierarchy=section_hierarchy,
                    chunk_index=chunk_counter,
                    line_start=line_start,
                    line_end=line_end,
                )
                
                metadata_dict = metadata.to_dict()
                metadata_dict["is_sub_chunk"] = False
                metadata_dict["sub_chunk_index"] = 0
                metadata_dict["sub_chunk_total"] = 1
                
                final_documents.append(Document(
                    page_content=chunk_content.strip() if self.config.strip_whitespace else chunk_content,
                    metadata=metadata_dict,
                ))
                chunk_counter += 1
        
        # Update total chunks count
        for doc in final_documents:
            doc.metadata["total_chunks"] = len(final_documents)
        
        return final_documents
    
    def chunk_file_lazy(self, file_path: str | Path) -> Generator[Document, None, None]:
        """
        Lazily chunk a markdown file (memory efficient for large files).
        
        Args:
            file_path: Path to the markdown file
            
        Yields:
            LangChain Document objects with rich metadata
        """
        documents = self.chunk_file(file_path)
        yield from documents
    
    def _extract_doc_title(self, content: str) -> str:
        """Extract document title from first H1 header."""
        match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
        return match.group(1).strip() if match else ""
    
    def _build_line_index(self, content: str) -> list[int]:
        """Build index of line start positions."""
        index = [0]
        for i, char in enumerate(content):
            if char == "\n":
                index.append(i + 1)
        return index
    
    def _find_line_range(
        self,
        chunk: str,
        full_content: str,
        line_index: list[int],
    ) -> tuple[int, int]:
        """Find the line range of a chunk in the original content."""
        # Find chunk position in full content
        chunk_start = full_content.find(chunk[:100] if len(chunk) > 100 else chunk)
        if chunk_start == -1:
            return (0, 0)
        
        chunk_end = chunk_start + len(chunk)
        
        # Find line numbers
        line_start = 0
        line_end = len(line_index)
        
        for i, pos in enumerate(line_index):
            if pos <= chunk_start:
                line_start = i + 1
            if pos <= chunk_end:
                line_end = i + 1
        
        return (line_start, line_end)
    
    def _build_section_hierarchy(self, header_metadata: dict) -> list[str]:
        """Build section hierarchy from header metadata."""
        hierarchy = []
        
        for level in ["h1", "h2", "h3", "h4"]:
            if level in header_metadata:
                hierarchy.append(header_metadata[level])
        
        return hierarchy


def chunk_markdown_file(
    file_path: str | Path,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
) -> list[Document]:
    """
    Convenience function to chunk a markdown file.
    
    Args:
        file_path: Path to the markdown file
        chunk_size: Maximum chunk size in characters
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of LangChain Document objects with rich metadata
    """
    config = ChunkConfig(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunker = EnhancedMarkdownChunker(config)
    return chunker.chunk_file(file_path)

