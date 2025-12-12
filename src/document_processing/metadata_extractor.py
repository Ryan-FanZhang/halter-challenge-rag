"""
Metadata Extractor Module
Extracts rich metadata from document chunks for enhanced filtering and retrieval.
"""

import re
from typing import Any
from dataclasses import dataclass, field, asdict


@dataclass
class ChunkMetadata:
    """Rich metadata structure for document chunks."""
    
    # Structural metadata
    doc_title: str = ""
    doc_source: str = ""
    section_hierarchy: list[str] = field(default_factory=list)
    section_level: int = 0
    chunk_index: int = 0
    total_chunks: int = 0
    line_start: int = 0
    line_end: int = 0
    
    # Content type metadata
    content_type: str = "general"  # definition, example, workflow, best_practice, appendix
    has_diagram: bool = False
    has_code: bool = False
    has_list: bool = False
    has_table: bool = False
    
    # Semantic metadata
    topics: list[str] = field(default_factory=list)
    workflow_type: str | None = None  # prompt_chaining, routing, parallelization, etc.
    difficulty: str = "intermediate"  # beginner, intermediate, advanced
    
    # Retrieval optimization metadata
    keywords: list[str] = field(default_factory=list)
    summary: str = ""
    token_count: int = 0
    char_count: int = 0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert metadata to dictionary for vector store storage."""
        return asdict(self)


class MetadataExtractor:
    """Extracts rich metadata from markdown content chunks."""
    
    # Workflow type patterns
    WORKFLOW_PATTERNS = {
        "prompt_chaining": r"prompt\s*chain",
        "routing": r"\brouting\b",
        "parallelization": r"parallel",
        "orchestrator_workers": r"orchestrator.*worker|worker.*orchestrator",
        "evaluator_optimizer": r"evaluator.*optimizer|optimizer.*evaluator",
        "agents": r"\bagents?\b(?!.*workflow)",
    }
    
    # Content type patterns
    CONTENT_TYPE_PATTERNS = {
        "definition": r"^#+\s*what\s+(is|are)|defined?\s+as|means?\s+that",
        "example": r"example|for\s+instance|such\s+as|e\.g\.|use\s+case",
        "workflow": r"workflow|pattern|approach|method",
        "best_practice": r"best\s+practice|recommend|suggestion|should|principle",
        "appendix": r"appendix|附录",
    }
    
    # Topic extraction patterns
    TOPIC_KEYWORDS = [
        "LLM", "agent", "workflow", "tool", "prompt", "retrieval", "memory",
        "orchestrator", "worker", "routing", "classification", "parallelization",
        "evaluation", "optimization", "feedback", "iteration", "API", "framework",
        "customer support", "coding", "SWE-bench", "Claude", "Anthropic",
    ]
    
    def __init__(self, doc_title: str = "", doc_source: str = ""):
        self.doc_title = doc_title
        self.doc_source = doc_source
    
    def extract(
        self,
        content: str,
        section_hierarchy: list[str],
        chunk_index: int = 0,
        total_chunks: int = 1,
        line_start: int = 0,
        line_end: int = 0,
    ) -> ChunkMetadata:
        """
        Extract rich metadata from a chunk of content.
        
        Args:
            content: The text content of the chunk
            section_hierarchy: List of parent section titles
            chunk_index: Index of this chunk in the document
            total_chunks: Total number of chunks in the document
            line_start: Starting line number in original document
            line_end: Ending line number in original document
            
        Returns:
            ChunkMetadata object with all extracted metadata
        """
        metadata = ChunkMetadata(
            doc_title=self.doc_title,
            doc_source=self.doc_source,
            section_hierarchy=section_hierarchy,
            section_level=len(section_hierarchy),
            chunk_index=chunk_index,
            total_chunks=total_chunks,
            line_start=line_start,
            line_end=line_end,
        )
        
        # Extract content characteristics
        metadata.has_diagram = self._has_diagram(content)
        metadata.has_code = self._has_code(content)
        metadata.has_list = self._has_list(content)
        metadata.has_table = self._has_table(content)
        
        # Extract content type
        metadata.content_type = self._detect_content_type(content, section_hierarchy)
        
        # Extract workflow type if applicable
        metadata.workflow_type = self._detect_workflow_type(content, section_hierarchy)
        
        # Extract topics
        metadata.topics = self._extract_topics(content, section_hierarchy)
        
        # Extract keywords
        metadata.keywords = self._extract_keywords(content)
        
        # Estimate difficulty
        metadata.difficulty = self._estimate_difficulty(content, section_hierarchy)
        
        # Count tokens and chars
        metadata.char_count = len(content)
        metadata.token_count = self._estimate_tokens(content)
        
        # Generate summary (first sentence or first 150 chars)
        metadata.summary = self._generate_summary(content)
        
        return metadata
    
    def _has_diagram(self, content: str) -> bool:
        """Check if content contains diagram/image references."""
        patterns = [
            r"!\[.*?\]\(.*?\)",  # Markdown image
            r"diagram\s*reference",
            r"<img\s",
        ]
        return any(re.search(p, content, re.IGNORECASE) for p in patterns)
    
    def _has_code(self, content: str) -> bool:
        """Check if content contains code blocks or inline code."""
        patterns = [
            r"```[\s\S]*?```",  # Code block
            r"`[^`]+`",  # Inline code
            r"def\s+\w+\s*\(",  # Python function
            r"function\s+\w+\s*\(",  # JS function
        ]
        return any(re.search(p, content) for p in patterns)
    
    def _has_list(self, content: str) -> bool:
        """Check if content contains bullet or numbered lists."""
        patterns = [
            r"^\s*[-*+]\s+",  # Bullet list
            r"^\s*\d+\.\s+",  # Numbered list
            r"^\s*[○◦•]\s*",  # Special bullets
        ]
        return any(re.search(p, content, re.MULTILINE) for p in patterns)
    
    def _has_table(self, content: str) -> bool:
        """Check if content contains markdown tables."""
        return bool(re.search(r"\|.*\|.*\|", content))
    
    def _detect_content_type(self, content: str, hierarchy: list[str]) -> str:
        """Detect the primary content type of the chunk."""
        combined = f"{' '.join(hierarchy)} {content}".lower()
        
        # Check hierarchy first
        for section in hierarchy:
            section_lower = section.lower()
            if "appendix" in section_lower:
                return "appendix"
            if "example" in section_lower:
                return "example"
            if "workflow" in section_lower:
                return "workflow"
        
        # Check content patterns
        for content_type, pattern in self.CONTENT_TYPE_PATTERNS.items():
            if re.search(pattern, combined, re.IGNORECASE):
                return content_type
        
        return "general"
    
    def _detect_workflow_type(self, content: str, hierarchy: list[str]) -> str | None:
        """Detect specific workflow type if applicable."""
        combined = f"{' '.join(hierarchy)} {content}".lower()
        
        for workflow_type, pattern in self.WORKFLOW_PATTERNS.items():
            if re.search(pattern, combined, re.IGNORECASE):
                return workflow_type
        
        return None
    
    def _extract_topics(self, content: str, hierarchy: list[str]) -> list[str]:
        """Extract relevant topics from content."""
        combined = f"{' '.join(hierarchy)} {content}"
        topics = []
        
        for keyword in self.TOPIC_KEYWORDS:
            if re.search(rf"\b{re.escape(keyword)}\b", combined, re.IGNORECASE):
                topics.append(keyword.lower())
        
        return list(set(topics))[:10]  # Limit to top 10 topics
    
    def _extract_keywords(self, content: str) -> list[str]:
        """Extract important keywords from content."""
        # Remove markdown syntax
        clean_content = re.sub(r"[#*`\[\]()]", " ", content)
        clean_content = re.sub(r"!\[.*?\]\(.*?\)", "", clean_content)
        clean_content = re.sub(r"https?://\S+", "", clean_content)
        
        # Extract capitalized words and technical terms
        words = re.findall(r"\b[A-Z][a-z]+(?:[A-Z][a-z]+)*\b", clean_content)
        words += re.findall(r"\b[A-Z]{2,}\b", clean_content)  # Acronyms
        
        # Count and filter
        word_counts: dict[str, int] = {}
        for word in words:
            word_lower = word.lower()
            if len(word) > 2:
                word_counts[word_lower] = word_counts.get(word_lower, 0) + 1
        
        # Return top keywords
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        return [w[0] for w in sorted_words[:15]]
    
    def _estimate_difficulty(self, content: str, hierarchy: list[str]) -> str:
        """Estimate content difficulty level."""
        combined = f"{' '.join(hierarchy)} {content}".lower()
        
        advanced_indicators = [
            "complex", "sophisticated", "advanced", "architecture",
            "implementation detail", "optimization", "edge case",
        ]
        beginner_indicators = [
            "introduction", "what is", "what are", "basic", "simple",
            "getting started", "overview",
        ]
        
        advanced_score = sum(1 for ind in advanced_indicators if ind in combined)
        beginner_score = sum(1 for ind in beginner_indicators if ind in combined)
        
        if advanced_score > beginner_score:
            return "advanced"
        elif beginner_score > advanced_score:
            return "beginner"
        return "intermediate"
    
    def _estimate_tokens(self, content: str) -> int:
        """Estimate token count (rough approximation: ~4 chars per token)."""
        return len(content) // 4
    
    def _generate_summary(self, content: str, max_length: int = 200) -> str:
        """Generate a brief summary from content."""
        # Remove markdown formatting
        clean = re.sub(r"^#+\s*", "", content, flags=re.MULTILINE)
        clean = re.sub(r"!\[.*?\]\(.*?\)", "[图片]", clean)
        clean = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", clean)
        clean = re.sub(r"[*_`]", "", clean)
        
        # Get first meaningful sentence
        sentences = re.split(r"[.!?]\s+", clean.strip())
        summary = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20:
                summary = sentence[:max_length]
                if len(sentence) > max_length:
                    summary = summary.rsplit(" ", 1)[0] + "..."
                break
        
        return summary if summary else clean[:max_length]

