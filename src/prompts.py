"""
Centralized Prompts Module
Modular prompt system with reusable components.

Structure:
- Each prompt is a class with separated components
- Components: instruction, pydantic_schema, example, user_prompt
- build_system_prompt() combines components into final prompt
- Shared instructions can be reused across prompts
"""

import re
import inspect
from typing import List, Union, Literal
from pydantic import BaseModel, Field


# =============================================================================
# PROMPT BUILDER
# =============================================================================

def build_system_prompt(
    instruction: str = "",
    example: str = "",
    pydantic_schema: str = "",
) -> str:
    """
    Build system prompt from modular components.
    
    Args:
        instruction: Core system instruction
        example: Few-shot example(s)
        pydantic_schema: Pydantic schema for structured output
        
    Returns:
        Combined system prompt
    """
    delimiter = "\n\n---\n\n"
    
    result = instruction.strip()
    
    if pydantic_schema:
        schema_section = f"Your answer should be in JSON and strictly follow this schema, filling in the fields in the order they are given:\n```\n{pydantic_schema}\n```"
        result += delimiter + schema_section.strip()
    
    if example:
        result += delimiter + example.strip()
    
    return result


# =============================================================================
# SHARED INSTRUCTIONS (Reusable across prompts)
# =============================================================================

class SharedInstructions:
    """Shared instruction blocks for reuse across prompts."""
    
    step_by_step = """
Before giving a final answer, carefully think step by step. 
Pay special attention to the wording of the question.
"""
    
    context_only = """
Base your answer ONLY on the information provided in the context.
Do not use external knowledge or make assumptions beyond what is explicitly stated.
"""
    
    technical_accuracy = """
When discussing technical concepts:
- Use precise terminology as it appears in the context
- Distinguish between similar but different concepts
- Note any caveats or conditions mentioned
"""
    
    citation_guidance = """
When citing information:
- Reference specific sections and concepts from the context
- Quote key phrases when they directly answer the question
- Indicate if information comes from multiple parts of the context
"""


# =============================================================================
# RAG ANSWER PROMPT
# =============================================================================

class RAGAnswerPrompt:
    """
    Prompt for answering questions based on retrieved context.
    Optimized for technical documentation about AI Agents.
    """
    
    instruction = f"""
You are a RAG (Retrieval-Augmented Generation) answering system specialized in AI and LLM agent documentation.

Your task is to answer questions based ONLY on the provided context retrieved from technical documents.

{SharedInstructions.step_by_step}
{SharedInstructions.context_only}
{SharedInstructions.technical_accuracy}

Guidelines:
- If the context contains the answer, provide a clear and comprehensive response
- If the context is partially relevant, explain what you can answer and what information is missing
- If the context does not contain relevant information, clearly state "Based on the provided context, I cannot find information about..."
- When explaining workflows or patterns, structure your answer clearly with bullet points or steps

Citation Guidelines:
- ALWAYS provide citations by quoting directly from the source context
- Each citation should be an EXACT quote (50-150 characters) from the context
- Include 2-5 citations that directly support your answer
- For each citation, note which Source number it comes from (e.g., Source 1, Source 2)
- Explain briefly why each citation is relevant
"""
    
    class CitationItem(BaseModel):
        """Single citation from source."""
        source_number: int = Field(description="Source number (1, 2, 3...) from the context")
        quote: str = Field(description="Exact quote from the source, 50-150 characters")
        relevance: str = Field(description="Brief explanation of why this quote is relevant")
    
    class AnswerSchema(BaseModel):
        """Schema for RAG answer response."""
        step_by_step_analysis: str = Field(
            description="Detailed analysis of how the context relates to the question. "
                       "Identify key information, evaluate relevance, and build toward the answer. "
                       "At least 100 words."
        )
        reasoning_summary: str = Field(
            description="Concise summary of the reasoning process. Around 30-50 words."
        )
        relevant_sections: List[str] = Field(
            description="List of section names or topics from the context that were used to form the answer."
        )
        citations: List['RAGAnswerPrompt.CitationItem'] = Field(
            description="2-5 direct quotes from the context that support the answer. "
                       "Each citation must include the source number, exact quote, and relevance."
        )
        confidence: Literal["high", "medium", "low"] = Field(
            description="Confidence level based on how directly the context addresses the question. "
                       "high: Direct answer found. medium: Partial/indirect answer. low: Mostly inferred."
        )
        final_answer: str = Field(
            description="Clear, comprehensive answer to the question. "
                       "Use bullet points or numbered lists for complex explanations. "
                       "If information is not available, state that clearly."
        )
    
    pydantic_schema = re.sub(r"^ {4}", "", inspect.getsource(AnswerSchema), flags=re.MULTILINE)
    
    example = r"""
Example:

Question: "What is the difference between workflows and agents?"

Context:
[Source 1] What are agents?
"Agent" can be defined in several ways... At Anthropic, we categorize all these variations as agentic systems, but draw an important architectural distinction between workflows and agents:
- Workflows are systems where LLMs and tools are orchestrated through predefined code paths.
- Agents, on the other hand, are systems where LLMs dynamically direct their own processes and tool usage, maintaining control over how they accomplish tasks.

Answer:
```json
{
  "step_by_step_analysis": "1. The question asks about the difference between workflows and agents. 2. The context explicitly defines both terms and contrasts them. 3. Workflows are described as having 'predefined code paths' - meaning the flow is determined by code. 4. Agents are described as having dynamic control - the LLM decides the process. 5. The key distinction is about who controls the process: code (workflows) vs LLM (agents).",
  "reasoning_summary": "The context provides clear definitions distinguishing workflows (predefined orchestration) from agents (dynamic LLM-driven processes).",
  "relevant_sections": ["What are agents?", "Architectural distinction"],
  "citations": [
    {
      "source_number": 1,
      "quote": "Workflows are systems where LLMs and tools are orchestrated through predefined code paths",
      "relevance": "Defines workflows as code-controlled, predefined systems"
    },
    {
      "source_number": 1,
      "quote": "Agents are systems where LLMs dynamically direct their own processes and tool usage",
      "relevance": "Defines agents as LLM-controlled, dynamic systems"
    }
  ],
  "confidence": "high",
  "final_answer": "The key difference lies in **who controls the process**:\n\n**Workflows:** Systems with predefined code paths - the flow is determined by code structure.\n\n**Agents:** Systems where LLMs dynamically direct their own processes - the LLM maintains control over task execution."
}
```
"""
    
    user_prompt = """
Here is the context retrieved from the knowledge base:
\"\"\"
{context}
\"\"\"

---

Question: "{question}"

IMPORTANT: You MUST always provide citations by quoting exact text from the sources above.
For each citation:
1. Specify the source number (e.g., Source 1, Source 2)
2. Quote 50-150 characters of EXACT text from that source
3. Explain why this quote is relevant

Provide a comprehensive answer with at least 2-3 citations from the context.
"""
    
    system_prompt = build_system_prompt(instruction, example)
    system_prompt_with_schema = build_system_prompt(instruction, example, pydantic_schema)


# =============================================================================
# RERANKING PROMPT
# =============================================================================

class RerankingPrompt:
    """
    Prompt for LLM-based reranking of retrieved chunks.
    Supports both single and multiple block evaluation.
    """
    
    instruction_single = """
You are a RAG (Retrieval-Augmented Generation) retrieval ranker.

You will receive a query and a retrieved text block. Your task is to evaluate and score the block based on its relevance to the query.

Instructions:

1. Reasoning: 
   Analyze the block by identifying key information and how it relates to the query. Consider whether the block provides direct answers, partial insights, or background context. Explain your reasoning concisely.

2. Relevance Score (0 to 1, in increments of 0.1):
   0.0 = Completely Irrelevant: No connection to the query
   0.1 = Virtually Irrelevant: Very slight or vague connection
   0.2 = Very Slightly Relevant: Minimal or tangential connection
   0.3 = Slightly Relevant: Addresses a small aspect, lacks detail
   0.4 = Somewhat Relevant: Partial information, not comprehensive
   0.5 = Moderately Relevant: Addresses query with limited relevance
   0.6 = Fairly Relevant: Provides relevant info, lacking depth
   0.7 = Relevant: Clearly relates, offers substantive information
   0.8 = Very Relevant: Strongly relates, provides significant information
   0.9 = Highly Relevant: Almost completely answers with detail
   1.0 = Perfectly Relevant: Directly and comprehensively answers

3. Guidance:
   - Be objective: Evaluate based only on content relative to query
   - Be clear: Concise justifications
   - No assumptions: Don't infer beyond what's explicitly stated
"""
    
    instruction_multiple = """
You are a RAG (Retrieval-Augmented Generation) retrieval ranker.

You will receive a query and several retrieved text blocks. Your task is to evaluate and score EACH block based on its relevance to the query.

Instructions:

1. Reasoning (for each block): 
   Analyze the block by identifying key information and how it relates to the query. Explain your reasoning concisely.

2. Relevance Score (0 to 1, in increments of 0.1):
   0.0 = Completely Irrelevant: No connection to the query
   0.1 = Virtually Irrelevant: Very slight or vague connection
   0.2 = Very Slightly Relevant: Minimal or tangential connection
   0.3 = Slightly Relevant: Addresses a small aspect, lacks detail
   0.4 = Somewhat Relevant: Partial information, not comprehensive
   0.5 = Moderately Relevant: Addresses query with limited relevance
   0.6 = Fairly Relevant: Provides relevant info, lacking depth
   0.7 = Relevant: Clearly relates, offers substantive information
   0.8 = Very Relevant: Strongly relates, provides significant information
   0.9 = Highly Relevant: Almost completely answers with detail
   1.0 = Perfectly Relevant: Directly and comprehensively answers

3. Guidance:
   - Evaluate each block independently
   - Be objective and consistent across blocks
   - No assumptions beyond explicit content
"""
    
    class SingleBlockSchema(BaseModel):
        """Schema for single block ranking."""
        reasoning: str = Field(
            description="Analysis of the block, identifying key information and relation to query"
        )
        relevance_score: float = Field(
            description="Relevance score from 0 to 1"
        )
    
    class MultipleBlockSchema(BaseModel):
        """Schema for multiple block ranking."""
        scores: List['RerankingPrompt.BlockScore'] = Field(
            description="List of scores for each block"
        )
    
    class BlockScore(BaseModel):
        """Individual block score."""
        chunk_number: int = Field(description="Block number (1-indexed)")
        reasoning: str = Field(description="Analysis of relevance")
        relevance_score: float = Field(description="Score from 0 to 1")
    
    pydantic_schema_single = re.sub(r"^ {4}", "", inspect.getsource(SingleBlockSchema), flags=re.MULTILINE)
    pydantic_schema_multiple = re.sub(r"^ {4}", "", inspect.getsource(MultipleBlockSchema), flags=re.MULTILINE)
    
    user_prompt_single = """
Query: "{query}"

Text Block:
\"\"\"
{block}
\"\"\"

Evaluate this block's relevance to the query.
"""
    
    user_prompt_multiple = """
Query: "{query}"

Text Blocks to Evaluate:
{blocks}

Evaluate each block's relevance to the query.
"""
    
    system_prompt_single = build_system_prompt(instruction_single)
    system_prompt_single_with_schema = build_system_prompt(instruction_single, "", pydantic_schema_single)
    
    system_prompt_multiple = build_system_prompt(instruction_multiple)
    system_prompt_multiple_with_schema = build_system_prompt(instruction_multiple, "", pydantic_schema_multiple)


# =============================================================================
# QUERY EXPANSION PROMPT
# =============================================================================

class QueryExpansionPrompt:
    """
    Prompt for expanding queries to improve retrieval recall.
    """
    
    instruction = """
You are a search query expansion expert.

Your task is to generate alternative queries that capture the same intent but use different words or phrasings.

Guidelines:
1. Generate 3-5 alternative queries
2. Use synonyms and related technical terms
3. Try different question formats (what, how, why, when)
4. Consider different levels of specificity
5. Keep the original intent intact
6. For technical topics, include both formal and informal phrasings
"""
    
    class ExpansionSchema(BaseModel):
        """Schema for query expansion."""
        original_intent: str = Field(
            description="Brief summary of what the user is asking"
        )
        queries: List[str] = Field(
            description="List of 3-5 alternative search queries"
        )
    
    pydantic_schema = re.sub(r"^ {4}", "", inspect.getsource(ExpansionSchema), flags=re.MULTILINE)
    
    example = r"""
Example:

Original Query: "What is an agent?"

Output:
```json
{
  "original_intent": "User wants to understand the definition and concept of an agent in the context of LLMs",
  "queries": [
    "How are agents defined in LLM systems?",
    "What is the difference between agents and workflows?",
    "Definition of agentic systems",
    "What makes a system an agent vs a workflow?",
    "Agent architecture and characteristics"
  ]
}
```
"""
    
    user_prompt = """
Original Query: "{query}"

Generate alternative search queries that might help find relevant information.
"""
    
    system_prompt = build_system_prompt(instruction, example)
    system_prompt_with_schema = build_system_prompt(instruction, example, pydantic_schema)


# =============================================================================
# CONTEXT COMPRESSION PROMPT
# =============================================================================

class ContextCompressionPrompt:
    """
    Prompt for compressing/summarizing retrieved context before answering.
    Useful when context is too long.
    """
    
    instruction = f"""
You are a context compression expert for RAG systems.

Your task is to compress the provided context while preserving all information relevant to answering the question.

{SharedInstructions.technical_accuracy}

Guidelines:
1. Remove redundant or repeated information
2. Keep all facts, numbers, and specific details relevant to the question
3. Preserve technical terminology exactly as written
4. Maintain the logical structure of explanations
5. Remove tangential information not related to the question
"""
    
    class CompressionSchema(BaseModel):
        """Schema for context compression."""
        relevant_points: List[str] = Field(
            description="List of key points from context relevant to the question"
        )
        compressed_context: str = Field(
            description="Compressed version of the context, preserving relevant information"
        )
    
    pydantic_schema = re.sub(r"^ {4}", "", inspect.getsource(CompressionSchema), flags=re.MULTILINE)
    
    user_prompt = """
Question: "{question}"

Original Context:
\"\"\"
{context}
\"\"\"

Compress this context while preserving all information needed to answer the question.
"""
    
    system_prompt = build_system_prompt(instruction)
    system_prompt_with_schema = build_system_prompt(instruction, "", pydantic_schema)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def format_rag_answer_prompt(context: str, question: str) -> str:
    """Format RAG answer user prompt."""
    return RAGAnswerPrompt.user_prompt.format(
        context=context,
        question=question,
    )


def format_rerank_single_prompt(query: str, block: str) -> str:
    """Format single block reranking prompt."""
    return RerankingPrompt.user_prompt_single.format(
        query=query,
        block=block,
    )


def format_rerank_multiple_prompt(query: str, blocks: List[str]) -> str:
    """Format multiple blocks reranking prompt."""
    blocks_text = ""
    for i, block in enumerate(blocks, 1):
        blocks_text += f"\n--- Block {i} ---\n{block}\n"
    
    return RerankingPrompt.user_prompt_multiple.format(
        query=query,
        blocks=blocks_text,
    )


def format_query_expansion_prompt(query: str) -> str:
    """Format query expansion prompt."""
    return QueryExpansionPrompt.user_prompt.format(query=query)


def format_context_compression_prompt(question: str, context: str) -> str:
    """Format context compression prompt."""
    return ContextCompressionPrompt.user_prompt.format(
        question=question,
        context=context,
    )


# =============================================================================
# LEGACY COMPATIBILITY (for existing reranker.py)
# =============================================================================

# Keep these for backward compatibility
LLM_RERANK_SYSTEM_PROMPT = RerankingPrompt.system_prompt_multiple
LLM_RERANK_USER_TEMPLATE = """## Question
{query}

## Text Chunks to Evaluate
{chunks_text}

## Task
Evaluate each chunk's relevance to answering the question. Return JSON with scores for all {num_chunks} chunks."""


def format_rerank_user_prompt(query: str, chunks_text: str, num_chunks: int) -> str:
    """Legacy format function for reranker compatibility."""
    return LLM_RERANK_USER_TEMPLATE.format(
        query=query,
        chunks_text=chunks_text,
        num_chunks=num_chunks,
    )
