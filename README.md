# Enterprise RAG System with AI Agent

An enterprise-grade Retrieval-Augmented Generation (RAG) system built with LangChain, featuring an intelligent AI agent that can autonomously route queries to the appropriate tools.

## 🌟 Features

### RAG Pipeline
- **Hybrid Search**: Combines dense (semantic) and sparse (keyword) retrieval
- **LLM Reranking**: Uses GPT-4o-mini to rerank retrieved chunks for better relevance
- **Rich Metadata**: Extracts and stores comprehensive metadata for filtering
- **Citations**: Provides exact quotes from source documents

### AI Agent (ReAct Pattern)
- **Intelligent Routing**: Automatically determines the best tool for each query
- **Multi-Tool Support**: RAG, API Query, and Ticket Escalation
- **Conversation Memory**: Maintains context across multiple turns
- **Transparency**: Shows which tools are being used and why

### Tools
| Tool | Purpose | Trigger |
|------|---------|---------|
| **RAG Knowledge Base** | Technical docs, concepts, best practices | "What is...", "How does...", "Explain..." |
| **Query API** | Agent status, token usage, billing, metrics | "How many tokens...", "Show my agents..." |
| **Ticket Escalation** | Create support tickets | "Talk to human", low confidence, errors |

## 📁 Project Structure

```
halter-challenge-rag/
├── src/
│   ├── agents/                    # AI Agent system
│   │   ├── orchestrator.py        # Main agent with tool calling
│   │   └── tools/
│   │       ├── rag_tool.py        # RAG knowledge base tool
│   │       ├── query_api_tool.py  # API query tool
│   │       └── ticket_tool.py     # Ticket escalation tool
│   ├── retriever/                 # Retrieval components
│   │   ├── search.py              # Basic semantic search
│   │   ├── hybrid_search.py       # Hybrid search (dense + sparse)
│   │   ├── reranker.py            # LLM-based reranking
│   │   └── pipeline.py            # Full retrieval pipeline
│   ├── generator/                 # Generation components
│   │   └── rag_generator.py       # Answer generation with citations
│   ├── vectorstore/               # Vector database
│   │   ├── embeddings.py          # OpenAI embeddings
│   │   └── pinecone_store.py      # Pinecone vector store
│   ├── document_processing/       # Document processing
│   │   ├── chunker.py             # Markdown chunking
│   │   └── metadata_extractor.py  # Rich metadata extraction
│   ├── data/                      # Mock data
│   │   └── mock_api_data.py       # Simulated API responses
│   └── prompts.py                 # Centralized prompt management
├── scripts/
│   ├── agent.py                   # Interactive AI agent
│   ├── rag.py                     # Standalone RAG Q&A
│   ├── chunking.py                # Document chunking
│   └── upload_to_pinecone.py      # Vector upload
├── raw_data/                      # Source documents
├── processed_data/                # Processed chunks (JSON)
├── logs/                          # Support tickets
├── requirements.txt
└── .env                           # API keys (not committed)
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repo-url>
cd halter-challenge-rag

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Create a `.env` file in the project root:

```bash
# OpenAI API Key
OPENAI_API_KEY=sk-your-openai-key

# Pinecone Configuration
PINECONE_API_KEY=your-pinecone-key
PINECONE_INDEX_NAME=your-index-name

# Embedding Model
EMBEDDING_MODEL=text-embedding-3-large
EMBEDDING_DIMENSION=1024
```

### 3. Process Documents

```bash
# Chunk documents and extract metadata
python scripts/chunking.py

# Upload to Pinecone
python scripts/upload_to_pinecone.py
```

### 4. Run the AI Agent

```bash
python scripts/agent.py
```

## 💡 Usage Examples

### AI Agent (Recommended)

```bash
python scripts/agent.py
```

```
❓ You: What is the difference between workflows and agents?
🔄 Processing...
🤖 Running agent to determine best tool...
🔧 Tool Selected: rag_knowledge_base

📚 Response [RAG]
────────────────────────────────────────────────────────────
The key difference lies in who controls the process:

**Workflows:** Systems with predefined code paths...
**Agents:** Systems where LLMs dynamically direct their own processes...
────────────────────────────────────────────────────────────

❓ You: How many tokens have I used?
🔄 Processing...
🔧 Tool Selected: query_api

🔌 Response [API]
────────────────────────────────────────────────────────────
📈 **Token Usage Summary**
**Total Tokens:** 4,200,000
**Quota:** 45.0% used
────────────────────────────────────────────────────────────
```

### Standalone RAG

```bash
python scripts/rag.py
```

### Programmatic Usage

```python
from src.agents import AgentOrchestrator

# Initialize agent
agent = AgentOrchestrator(
    model="gpt-4o-mini",
    memory_window=10,
    verbose=True
)

# Ask questions
response = agent.ask("What is an agent?")
print(response.answer)
print(f"Source: {response.source}")
print(f"Tools used: {response.tool_calls}")
```

## 🔧 Architecture

### RAG Pipeline

```
Query → Hybrid Search → LLM Reranking → Context Augmentation → Generation
         ↓                  ↓                   ↓                   ↓
    Dense + Sparse    GPT-4o-mini         Format + Compress    Structured Output
    (top 15)          (top 5)             with metadata        with citations
```

### Agent Flow

```
User Question
     │
     ▼
┌─────────────────────────────────────┐
│       Agent Orchestrator            │
│      (Tool Calling Pattern)         │
└────────────────┬────────────────────┘
                 │
    ┌────────────┼────────────┐
    ▼            ▼            ▼
┌───────┐   ┌────────┐   ┌─────────┐
│  RAG  │   │  API   │   │ Ticket  │
│ Tool  │   │ Query  │   │  Tool   │
└───────┘   └────────┘   └─────────┘
```

## 📊 Metadata Schema

Each chunk includes rich metadata for filtering:

| Field | Type | Description |
|-------|------|-------------|
| `doc_title` | string | Document title |
| `section_hierarchy` | list | Section path (e.g., ["Building blocks", "Agents"]) |
| `content_type` | string | definition, example, workflow, appendix |
| `workflow_type` | string | routing, parallelization, agents, etc. |
| `has_diagram` | bool | Contains diagrams |
| `has_code` | bool | Contains code |
| `topics` | list | Extracted topics |

## 🎛️ Configuration Options

### RAG Pipeline

```python
from src.retriever.pipeline import PipelineConfig

config = PipelineConfig(
    namespace="agents-doc",
    initial_k=15,           # Initial retrieval count
    final_k=5,              # After reranking
    use_hybrid=True,        # Enable hybrid search
    use_reranking=True,     # Enable LLM reranking
    vector_weight=0.3,      # Weight for vector score
    rerank_weight=0.7,      # Weight for LLM score
)
```

### Generator

```python
from src.generator import GeneratorConfig

config = GeneratorConfig(
    model="gpt-4o-mini",
    temperature=0.1,
    max_tokens=2000,
    use_structured_output=True,
)
```

## 📝 Prompts

All prompts are centralized in `src/prompts.py` with modular structure:

- `RAGAnswerPrompt` - Knowledge base Q&A
- `RerankingPrompt` - LLM reranking
- `QueryExpansionPrompt` - Query expansion
- `ContextCompressionPrompt` - Context compression

## 🔒 Ticket System

When the agent cannot answer or user requests human help:

```json
{
  "ticket_id": "TKT-20241212-001",
  "priority": "medium",
  "reason": "low_confidence",
  "user_question": "...",
  "status": "open",
  "created_at": "2024-12-12T10:30:00Z"
}
```

Tickets are saved to `logs/ticket_*.json`.

## 🛠️ Development

### Adding New Tools

1. Create tool in `src/agents/tools/`
2. Inherit from `BaseTool`
3. Define `name`, `description`, `args_schema`
4. Implement `_run()` method
5. Add to `orchestrator.py` tools list

### Adding New Documents

1. Place markdown files in `raw_data/`
2. Run `python scripts/chunking.py`
3. Run `python scripts/upload_to_pinecone.py`

## 📦 Dependencies

- **LangChain** - Agent framework
- **OpenAI** - LLM and embeddings
- **Pinecone** - Vector database
- **Pydantic** - Data validation
- **Loguru** - Logging

## 📄 License

MIT License

## 🙏 Acknowledgments

- Built with [LangChain](https://langchain.com/)
- Vector storage by [Pinecone](https://pinecone.io/)
- Inspired by Anthropic's "Building Effective Agents" guide
