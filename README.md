# Enterprise Modular Research-Paper RAG POC

Phase 1 / Milestone 1: a complete, working **Naive RAG** baseline built with a
clean, provider-independent architecture.

The goal of this repository is to be a stable `main` branch that multiple
contributors can build on. Phase 1 ships a deliberately simple pipeline
(chunk → embed → store → retrieve → generate) while establishing the extension
points needed for later phases (hybrid retrieval, reranking, evaluation, etc.).

## Architecture

The code follows a ports-and-adapters (hexagonal) layout. SDK-specific types
(Azure OpenAI, Azure AI Search) never leak into the domain or application layers.

```
src/enterprise_rag/
  domain/          # Pure models + Protocol "ports". No third-party SDKs.
    models.py      # Document, Chunk, EmbeddedChunk, RetrievedChunk, RagAnswer, ...
    ports.py       # EmbeddingProvider, ChatModel, VectorStore, Retriever, ...
    errors.py
  application/      # Use cases & default strategies. Depend only on domain ports.
    chunking.py     # FixedSizeChunker (ChunkingStrategy)
    prompts.py      # DefaultPromptBuilder (PromptBuilder)
    retrieval.py    # VectorRetriever (Retriever)
    ingestion.py    # IngestionService
    query.py        # RagQueryService
  infrastructure/   # Concrete adapters. SDK wrappers live *only* here.
    azure_openai/   # AzureOpenAIEmbeddingProvider, AzureOpenAIChatModel
    azure_search/   # AzureAISearchVectorStore
    memory/         # InMemoryVectorStore, HashingEmbeddingProvider (offline/tests)
    loaders/        # TextFileLoader, PdfLoader
  config/           # pydantic-settings configuration
  composition.py    # Composition root: wires Azure adapters into services
  cli.py            # `rag ingest <path>` / `rag query "<question>"`
```

### Why direct SDK integration (no LangChain core)

Phase 1 integrates the Azure SDKs directly behind narrow domain ports. This keeps
the core small, avoids coupling the architecture to an orchestration framework,
and makes it trivial to swap or add providers later. If a framework wrapper is
ever needed, it must be confined to an infrastructure adapter.

## Extension points (for Phase 2 / Phase 3)

These interfaces exist so future work is additive (new adapter/strategy) rather
than a rewrite:

| Port | Phase 1 implementation | Future examples |
| --- | --- | --- |
| `EmbeddingProvider` | `AzureOpenAIEmbeddingProvider` | local/other embedding models |
| `ChatModel` | `AzureOpenAIChatModel` | other LLM providers |
| `VectorStore` | `AzureAISearchVectorStore` | pgvector, FAISS, Qdrant |
| `Retriever` | `VectorRetriever` | hybrid search, reranking, HyDE |
| `ChunkingStrategy` | `FixedSizeChunker` | semantic / layout-aware chunking |
| `PromptBuilder` | `DefaultPromptBuilder` | citation-aware / templated prompts |
| `DocumentLoader` | `TextFileLoader`, `PdfLoader` | HTML, DOCX, connectors |

## Getting started

### Prerequisites

- Python >= 3.10
- (Optional) [uv](https://docs.astral.sh/uv/) for fast, locked dependency management

### Install (uv)

```powershell
uv sync --extra dev
```

### Install (pip)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

### Configure

Copy `.env.example` to `.env` and fill in your Azure values.

### Run

```powershell
# One-time: create the Azure AI Search index
rag init-index

# Ingest a folder of .txt research papers (or a single file)
rag ingest ./data/papers

# Ask a question
rag query "What datasets were used to evaluate the method?"
```

## Development

```powershell
ruff check .
mypy
pytest
```

Unit tests run fully offline — no network access or Azure credentials required —
by wiring the domain/application layers against in-memory fakes.

## Roadmap

- **Phase 1 (this milestone):** Naive RAG baseline, Azure providers, clean ports.
- **Phase 2:** advanced retrieval strategies (hybrid, reranking) behind `Retriever`.
- **Phase 3:** evaluation harness, additional providers, and connectors.
