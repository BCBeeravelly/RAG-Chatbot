# Context-Aware QA Assistant with Dynamic Web Search

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An intelligent question-answering system that combines document retrieval with web search capabilities, using LLM-as-judge decision making.

## Key Features

- **Hybrid Retrieval System**

  - Parent-child document architecture for context preservation
  - Voyage-3-large embeddings with Qdrant vector store
  - Chunking with configurable text splitting

- **LLM-Powered Intelligence**

  - GPT-4 based answering with conversation memory
  - LLM-as-judge for web search triggering
  - Confidence-based search decision making

- **Web Integration**
  - Tavily API for precision web searching
  - Context-aware result integration
  - Cost-efficient search triggering

## Workflow Diagram

```mermaid
graph TD
    A[User Input] --> B{Document Context?}
    B -->|Sufficient| C[Generate Answer]
    B -->|Insufficient| D[Tavily Web Search]
    C --> E[LLM Confidence Check]
    E -->|High Confidence| F[Direct Answer]
    E -->|Low Confidence| D
    D --> G[Combine Contexts]
    G --> H[Enhanced Answer]
    H --> I[Update Memory]
    F --> I
```
