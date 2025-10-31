# RAG Document Assistant

A production-ready Retrieval-Augmented Generation (RAG) system for intelligent document Q&A, built with FastAPI and LangChain.

## 🌟 Features

- **📁 Multi-Format Support**: Upload and process PDF, Word, Markdown, CSV, JSON, and text files
- **💬 Dual Chat Modes**: 
  - Simple Q&A for standalone questions
  - Conversational mode with memory for context-aware discussions
- **🔍 Smart Search**: Semantic search using FAISS vector database with HuggingFace embeddings
- **🤖 Multiple LLMs**: Support for OpenAI, Google Gemini, and Anthropic Claude
- **📚 Document Filtering**: Scope questions to specific documents
- **💾 Persistent Storage**: Vector embeddings saved to disk for fast restart
- **🎨 Web Interface**: Clean, modern UI for document management and chat

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Virtual environment (recommended)

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd <project-directory>
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   
   Create a `.env` file in the root directory:
   ```env
   # At least one API key is required
   OPENAI_API_KEY=your_openai_api_key_here
   GOOGLE_API_KEY=your_google_api_key_here
   ANTHROPIC_API_KEY=your_anthropic_api_key_here

   # Model configurations (optional)
   DEFAULT_OPENAI_MODEL=gpt-3.5-turbo
   DEFAULT_GEMINI_MODEL=gemini-pro
   DEFAULT_CLAUDE_MODEL=claude-3-opus-20240229

   # App settings (optional)
   DEBUG=False
   LOG_LEVEL=INFO
   MAX_TOKENS=1000
   ```

5. **Run the application**
   ```bash
   uvicorn main:app --reload
   ```

6. **Open your browser**
   
   Navigate to `http://localhost:8000`

## 📖 Usage

### Document Upload

1. Click on the upload section in the sidebar
2. Choose a file (PDF, DOCX, TXT, MD, CSV, or JSON)
3. Optionally add a custom title
4. Click "Upload & Process"
5. Wait for processing to complete

### Asking Questions

**Simple Q&A Mode:**
- Each question is treated independently
- Best for factual queries about documents

**Conversational Mode:**
- Maintains context between questions
- Ideal for follow-up questions and discussions
- System remembers previous exchanges in the session

### Document Filtering

Use the dropdown menu to limit searches to specific documents, improving accuracy and speed.

## 🏗️ Architecture

```
project/
├── app/
│   ├── api/              # API route handlers
│   │   ├── chat_routes.py
│   │   ├── conversational_routes.py
│   │   └── documents_routes.py
│   ├── services/         # Business logic
│   │   ├── enhanced_rag_service.py
│   │   ├── conversational_memory.py
│   │   └── advanced_document_loaders.py
│   ├── schemas/          # Pydantic models
│   ├── database.py       # SQLAlchemy setup
│   └── config.py         # Configuration management
├── static/               # Frontend assets
├── templates/            # HTML templates
├── tests/                # Test utilities
└── main.py              # FastAPI application
```

## 🔧 Configuration

### Supported File Types

- **Text**: `.txt`, `.md`, `.markdown`
- **Documents**: `.pdf`, `.docx`, `.doc`
- **Data**: `.csv`, `.json`

### Embedding Model

Uses `all-MiniLM-L6-v2` from HuggingFace for generating embeddings (runs on CPU).

### Vector Storage

FAISS index is automatically saved to `./faiss_index/` for persistence across restarts.

## 📊 API Endpoints

### Documents
- `POST /api/v1/documents/upload-file` - Upload a document file
- `GET /api/v1/documents/list` - List all documents
- `DELETE /api/v1/documents/{id}` - Delete a document

### Chat
- `POST /api/v1/chat/ask` - Ask a question (simple mode)
- `POST /api/v1/conversation/ask` - Ask with conversation memory
- `GET /api/v1/conversation/history/{session_id}` - Get conversation history
- `GET /api/v1/chat/status` - Get system status

## 🛠️ Development

### Adding New Document Types

Extend the `MultiFormatDocumentLoader` class in `app/services/advanced_document_loaders.py`:

```python
def _load_custom(self, file_path: str, metadata: Dict[str, Any]) -> List[Document]:
    # Your custom loader logic
    pass
```

### Customizing the Prompt

Modify the prompt template in `enhanced_rag_service.py`:

```python
prompt_template = """Your custom prompt here
Context: {context}
Question: {question}
Answer:"""
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- Built with [FastAPI](https://fastapi.tiangolo.com/)
- Powered by [LangChain](https://langchain.com/)
- Vector search by [FAISS](https://faiss.ai/)
- UI inspired by modern chat interfaces

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

---

**Note**: This is a learning project focused on RAG implementation and LLM integration. For production use, consider adding authentication, rate limiting, and comprehensive error handling.