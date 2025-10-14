"""
Advanced document loaders for different file types using LangChain.
"""

import os
import tempfile
from typing import List, Dict, Any, Optional
from pathlib import Path

from langchain.schema import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredMarkdownLoader,
    CSVLoader,
    JSONLoader,
    TextLoader
)
from langchain.document_loaders.base import BaseLoader


class MultiFormatDocumentLoader:
    """Enhanced document loader that supports multiple file formats."""

    def __init__(self):
        self.supported_formats = {
            '.txt': self._load_text,
            '.pdf': self._load_pdf,
            '.docx': self._load_word,
            '.doc': self._load_word,
            '.md': self._load_markdown,
            '.csv': self._load_csv,
            '.json': self._load_json,
        }

    def get_supported_formats(self) -> List[str]:
        """Return list of supported file extensions."""
        return list(self.supported_formats.keys())

    def load_from_content(self, content: str, filename: str, metadata: Dict[str, Any]) -> List[Document]:
        """Load document from string content based on filename extension."""
        try:
            file_ext = Path(filename).suffix.lower()

            if file_ext not in self.supported_formats:
                # Default to text loading
                return [Document(page_content=content, metadata=metadata)]

            # For text content, we'll create a temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix=file_ext, delete=False) as temp_file:
                temp_file.write(content)
                temp_path = temp_file.name

            try:
                # Load using appropriate loader
                documents = self.supported_formats[file_ext](temp_path, metadata)
                return documents
            finally:
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)

        except Exception as e:
            print(f"❌ Error loading document {filename}: {str(e)}")
            # Fallback to simple text document
            return [Document(page_content=content, metadata=metadata)]

    def load_from_file(self, file_path: str, metadata: Dict[str, Any]) -> List[Document]:
        """Load document from file path."""
        try:
            file_ext = Path(file_path).suffix.lower()

            if file_ext not in self.supported_formats:
                raise ValueError(f"Unsupported file format: {file_ext}")

            return self.supported_formats[file_ext](file_path, metadata)

        except Exception as e:
            print(f"❌ Error loading file {file_path}: {str(e)}")
            raise

    def _load_text(self, file_path: str, metadata: Dict[str, Any]) -> List[Document]:
        """Load plain text file."""
        loader = TextLoader(file_path, encoding='utf-8')
        documents = loader.load()

        # Add custom metadata
        for doc in documents:
            doc.metadata.update(metadata)

        return documents

    def _load_pdf(self, file_path: str, metadata: Dict[str, Any]) -> List[Document]:
        """Load PDF file."""
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        # Add page numbers and custom metadata
        for i, doc in enumerate(documents):
            doc.metadata.update(metadata)
            doc.metadata['page'] = i + 1
            doc.metadata['source_type'] = 'pdf'

        return documents

    def _load_word(self, file_path: str, metadata: Dict[str, Any]) -> List[Document]:
        """Load Word document (.docx or .doc)."""
        loader = UnstructuredWordDocumentLoader(file_path)
        documents = loader.load()

        # Add custom metadata
        for doc in documents:
            doc.metadata.update(metadata)
            doc.metadata['source_type'] = 'word'

        return documents

    def _load_markdown(self, file_path: str, metadata: Dict[str, Any]) -> List[Document]:
        """Load Markdown file."""
        loader = UnstructuredMarkdownLoader(file_path)
        documents = loader.load()

        # Add custom metadata
        for doc in documents:
            doc.metadata.update(metadata)
            doc.metadata['source_type'] = 'markdown'

        return documents

    def _load_csv(self, file_path: str, metadata: Dict[str, Any]) -> List[Document]:
        """Load CSV file."""
        loader = CSVLoader(file_path)
        documents = loader.load()

        # Add custom metadata
        for i, doc in enumerate(documents):
            doc.metadata.update(metadata)
            doc.metadata['row'] = i + 1
            doc.metadata['source_type'] = 'csv'

        return documents

    def _load_json(self, file_path: str, metadata: Dict[str, Any]) -> List[Document]:
        """Load JSON file."""
        # For JSON, we need to specify the content key
        loader = JSONLoader(file_path, jq_schema='.', text_content=False)
        documents = loader.load()

        # Add custom metadata
        for doc in documents:
            doc.metadata.update(metadata)
            doc.metadata['source_type'] = 'json'

        return documents


class WebDocumentLoader:
    """Loader for web content and URLs."""

    def __init__(self):
        try:
            from langchain_community.document_loaders import WebBaseLoader
            self.WebBaseLoader = WebBaseLoader
            self.available = True
        except ImportError:
            print("⚠️ WebBaseLoader not available. Install with: pip install beautifulsoup4")
            self.available = False

    def load_from_url(self, url: str, metadata: Dict[str, Any]) -> List[Document]:
        """Load document from URL."""
        if not self.available:
            raise ImportError("WebBaseLoader not available")

        try:
            loader = self.WebBaseLoader(url)
            documents = loader.load()

            # Add custom metadata
            for doc in documents:
                doc.metadata.update(metadata)
                doc.metadata['source_type'] = 'web'
                doc.metadata['url'] = url

            return documents

        except Exception as e:
            print(f"❌ Error loading URL {url}: {str(e)}")
            raise


class DocumentProcessor:
    """Enhanced document processor with metadata extraction."""

    def __init__(self):
        self.multi_loader = MultiFormatDocumentLoader()
        self.web_loader = WebDocumentLoader()

    def extract_metadata(self, filename: str, content: str) -> Dict[str, Any]:
        """Extract metadata from document content and filename."""
        metadata = {
            'filename': filename,
            'file_extension': Path(filename).suffix.lower(),
            'content_length': len(content),
            'word_count': len(content.split()),
        }

        # Add format-specific metadata
        file_ext = Path(filename).suffix.lower()

        if file_ext == '.pdf':
            # Estimate pages (rough calculation)
            metadata['estimated_pages'] = max(1, len(content) // 3000)

        elif file_ext in ['.docx', '.doc']:
            # Count paragraphs
            metadata['paragraph_count'] = content.count('\n\n') + 1

        elif file_ext == '.md':
            # Count headers
            metadata['header_count'] = content.count('#')

        elif file_ext == '.csv':
            # Count rows (rough estimate)
            metadata['estimated_rows'] = content.count('\n')

        # Content analysis
        if 'magic' in content.lower():
            metadata['contains_magic'] = True
        if 'dragon' in content.lower():
            metadata['contains_dragons'] = True
        if 'kingdom' in content.lower():
            metadata['contains_kingdoms'] = True

        return metadata

    def process_content(self, content: str, filename: str, base_metadata: Dict[str, Any]) -> List[Document]:
        """Process document content with enhanced metadata."""
        # Extract additional metadata
        extracted_metadata = self.extract_metadata(filename, content)

        # Combine metadata
        full_metadata = {**base_metadata, **extracted_metadata}

        # Load documents using appropriate loader
        documents = self.multi_loader.load_from_content(content, filename, full_metadata)

        return documents

    def process_url(self, url: str, base_metadata: Dict[str, Any]) -> List[Document]:
        """Process document from URL."""
        if not self.web_loader.available:
            raise ImportError("Web loading not available")

        documents = self.web_loader.load_from_url(url, base_metadata)
        return documents


# Global instances
document_processor = DocumentProcessor()
multi_loader = MultiFormatDocumentLoader()