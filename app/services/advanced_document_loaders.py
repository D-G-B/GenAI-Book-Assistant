"""
Advanced document loaders for different file types using LangChain.
Now includes EPUB support for e-books.
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


class EpubLoader:
    """
    Custom EPUB loader using ebooklib for better chapter awareness.
    Falls back to UnstructuredEPubLoader if ebooklib is not available.
    """

    def __init__(self):
        self.ebooklib_available = False
        self.bs4_available = False
        self.unstructured_available = False

        # Check for ebooklib
        try:
            import ebooklib
            from ebooklib import epub
            self.ebooklib_available = True
        except ImportError:
            print("⚠️ ebooklib not available. Install with: pip install ebooklib")

        # Check for BeautifulSoup (needed to parse HTML from epub)
        try:
            from bs4 import BeautifulSoup
            self.bs4_available = True
        except ImportError:
            print("⚠️ BeautifulSoup not available. Install with: pip install beautifulsoup4")

        # Check for unstructured fallback
        try:
            from langchain_community.document_loaders import UnstructuredEPubLoader
            self.unstructured_available = True
        except ImportError:
            pass

    def is_available(self) -> bool:
        """Check if EPUB loading is available."""
        return (self.ebooklib_available and self.bs4_available) or self.unstructured_available

    def load(self, file_path: str, metadata: Dict[str, Any]) -> List[Document]:
        """
        Load an EPUB file and return documents with chapter-aware metadata.

        Args:
            file_path: Path to the EPUB file
            metadata: Base metadata to include in all documents

        Returns:
            List of Document objects, one per chapter
        """
        if self.ebooklib_available and self.bs4_available:
            return self._load_with_ebooklib(file_path, metadata)
        elif self.unstructured_available:
            return self._load_with_unstructured(file_path, metadata)
        else:
            raise ImportError(
                "No EPUB loader available. Install ebooklib and beautifulsoup4: "
                "pip install ebooklib beautifulsoup4"
            )

    def _load_with_ebooklib(self, file_path: str, metadata: Dict[str, Any]) -> List[Document]:
        """Load EPUB using ebooklib for chapter-aware extraction."""
        import ebooklib
        from ebooklib import epub
        from bs4 import BeautifulSoup

        documents = []

        try:
            # Read the EPUB file
            book = epub.read_epub(file_path)

            # Extract book metadata
            book_title = book.get_metadata('DC', 'title')
            book_title = book_title[0][0] if book_title else Path(file_path).stem

            book_author = book.get_metadata('DC', 'creator')
            book_author = book_author[0][0] if book_author else 'Unknown'

            # Update metadata with book info
            epub_metadata = metadata.copy()
            epub_metadata.update({
                'book_title': book_title,
                'book_author': book_author,
                'source_type': 'epub'
            })

            # Get the spine (reading order)
            spine_ids = [item[0] for item in book.spine]

            # Track chapter numbers
            chapter_num = 0

            # Process items in spine order for correct chapter sequence
            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    # Check if this item is in the spine (main content)
                    if item.get_id() in spine_ids:
                        content = item.get_content()

                        # Parse HTML content
                        soup = BeautifulSoup(content, 'html.parser')

                        # Extract text from paragraphs
                        text_parts = []

                        # Get title/header if present
                        title_elem = soup.find(['h1', 'h2', 'h3', 'title'])
                        chapter_title = title_elem.get_text(strip=True) if title_elem else None

                        # Extract all paragraph text
                        for tag in ['p', 'div', 'span']:
                            for element in soup.find_all(tag):
                                text = element.get_text(' ', strip=True)
                                if text and len(text) > 10:  # Filter out very short fragments
                                    text_parts.append(text)

                        # Combine text
                        full_text = '\n\n'.join(text_parts)

                        # Skip empty or very short chapters (likely TOC, copyright, etc.)
                        if len(full_text.strip()) < 100:
                            continue

                        chapter_num += 1

                        # Detect if this is reference material
                        item_name = item.get_name().lower()
                        is_reference = self._detect_reference_section(item_name, chapter_title, full_text)

                        # Create chapter metadata
                        chapter_metadata = epub_metadata.copy()
                        chapter_metadata.update({
                            'chapter_number': chapter_num,
                            'chapter_title': chapter_title or f'Chapter {chapter_num}',
                            'item_name': item.get_name(),
                            'is_reference_material': is_reference,
                            'content_length': len(full_text)
                        })

                        documents.append(Document(
                            page_content=full_text,
                            metadata=chapter_metadata
                        ))

            print(f"✅ Loaded EPUB with {len(documents)} chapters using ebooklib")
            return documents

        except Exception as e:
            print(f"⚠️ Error loading EPUB with ebooklib: {e}")
            # Try fallback
            if self.unstructured_available:
                print("   Trying UnstructuredEPubLoader fallback...")
                return self._load_with_unstructured(file_path, metadata)
            raise

    def _detect_reference_section(
        self,
        item_name: str,
        chapter_title: Optional[str],
        content: str
    ) -> bool:
        """Detect if a chapter is reference material (glossary, appendix, etc.)."""
        reference_markers = [
            'glossary', 'appendix', 'appendices', 'dramatis personae',
            'cast of characters', 'index', 'notes', 'bibliography',
            'pronunciation guide', 'world guide', 'character list',
            'map', 'timeline', 'afterword', 'acknowledgments'
        ]

        # Check item name
        if any(marker in item_name for marker in reference_markers):
            return True

        # Check chapter title
        if chapter_title:
            title_lower = chapter_title.lower()
            if any(marker in title_lower for marker in reference_markers):
                return True

        # Check content start (first 500 chars)
        content_start = content[:500].lower()
        if any(marker in content_start for marker in reference_markers):
            return True

        return False

    def _load_with_unstructured(self, file_path: str, metadata: Dict[str, Any]) -> List[Document]:
        """Fallback loader using UnstructuredEPubLoader."""
        from langchain_community.document_loaders import UnstructuredEPubLoader

        try:
            # Use elements mode for better structure
            loader = UnstructuredEPubLoader(file_path, mode="elements")
            docs = loader.load()

            # Add custom metadata
            for i, doc in enumerate(docs):
                doc.metadata.update(metadata)
                doc.metadata['source_type'] = 'epub'
                doc.metadata['element_index'] = i

            print(f"✅ Loaded EPUB with {len(docs)} elements using UnstructuredEPubLoader")
            return docs

        except Exception as e:
            print(f"❌ Error loading EPUB with unstructured: {e}")
            raise


class MultiFormatDocumentLoader:
    """Enhanced document loader that supports multiple file formats including EPUB."""

    def __init__(self):
        # Initialize EPUB loader
        self.epub_loader = EpubLoader()

        self.supported_formats = {
            '.txt': self._load_text,
            '.pdf': self._load_pdf,
            '.docx': self._load_word,
            '.doc': self._load_word,
            '.md': self._load_markdown,
            '.csv': self._load_csv,
            '.json': self._load_json,
            '.epub': self._load_epub,
        }

    def get_supported_formats(self) -> List[str]:
        """Return list of supported file extensions."""
        formats = list(self.supported_formats.keys())

        # Only include epub if loader is available
        if not self.epub_loader.is_available():
            formats = [f for f in formats if f != '.epub']

        return formats

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

    def load_from_bytes(self, content: bytes, filename: str, metadata: Dict[str, Any]) -> List[Document]:
        """
        Load document from binary content.
        Useful for EPUB and other binary formats.
        """
        file_ext = Path(filename).suffix.lower()

        # Determine write mode based on format
        binary_formats = {'.epub', '.pdf', '.docx', '.doc'}
        mode = 'wb' if file_ext in binary_formats else 'w'

        with tempfile.NamedTemporaryFile(mode=mode, suffix=file_ext, delete=False) as temp_file:
            if mode == 'wb':
                temp_file.write(content)
            else:
                temp_file.write(content.decode('utf-8'))
            temp_path = temp_file.name

        try:
            return self.load_from_file(temp_path, metadata)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

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

    def _load_epub(self, file_path: str, metadata: Dict[str, Any]) -> List[Document]:
        """Load EPUB file with chapter awareness."""
        if not self.epub_loader.is_available():
            raise ImportError(
                "EPUB loading requires ebooklib and beautifulsoup4. "
                "Install with: pip install ebooklib beautifulsoup4"
            )

        return self.epub_loader.load(file_path, metadata)


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

        elif file_ext == '.epub':
            # EPUB-specific metadata is handled by the loader
            metadata['format'] = 'ebook'

        # Content analysis for fantasy/sci-fi detection
        content_lower = content.lower()
        fantasy_terms = ['magic', 'dragon', 'kingdom', 'wizard', 'spell', 'sword']
        scifi_terms = ['spaceship', 'planet', 'alien', 'galaxy', 'starship', 'colony']

        fantasy_count = sum(1 for term in fantasy_terms if term in content_lower)
        scifi_count = sum(1 for term in scifi_terms if term in content_lower)

        if fantasy_count >= 2:
            metadata['detected_genre'] = 'fantasy'
        elif scifi_count >= 2:
            metadata['detected_genre'] = 'science_fiction'

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

    def process_bytes(self, content: bytes, filename: str, base_metadata: Dict[str, Any]) -> List[Document]:
        """Process binary document content (for EPUB, PDF, etc.)."""
        file_ext = Path(filename).suffix.lower()

        # For binary formats, use load_from_bytes
        documents = self.multi_loader.load_from_bytes(content, filename, base_metadata)

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