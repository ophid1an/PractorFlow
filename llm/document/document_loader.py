"""
DocumentLoader - Handles loading, parsing, and chunking various file types.

Supports:
- Text files: .txt, .json, .csv, .xml, .yaml, .yml, .log, .rst
- Code files: .py, .js, .ts, .jsx, .tsx, .java, .cpp, .c, .h, .cs, .csproj, .sln,
              .go, .rs, .php, .rb, .swift, .kt, .sql, .sh, .bash, .css
- Docling files: .pdf, .docx, .pptx, .xlsx, .html, .md, images, audio
- No extension files: Treated as text, rejected if binary
- Base64 uploads: Web upload support with auto-detection
- Stream uploads: FastAPI UploadFile and other file streams

Uses Docling's HybridChunker for intelligent document chunking.
"""

import os
import base64
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional, BinaryIO, Any as TokenizerType
import chardet
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker


class DocumentLoader:
    """Loads, parses, and chunks documents for RAG."""
    
    TEXT_EXTENSIONS = {
        '.txt', '.json', '.csv', '.xml', 
        '.yaml', '.yml', '.log', '.rst'
    }
    
    CODE_EXTENSIONS = {
        '.py', '.js', '.ts', '.jsx', '.tsx',
        '.java', '.cpp', '.c', '.h', 
        '.cs', '.csproj', '.sln',
        '.go', '.rs', '.php', '.rb', 
        '.swift', '.kt', '.sql',
        '.sh', '.bash', '.css'
    }
    
    DOCLING_EXTENSIONS = {
        '.pdf', '.docx', '.pptx', '.xlsx', '.xls',
        '.html', '.htm', '.md',
        '.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp',
        '.wav', '.mp3', '.vtt',
        '.asciidoc', '.adoc',
    }
    
    MIME_TO_EXT = {
        'application/pdf': '.pdf',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '.docx',
        'application/vnd.openxmlformats-officedocument.presentationml.presentation': '.pptx',
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': '.xlsx',
        'application/vnd.ms-excel': '.xls',
        'text/html': '.html',
        'text/markdown': '.md',
        'text/plain': '.txt',
        'text/csv': '.csv',
        'application/json': '.json',
        'application/xml': '.xml',
        'text/xml': '.xml',
        'application/x-yaml': '.yaml',
        'text/yaml': '.yaml',
        'image/png': '.png',
        'image/jpeg': '.jpg',
        'image/tiff': '.tiff',
        'image/bmp': '.bmp',
        'audio/wav': '.wav',
        'audio/mpeg': '.mp3',
        'text/vtt': '.vtt',
    }
    
    def __init__(
        self,
        use_docling: bool = True,
        tokenizer: Optional[TokenizerType] = None,
        chunk_size: int = 512,
        chunk_overlap: int = 50
    ):
        """
        Initialize DocumentLoader.
        
        Args:
            use_docling: Use Docling for PDF/DOCX parsing
            tokenizer: Model tokenizer for chunking (from LLM)
            chunk_size: Target chunk size in tokens
            chunk_overlap: Overlap between chunks in tokens
        """
        self.use_docling = use_docling
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.docling_parser = None
        self.chunker = None
        
        if self.use_docling:
            self.docling_parser = DocumentConverter()
            self._initialize_chunker()
    
    def _initialize_chunker(self):
        """Initialize Docling's HybridChunker with model tokenizer."""
        if not self.tokenizer:
            return
        
        self.chunker = HybridChunker(
            tokenizer=self.tokenizer,
            max_tokens=self.chunk_size,
            overlap=self.chunk_overlap
        )
        print(f"[DocumentLoader] HybridChunker initialized (size={self.chunk_size}, overlap={self.chunk_overlap})")
    
    @property
    def supported_extensions(self) -> set:
        """Get all supported file extensions."""
        extensions = self.TEXT_EXTENSIONS | self.CODE_EXTENSIONS
        if self.use_docling:
            extensions |= self.DOCLING_EXTENSIONS
        return extensions
    
    def is_supported(self, filepath: str) -> bool:
        """Check if file is supported."""
        ext = Path(filepath).suffix.lower()
        return not ext or ext in self.supported_extensions
    
    def load_file(self, filepath: str) -> Dict[str, Any]:
        """Load and chunk a single file."""
        path = Path(filepath)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        if not path.is_file():
            raise ValueError(f"Not a file: {filepath}")
        
        ext = path.suffix.lower()
        
        if ext and ext not in self.supported_extensions:
            raise ValueError(f"Unsupported file type: {ext}")
        
        if not ext:
            return self._load_text_file(path, "no_extension")
        elif ext in self.TEXT_EXTENSIONS:
            return self._load_text_file(path, "text")
        elif ext in self.CODE_EXTENSIONS:
            return self._load_text_file(path, "code")
        elif ext in self.DOCLING_EXTENSIONS:
            if not self.use_docling:
                raise ValueError(f"Docling not available for {ext}")
            return self._load_docling_file(path)
        else:
            raise ValueError(f"Unsupported extension: {ext}")
    
    def load_directory(
        self,
        directory: str,
        recursive: bool = True,
        skip_unsupported: bool = True
    ) -> List[Dict[str, Any]]:
        """Load all supported files from directory."""
        path = Path(directory)
        
        if not path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        if not path.is_dir():
            raise ValueError(f"Not a directory: {directory}")
        
        documents = []
        pattern = "**/*" if recursive else "*"
        
        for file_path in path.glob(pattern):
            if not file_path.is_file():
                continue
            
            try:
                doc = self.load_file(str(file_path))
                documents.append(doc)
                chunk_count = len(doc.get('chunks', []))
                print(f"✓ Loaded: {file_path.name} ({chunk_count} chunks)")
            except ValueError as e:
                if skip_unsupported:
                    print(f"⊘ Skipped: {file_path.name} - {str(e)}")
                else:
                    raise
            except Exception as e:
                print(f"✗ Error: {file_path.name} - {str(e)}")
                if not skip_unsupported:
                    raise
        
        return documents
    
    def load_from_stream(
        self,
        file_stream: BinaryIO,
        filename: str,
        mime_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """Load from file stream."""
        file_bytes = file_stream.read()
        return self.load_from_bytes(file_bytes, filename, mime_type)
    
    def load_from_base64(
        self,
        base64_data: str,
        filename: Optional[str] = None,
        mime_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """Load from base64 data."""
        if base64_data.startswith('data:'):
            if ';base64,' in base64_data:
                prefix, base64_data = base64_data.split(';base64,', 1)
                if not mime_type:
                    mime_type = prefix.split(':', 1)[1]
            else:
                raise ValueError("Invalid data URI format")
        
        try:
            file_bytes = base64.b64decode(base64_data)
        except Exception as e:
            raise ValueError(f"Failed to decode base64: {str(e)}")
        
        return self.load_from_bytes(file_bytes, filename, mime_type)
    
    def load_from_bytes(
        self,
        file_bytes: bytes,
        filename: str,
        mime_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """Load from raw bytes."""
        extension = self._determine_extension(file_bytes, filename, mime_type)
        
        if not extension:
            raise ValueError("Could not determine file type")
        
        if not filename:
            filename = f"upload{extension}"
        
        file_path = Path(filename)
        if not file_path.suffix or file_path.suffix.lower() != extension:
            filename = f"{file_path.stem}{extension}"
        
        if extension not in self.supported_extensions:
            raise ValueError(f"Unsupported file type: {extension}")
        
        if extension in self.TEXT_EXTENSIONS:
            return self._load_text_from_bytes(file_bytes, filename, extension, "text")
        elif extension in self.CODE_EXTENSIONS:
            return self._load_text_from_bytes(file_bytes, filename, extension, "code")
        elif extension in self.DOCLING_EXTENSIONS:
            if not self.use_docling:
                raise ValueError(f"Docling not available for {extension}")
            return self._load_docling_from_bytes(file_bytes, filename, extension)
        else:
            raise ValueError(f"Unsupported extension: {extension}")
    
    def _determine_extension(
        self,
        file_bytes: bytes,
        filename: Optional[str] = None,
        mime_type: Optional[str] = None
    ) -> Optional[str]:
        """Determine file extension."""
        if filename:
            ext = Path(filename).suffix.lower()
            if ext:
                return ext
        
        if mime_type and mime_type in self.MIME_TO_EXT:
            return self.MIME_TO_EXT[mime_type]
        
        return self._detect_from_magic_bytes(file_bytes)
    
    def _detect_from_magic_bytes(self, file_bytes: bytes) -> Optional[str]:
        """Detect file type from magic bytes."""
        if len(file_bytes) < 4:
            return None
        
        if file_bytes[:4] == b'%PDF':
            return '.pdf'
        if file_bytes[:8] == b'\x89PNG\r\n\x1a\n':
            return '.png'
        if file_bytes[:2] == b'\xff\xd8':
            return '.jpg'
        if file_bytes[:2] in (b'II', b'MM'):
            return '.tiff'
        if file_bytes[:2] == b'BM':
            return '.bmp'
        if file_bytes[:2] == b'PK':
            if b'word/' in file_bytes[:1000]:
                return '.docx'
            elif b'xl/' in file_bytes[:1000]:
                return '.xlsx'
            elif b'ppt/' in file_bytes[:1000]:
                return '.pptx'
        if file_bytes[:4] == b'RIFF' and file_bytes[8:12] == b'WAVE':
            return '.wav'
        if file_bytes[:2] == b'\xff\xfb' or file_bytes[:3] == b'ID3':
            return '.mp3'
        
        return None
    
    def _chunk_text_simple(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Simple text chunking fallback."""
        chunks = []
        paragraphs = text.split('\n\n')
        
        current_chunk = []
        current_size = 0
        
        for para in paragraphs:
            para_size = len(para.split())
            
            if current_size + para_size > self.chunk_size and current_chunk:
                chunk_text = '\n\n'.join(current_chunk)
                chunks.append({
                    "text": chunk_text,
                    "metadata": {**metadata, "chunk_index": len(chunks)}
                })
                
                current_chunk = [current_chunk[-1]] if self.chunk_overlap > 0 else []
                current_size = len(current_chunk[0].split()) if current_chunk else 0
            
            current_chunk.append(para)
            current_size += para_size
        
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            chunks.append({
                "text": chunk_text,
                "metadata": {**metadata, "chunk_index": len(chunks)}
            })
        
        return chunks if chunks else [{"text": text, "metadata": metadata}]
    
    def _load_text_from_bytes(
        self,
        file_bytes: bytes,
        filename: str,
        extension: str,
        category: str
    ) -> Dict[str, Any]:
        """Load text file from bytes."""
        try:
            content = file_bytes.decode('utf-8')
        except UnicodeDecodeError:
            try:
                detected = chardet.detect(file_bytes)
                encoding = detected['encoding']
                if encoding:
                    content = file_bytes.decode(encoding)
                else:
                    raise ValueError(f"Could not detect encoding: {filename}")
            except Exception as e:
                raise ValueError(f"Failed to decode file: {filename}\n{str(e)}")
        
        doc_id = Path(filename).stem
        
        metadata = {
            "size_bytes": len(file_bytes),
            "extension": extension,
            "source": "bytes",
            "filename": filename
        }
        
        chunks = self._chunk_text_simple(content, metadata)
        
        return {
            "id": doc_id,
            "filename": filename,
            "content": content,
            "file_type": extension,
            "category": category,
            "metadata": metadata,
            "chunks": chunks
        }
    
    def _load_docling_from_bytes(
        self,
        file_bytes: bytes,
        filename: str,
        extension: str
    ) -> Dict[str, Any]:
        """Load and chunk file using Docling."""
        if not self.docling_parser:
            raise ValueError("Docling parser not initialized")
        
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as tmp:
                tmp.write(file_bytes)
                tmp.flush()
                tmp_path = tmp.name
            
            result = self.docling_parser.convert(tmp_path)
            doc = result.document
            content = doc.export_to_markdown()
            
            # Chunk using HybridChunker
            chunks = []
            if self.chunker:
                try:
                    doc_chunks = list(self.chunker.chunk(doc))
                    
                    for idx, chunk in enumerate(doc_chunks):
                        chunk_metadata = {
                            "chunk_index": idx,
                            "filename": filename,
                            "extension": extension,
                            "source": "docling"
                        }
                        
                        # Extract page number if available
                        if hasattr(chunk, 'meta') and hasattr(chunk.meta, 'doc_items'):
                            doc_items = chunk.meta.doc_items
                            if doc_items and hasattr(doc_items[0], 'prov') and doc_items[0].prov:
                                chunk_metadata["page"] = doc_items[0].prov[0].page_no
                        
                        chunks.append({
                            "text": chunk.text,
                            "metadata": chunk_metadata
                        })
                    
                    print(f"[DocumentLoader] Created {len(chunks)} chunks from {filename}")
                    
                except Exception as e:
                    print(f"[DocumentLoader] Chunking failed, using simple chunking: {e}")
                    chunks = self._chunk_text_simple(content, {"filename": filename})
            else:
                chunks = self._chunk_text_simple(content, {"filename": filename})
            
            # Extract tables
            tables = []
            if hasattr(doc, 'tables'):
                for table_idx, table in enumerate(doc.tables):
                    try:
                        table_df = table.export_to_dataframe()
                        tables.append({
                            "index": table_idx,
                            "data": table_df.to_dict(orient="records")
                        })
                    except Exception:
                        pass
            
            metadata = {
                "extension": extension,
                "size_bytes": len(file_bytes),
                "has_tables": len(tables) > 0,
                "table_count": len(tables),
                "source": "docling",
                "chunk_count": len(chunks)
            }
            
            if hasattr(doc, 'pages'):
                metadata["page_count"] = len(doc.pages)
            
            return {
                "id": Path(filename).stem,
                "filename": filename,
                "content": content,
                "file_type": extension,
                "category": "docling",
                "metadata": metadata,
                "tables": tables,
                "chunks": chunks
            }
            
        except Exception as e:
            raise ValueError(f"Failed to parse document '{filename}': {str(e)}")
            
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
    
    def _load_text_file(self, path: Path, category: str) -> Dict[str, Any]:
        """Load text file."""
        try:
            content = path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            try:
                raw_data = path.read_bytes()
                detected = chardet.detect(raw_data)
                encoding = detected['encoding']
                
                if encoding:
                    content = raw_data.decode(encoding)
                else:
                    raise ValueError(f"Could not detect encoding: {path.name}")
            except Exception as e:
                raise ValueError(f"Failed to decode file: {path.name}\n{str(e)}")
        
        doc_id = path.stem if path.stem else path.name
        
        metadata = {
            "size_bytes": path.stat().st_size,
            "extension": path.suffix.lower() if path.suffix else None,
            "filename": path.name
        }
        
        chunks = self._chunk_text_simple(content, metadata)
        
        return {
            "id": doc_id,
            "filename": path.name,
            "content": content,
            "file_type": path.suffix.lower() if path.suffix else "no_extension",
            "category": category,
            "metadata": metadata,
            "chunks": chunks
        }
    
    def _load_docling_file(self, path: Path) -> Dict[str, Any]:
        """Load and chunk file using Docling."""
        if not self.docling_parser:
            raise ValueError("Docling parser not initialized")
        
        result = self.docling_parser.convert(str(path))
        doc = result.document
        content = doc.export_to_markdown()
        
        # Chunk using HybridChunker
        chunks = []
        if self.chunker:
            try:
                doc_chunks = list(self.chunker.chunk(doc))
                
                for idx, chunk in enumerate(doc_chunks):
                    chunk_metadata = {
                        "chunk_index": idx,
                        "filename": path.name,
                        "extension": path.suffix.lower(),
                        "source": "docling"
                    }
                    
                    # Extract page number if available
                    if hasattr(chunk, 'meta') and hasattr(chunk.meta, 'doc_items'):
                        doc_items = chunk.meta.doc_items
                        if doc_items and hasattr(doc_items[0], 'prov') and doc_items[0].prov:
                            chunk_metadata["page"] = doc_items[0].prov[0].page_no
                    
                    chunks.append({
                        "text": chunk.text,
                        "metadata": chunk_metadata
                    })
                
                print(f"[DocumentLoader] Created {len(chunks)} chunks from {path.name}")
                
            except Exception as e:
                print(f"[DocumentLoader] Chunking failed, using simple chunking: {e}")
                chunks = self._chunk_text_simple(content, {"filename": path.name})
        else:
            chunks = self._chunk_text_simple(content, {"filename": path.name})
        
        # Extract tables
        tables = []
        if hasattr(doc, 'tables'):
            for table_idx, table in enumerate(doc.tables):
                try:
                    table_df = table.export_to_dataframe()
                    tables.append({
                        "index": table_idx,
                        "data": table_df.to_dict(orient="records")
                    })
                except Exception:
                    pass
        
        metadata = {
            "extension": path.suffix.lower(),
            "size_bytes": path.stat().st_size,
            "has_tables": len(tables) > 0,
            "table_count": len(tables),
            "chunk_count": len(chunks)
        }
        
        if hasattr(doc, 'pages'):
            metadata["page_count"] = len(doc.pages)
        
        return {
            "id": path.stem,
            "filename": path.name,
            "content": content,
            "file_type": path.suffix.lower(),
            "category": "docling",
            "metadata": metadata,
            "tables": tables,
            "chunks": chunks
        }
    
    def get_statistics(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about loaded documents."""
        total_docs = len(documents)
        total_chars = sum(len(doc["content"]) for doc in documents)
        total_chunks = sum(len(doc.get("chunks", [])) for doc in documents)
        
        categories = {}
        file_types = {}
        
        for doc in documents:
            cat = doc.get("category", "unknown")
            categories[cat] = categories.get(cat, 0) + 1
            
            ft = doc.get("file_type", "unknown")
            file_types[ft] = file_types.get(ft, 0) + 1
        
        return {
            "total_documents": total_docs,
            "total_characters": total_chars,
            "total_chunks": total_chunks,
            "avg_chunks_per_doc": total_chunks / total_docs if total_docs > 0 else 0,
            "categories": categories,
            "file_types": file_types
        }