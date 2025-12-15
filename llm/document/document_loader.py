"""
DocumentLoader - Handles loading, parsing, and chunking various file types.

Implements Small-to-Big chunking strategy:
- Small chunks for retrieval (better embedding similarity)
- Large parent chunks for generation (richer context for LLM)
"""

import os
import base64
import tempfile
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional, BinaryIO, Tuple
import chardet


class DocumentLoader:
    """Loads, parses, and chunks documents for RAG with Small-to-Big strategy."""
    
    TEXT_EXTENSIONS = {'.txt', '.json', '.csv', '.xml', '.yaml', '.yml', '.log', '.rst'}
    CODE_EXTENSIONS = {'.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c', '.h', 
                       '.cs', '.csproj', '.sln', '.go', '.rs', '.php', '.rb', '.swift', 
                       '.kt', '.sql', '.sh', '.bash', '.css'}
    DOCLING_EXTENSIONS = {'.pdf', '.docx', '.pptx', '.xlsx', '.xls', '.html', '.htm', '.md',
                          '.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.wav', '.mp3', 
                          '.vtt', '.asciidoc', '.adoc'}
    
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
        'image/png': '.png',
        'image/jpeg': '.jpg',
        'image/tiff': '.tiff',
        'image/bmp': '.bmp',
        'audio/wav': '.wav',
        'audio/mpeg': '.mp3',
    }
    
    def __init__(
        self,
        use_docling: bool = True,
        retrieval_chunk_size: int = 128,
        retrieval_chunk_overlap: int = 20,
        context_chunk_size: int = 1024,
        context_chunk_overlap: int = 100,
    ):
        """
        Initialize DocumentLoader with Small-to-Big chunking.
        
        Args:
            use_docling: Use Docling for PDF/DOCX parsing
            retrieval_chunk_size: Small chunk size for embedding (chars)
            retrieval_chunk_overlap: Overlap for small chunks
            context_chunk_size: Large chunk size for LLM context (chars)
            context_chunk_overlap: Overlap for large chunks
        """
        self.use_docling = use_docling
        self.retrieval_chunk_size = retrieval_chunk_size
        self.retrieval_chunk_overlap = retrieval_chunk_overlap
        self.context_chunk_size = context_chunk_size
        self.context_chunk_overlap = context_chunk_overlap
        self.docling_parser = None
        
        if self.use_docling:
            try:
                from docling.document_converter import DocumentConverter
                self.docling_parser = DocumentConverter()
            except ImportError:
                print("[DocumentLoader] Docling not available, using basic parsing")
                self.use_docling = False
        
        print(f"[DocumentLoader] Small-to-Big chunking initialized")
        print(f"[DocumentLoader] Retrieval chunks: {retrieval_chunk_size} chars, overlap {retrieval_chunk_overlap}")
        print(f"[DocumentLoader] Context chunks: {context_chunk_size} chars, overlap {context_chunk_overlap}")
    
    def _generate_unique_id(self, base_name: str) -> str:
        """
        Generate unique document ID.
        
        Args:
            base_name: Base name (e.g., filename stem)
            
        Returns:
            Unique ID in format: basename_uuid8
        """
        unique_suffix = uuid.uuid4().hex[:8]
        return f"{base_name}_{unique_suffix}"
    
    @property
    def supported_extensions(self) -> set:
        extensions = self.TEXT_EXTENSIONS | self.CODE_EXTENSIONS
        if self.use_docling:
            extensions |= self.DOCLING_EXTENSIONS
        return extensions
    
    def is_supported(self, filepath: str) -> bool:
        ext = Path(filepath).suffix.lower()
        return not ext or ext in self.supported_extensions
    
    def load_file(self, filepath: str) -> Dict[str, Any]:
        """Load file and create Small-to-Big chunks."""
        path = Path(filepath)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        if not path.is_file():
            raise ValueError(f"Not a file: {filepath}")
        
        ext = path.suffix.lower()
        
        if ext and ext not in self.supported_extensions:
            raise ValueError(f"Unsupported file type: {ext}")
        
        # Get content based on file type
        if ext in self.DOCLING_EXTENSIONS and self.use_docling:
            content, tables = self._parse_docling(path)
        else:
            content = self._read_text_file(path)
            tables = []
        
        # Generate unique document ID
        doc_id = self._generate_unique_id(path.stem)
        
        # Create Small-to-Big chunks
        retrieval_chunks, context_chunks = self._create_small_to_big_chunks(
            content=content,
            filename=path.name,
            extension=ext
        )
        
        metadata = {
            "extension": ext,
            "size_bytes": path.stat().st_size,
            "retrieval_chunk_count": len(retrieval_chunks),
            "context_chunk_count": len(context_chunks),
        }
        
        return {
            "id": doc_id,
            "filename": path.name,
            "content": content,
            "file_type": ext or "no_extension",
            "metadata": metadata,
            "retrieval_chunks": retrieval_chunks,  # Small chunks for embedding
            "context_chunks": context_chunks,       # Large chunks for LLM
            "tables": tables,
        }
    
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
        
        file_bytes = base64.b64decode(base64_data)
        return self.load_from_bytes(file_bytes, filename, mime_type)
    
    def load_from_stream(
        self,
        file_stream: BinaryIO,
        filename: str,
        mime_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """Load from file stream."""
        file_bytes = file_stream.read()
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
        
        # Get content
        if extension in self.DOCLING_EXTENSIONS and self.use_docling:
            content, tables = self._parse_docling_bytes(file_bytes, extension)
        else:
            content = self._decode_text(file_bytes, filename)
            tables = []
        
        # Generate unique document ID from filename stem
        base_name = Path(filename).stem
        doc_id = self._generate_unique_id(base_name)
        
        # Create Small-to-Big chunks
        retrieval_chunks, context_chunks = self._create_small_to_big_chunks(
            content=content,
            filename=filename,
            extension=extension
        )
        
        metadata = {
            "extension": extension,
            "size_bytes": len(file_bytes),
            "retrieval_chunk_count": len(retrieval_chunks),
            "context_chunk_count": len(context_chunks),
        }
        
        return {
            "id": doc_id,
            "filename": filename,
            "content": content,
            "file_type": extension,
            "metadata": metadata,
            "retrieval_chunks": retrieval_chunks,
            "context_chunks": context_chunks,
            "tables": tables,
        }
    
    def _create_small_to_big_chunks(
        self,
        content: str,
        filename: str,
        extension: str
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Create Small-to-Big chunks.
        
        Returns:
            Tuple of (retrieval_chunks, context_chunks)
            - retrieval_chunks: Small chunks for embedding, with parent_id reference
            - context_chunks: Large chunks for LLM context
        """
        # First, create large context chunks
        context_chunks = self._split_text(
            text=content,
            chunk_size=self.context_chunk_size,
            overlap=self.context_chunk_overlap
        )
        
        # Add IDs and metadata to context chunks
        for idx, chunk in enumerate(context_chunks):
            chunk["id"] = f"ctx_{idx}"
            chunk["metadata"] = {
                "chunk_index": idx,
                "chunk_type": "context",
                "filename": filename,
                "extension": extension,
                "char_start": chunk.get("char_start", 0),
                "char_end": chunk.get("char_end", len(chunk["text"])),
            }
        
        # Now create small retrieval chunks, linked to parent context chunks
        retrieval_chunks = []
        
        for ctx_idx, ctx_chunk in enumerate(context_chunks):
            ctx_text = ctx_chunk["text"]
            ctx_start = ctx_chunk.get("char_start", 0)
            
            # Split context chunk into small retrieval chunks
            small_chunks = self._split_text(
                text=ctx_text,
                chunk_size=self.retrieval_chunk_size,
                overlap=self.retrieval_chunk_overlap
            )
            
            for small_idx, small_chunk in enumerate(small_chunks):
                retrieval_chunks.append({
                    "id": f"ret_{ctx_idx}_{small_idx}",
                    "text": small_chunk["text"],
                    "parent_id": ctx_chunk["id"],  # Link to parent context chunk
                    "metadata": {
                        "chunk_index": len(retrieval_chunks),
                        "chunk_type": "retrieval",
                        "parent_index": ctx_idx,
                        "filename": filename,
                        "extension": extension,
                    }
                })
        
        print(f"[DocumentLoader] Created {len(retrieval_chunks)} retrieval chunks → {len(context_chunks)} context chunks")
        
        return retrieval_chunks, context_chunks
    
    def _split_text(
        self,
        text: str,
        chunk_size: int,
        overlap: int
    ) -> List[Dict[str, Any]]:
        """Split text into chunks with overlap."""
        chunks = []
        
        if not text:
            return chunks
        
        # Try to split on sentence boundaries
        sentences = self._split_sentences(text)
        
        current_chunk = []
        current_size = 0
        char_pos = 0
        chunk_start = 0
        
        for sentence in sentences:
            sentence_len = len(sentence)
            
            if current_size + sentence_len > chunk_size and current_chunk:
                # Save current chunk
                chunk_text = ' '.join(current_chunk)
                chunks.append({
                    "text": chunk_text,
                    "char_start": chunk_start,
                    "char_end": char_pos
                })
                
                # Calculate overlap - keep last few sentences
                overlap_text = ""
                overlap_sentences = []
                for s in reversed(current_chunk):
                    if len(overlap_text) + len(s) < overlap:
                        overlap_sentences.insert(0, s)
                        overlap_text = ' '.join(overlap_sentences)
                    else:
                        break
                
                current_chunk = overlap_sentences
                current_size = len(overlap_text)
                chunk_start = char_pos - len(overlap_text)
            
            current_chunk.append(sentence)
            current_size += sentence_len + 1  # +1 for space
            char_pos += sentence_len + 1
        
        # Don't forget the last chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                "text": chunk_text,
                "char_start": chunk_start,
                "char_end": char_pos
            })
        
        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        import re
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _read_text_file(self, path: Path) -> str:
        """Read text file with encoding detection."""
        try:
            return path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            raw_data = path.read_bytes()
            detected = chardet.detect(raw_data)
            encoding = detected.get('encoding', 'utf-8')
            return raw_data.decode(encoding or 'utf-8', errors='replace')
    
    def _decode_text(self, file_bytes: bytes, filename: str) -> str:
        """Decode bytes to text."""
        try:
            return file_bytes.decode('utf-8')
        except UnicodeDecodeError:
            detected = chardet.detect(file_bytes)
            encoding = detected.get('encoding', 'utf-8')
            return file_bytes.decode(encoding or 'utf-8', errors='replace')
    
    def _parse_docling(self, path: Path) -> Tuple[str, List]:
        """Parse document using Docling."""
        result = self.docling_parser.convert(str(path))
        doc = result.document
        content = doc.export_to_markdown()
        
        tables = []
        if hasattr(doc, 'tables'):
            for idx, table in enumerate(doc.tables):
                try:
                    table_df = table.export_to_dataframe()
                    tables.append({"index": idx, "data": table_df.to_dict(orient="records")})
                except:
                    pass
        
        return content, tables
    
    def _parse_docling_bytes(self, file_bytes: bytes, extension: str) -> Tuple[str, List]:
        """Parse bytes using Docling via temp file."""
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as tmp:
                tmp.write(file_bytes)
                tmp.flush()
                tmp_path = tmp.name
            
            result = self.docling_parser.convert(tmp_path)
            doc = result.document
            content = doc.export_to_markdown()
            
            tables = []
            if hasattr(doc, 'tables'):
                for idx, table in enumerate(doc.tables):
                    try:
                        table_df = table.export_to_dataframe()
                        tables.append({"index": idx, "data": table_df.to_dict(orient="records")})
                    except:
                        pass
            
            return content, tables
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    def _determine_extension(
        self,
        file_bytes: bytes,
        filename: Optional[str],
        mime_type: Optional[str]
    ) -> Optional[str]:
        """Determine file extension."""
        if filename:
            ext = Path(filename).suffix.lower()
            if ext:
                return ext
        
        if mime_type and mime_type in self.MIME_TO_EXT:
            return self.MIME_TO_EXT[mime_type]
        
        # Magic bytes detection
        if file_bytes[:4] == b'%PDF':
            return '.pdf'
        if file_bytes[:2] == b'PK':
            if b'word/' in file_bytes[:1000]:
                return '.docx'
            elif b'xl/' in file_bytes[:1000]:
                return '.xlsx'
            elif b'ppt/' in file_bytes[:1000]:
                return '.pptx'
        
        return '.txt'  # Default to text