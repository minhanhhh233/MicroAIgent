from datetime import datetime
import os
from pathlib import Path
import re
import sys
from typing import Dict, List, Optional, Set
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores.faiss import FAISS
from langchain.schema import Document
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv('API_KEY')

class HybridRetriever:
    def __init__(self, folder_path, api_key):
        self.folder_path = folder_path
        os.environ["NVIDIA_API_KEY"] = api_key
        self.embeddings = self.initialize_nvidia_components()
        self.doc_splits = self.load_and_split_documents()
        self.bm25_retriever, self.faiss_retriever = self.create_retrievers()
        self.hybrid_retriever = self.create_hybrid_retriever()

    def initialize_nvidia_components(self):
        embeddings =NVIDIAEmbeddings(model="nvidia/llama-3.2-nv-embedqa-1b-v2", truncate="END")
        return  embeddings

    def extract_service_name(self, filename):
        """Extract service name from log filename."""
        name = filename.replace('.log', '')
        
        if 'light-oauth2-' in name:
            parts = name.split('-')
            if len(parts) >= 4:
                service_name = '-'.join(parts[2:4])
                return service_name
        
        return name
    
    def extract_timestamp(self, log_line: str) -> Optional[str]:
        """
        Extract timestamp from log line.
        Supports format: 18:13:32.819 Time only (HH:MM:SS.mmm)
        
        """
        
        time_only_pattern = r'^(\d{2}:\d{2}:\d{2}\.\d{3})'
        match = re.match(time_only_pattern, log_line.strip())
        if match:
            time_str = match.group(1)
            try:
                # Parse time
                time_obj = datetime.strptime(time_str, "%H:%M:%S.%f").time()
                
                # Combine with today's date
                # NOTE: If your logs span midnight, you may need date detection logic
                today = datetime.now().date()
                full_datetime = datetime.combine(today, time_obj)
                
                return full_datetime.isoformat()
            except Exception as e:
                print(f"Warning: Failed to parse time '{time_str}': {e}")
                pass
        
        return None

    def parse_log_line(self, log_line: str) -> Optional[Dict]:
        """
        Parse two log formats:
        
        Format 1 (with trace_id):
        18:13:32.819 [XNIO-1 task-2]  flvyKuizR5uaJ8wpB0KASw DEBUG c.n.openapi.ApiNormalisedPath <init> - normalised = /oauth2/code
        
        Format 2 (without trace_id):
        18:13:32.142 [hz._hzInstance_1_dev.partition-operation.thread-11]   DEBUG c.n.oauth.cache.ServiceMapStore load - Load:7a4ca50d
        
        Returns dict with: timestamp, thread, trace_id (optional), log_level, logger, method, message
        """
        
        # Try format 1: with trace_id (has 2 spaces after thread, then trace_id, then space, then log level)
        pattern_with_trace = r'^(\d{2}:\d{2}:\d{2}\.\d{3})\s+\[([^\]]+)\]\s{2}(\S+)\s+(DEBUG|INFO|WARN|ERROR|TRACE)\s+(\S+)\s+(\S+)\s+-\s+(.+)$'
        
        match = re.match(pattern_with_trace, log_line.strip())
        if match:
            time_str, thread, trace_id, log_level, logger, method, message = match.groups()
            
            try:
                time_obj = datetime.strptime(time_str, "%H:%M:%S.%f").time()
                
                return {
                    'timestamp': time_obj.isoformat(),
                    'time_str': time_str,
                    'thread': thread,
                    'trace_id': trace_id,
                    'log_level': log_level,
                    'logger': logger,
                    'method': method,
                    'message': message.strip()
                }
            except Exception:
                pass
        
        # Try format 2: without trace_id (has multiple spaces after thread, then log level)
        pattern_without_trace = r'^(\d{2}:\d{2}:\d{2}\.\d{3})\s+\[([^\]]+)\]\s+(DEBUG|INFO|WARN|ERROR|TRACE)\s+(\S+)\s+(\S+)\s+-\s+(.+)$'
        
        match = re.match(pattern_without_trace, log_line.strip())
        if match:
            time_str, thread, log_level, logger, method, message = match.groups()
            
            try:
                time_obj = datetime.strptime(time_str, "%H:%M:%S.%f").time()
                
                return {
                    'timestamp': time_obj.isoformat(),
                    'time_str': time_str,
                    'thread': thread,
                    'trace_id': "None",  # No trace_id in this format
                    'log_level': log_level,
                    'logger': logger,
                    'method': method,
                    'message': message.strip()
                }
            except Exception:
                pass
        
        # If neither pattern matches, return None
        return None

    def extract_metadata_from_chunk(self, chunk_content: str) -> Dict:
        """
        Extract rich metadata from a chunk of logs.
        
        Extracts:
        - Trace IDs
        - Time range
        - Log levels
        - Threads
        - Endpoints
        - User IDs
        - Error info
        """
        lines = chunk_content.split('\n')
        
        # Collections for metadata
        trace_ids: Set[str] = set()
        log_levels: Set[str] = set()
        threads: Set[str] = set()
        timestamps: List[str] = []
        endpoints: Set[str] = set()
        user_ids: Set[str] = set()
        errors: List[str] = []
        
        for line in lines:
            if not line.strip():
                continue
            
            # Parse structured log line
            parsed = self.parse_log_line(line)
            if parsed:
                trace_ids.add(parsed['trace_id'])
                log_levels.add(parsed['log_level'])
                threads.add(parsed['thread'])
                timestamps.append(parsed['timestamp'])
                
                # Collect errors
                if parsed['log_level'] == 'ERROR':
                    errors.append(line[:200])
                
                # Extract from message
                message = parsed.get('message', '')
                
                # Extract endpoints (e.g., /oauth2/code, /user/login)
                endpoint_pattern = r'(/[\w/\-\.]+)'
                found_endpoints = re.findall(endpoint_pattern, message)
                endpoints.update(found_endpoints)
                
                # Extract user IDs
                user_patterns = [
                    r'user[_-]?id[:\s=]+(\w+)',
                    r'user[:\s]+(\w+)',
                    r'username[:\s=]+(\w+)'
                ]
                for pattern in user_patterns:
                    found_users = re.findall(pattern, message, re.IGNORECASE)
                    user_ids.update(found_users)
        
        # Calculate time range
        metadata = {
            'log_count': len([l for l in lines if l.strip()]),
            'trace_ids': ','.join(sorted(trace_ids))[:500],  # Limit length
            'trace_count': len(trace_ids),
            'primary_trace_id': list(trace_ids)[0] if trace_ids else None,
        }
        
        # Time metadata
        if timestamps:
            sorted_times = sorted(timestamps)
            start_time = sorted_times[0]
            
            metadata.update({
                'start_time': start_time,
            })
        
        # Content metadata
        metadata.update({
            'log_levels': ','.join(sorted(log_levels)),
            'has_errors': 'ERROR' in log_levels,
            'has_warnings': 'WARN' in log_levels,
            'error_count': len(errors),
            'threads': ','.join(sorted(threads))[:200],
            'thread_count': len(threads),
        })
        
        # Context metadata (for correlation!)
        metadata.update({
            'endpoints': ','.join(sorted(endpoints))[:300],
            'endpoint_count': len(endpoints),
            'has_oauth_endpoint': any('/oauth' in ep for ep in endpoints),
            'user_ids': ','.join(sorted(user_ids))[:200],
            'user_count': len(user_ids),
        })
        
        return metadata


    def load_and_split_documents(self):
        """
        Load all log files and split into chunks with RICH metadata extraction.
        """
        all_doc_splits = []
        folder = Path(self.folder_path)
        
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {self.folder_path}")
        
        log_files = list(folder.glob("*.log"))
        
        if not log_files:
            raise ValueError(f"No .log files found in {self.folder_path}")
        
        print(f"  Found {len(log_files)} log files")
        
        for log_file in log_files:
            try:
                # Load the file
                loader = TextLoader(str(log_file), encoding='utf-8')
                docs = loader.load()
                
                service_name = self.extract_service_name(log_file.name)
                
                # Add basic metadata
                for doc in docs:
                    doc.metadata.update({
                        'source': str(log_file),
                        'filename': log_file.name,
                        'service': service_name,
                        'error_type': 'update_password_404_user_not_found'
                    })
                
                # Split into chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=20000, 
                    chunk_overlap=10000
                )
                doc_splits = text_splitter.split_documents(docs)
                
                print(f"    • {service_name}: {len(doc_splits)} chunks")
                
                # Extract metadata from each chunk
                for idx, doc_split in enumerate(doc_splits):
                    # Extract rich metadata from chunk content
                    extracted_metadata = self.extract_metadata_from_chunk(
                        doc_split.page_content
                    )
                    
                    # Add chunk-specific metadata
                    extracted_metadata.update({
                        'chunk_index': idx,
                        'total_chunks': len(doc_splits)
                    })
                    
                    # Merge with existing metadata
                    doc_split.metadata.update(extracted_metadata)
                
                all_doc_splits.extend(doc_splits)
                
            except Exception as e:
                print(f"    ⚠️  Error loading {log_file.name}: {e}")
                import traceback
                print(f"       {traceback.format_exc()}")
                continue
        
        # Sort by timestamp if available
        all_doc_splits.sort(
            key=lambda x: x.metadata.get('start_time') or '9999-99-99'
        )
        
        print(f"\n✅ Total: {len(all_doc_splits)} chunks with metadata")
        
        return all_doc_splits

    def create_retrievers(self):
        bm25_retriever = BM25Retriever.from_documents(self.doc_splits)
        bm25_retriever.k = 5
        
        faiss_vectorstore = FAISS.from_documents(self.doc_splits, self.embeddings)        
        faiss_retriever = faiss_vectorstore.as_retriever(
            search_type="similarity_score_threshold", 
            search_kwargs={
                'score_threshold': 0.8,
                'k': 5
            }
        )
        
        return bm25_retriever, faiss_retriever

    def create_hybrid_retriever(self):
        hybrid_retriever = EnsembleRetriever(retrievers=[self.bm25_retriever, self.faiss_retriever], weights=[0.5, 0.5])
        return hybrid_retriever

    def get_retriever(self):
        return self.hybrid_retriever
    
    def retrieve(self, query, k=5):
        return self.hybrid_retriever.get_relevant_documents(query)
    
    def search_by_time_range(self, service: str, start_time: str, end_time: str) -> List[Document]:
        """
        Search documents by time range (for MCP tools).
        """
        results = []
        for doc in self.doc_splits:
            if doc.metadata.get('service') != service:
                continue
            
            doc_time = doc.metadata.get('start_time')
    
            if doc_time and start_time <= doc_time <= end_time:
                results.append(doc)

        return results


# ============================================
# Global Retriever Manager (Singleton Pattern)
# ============================================
class RetrieverManager:
    """
    Manages a single retriever instance across the application.
    Ensures documents are only ingested once.
    """
    _instance = None
    _retriever = None
    
    @classmethod
    def initialize(cls, folder_path, api_key):
        """
        Initialize the retriever once.
        Call this at application startup.
        """
        if cls._retriever is None:
            print("=" * 60)
            print("INITIALIZING RETRIEVER")
            print("=" * 60)
            cls._retriever = HybridRetriever(folder_path, api_key)
            print("=" * 60)
            print("RETRIEVER READY FOR QUERIES")
            print("=" * 60)
        else:
            print("⚠️  Retriever already initialized, skipping re-ingestion")
        
        return cls._retriever
    
    @classmethod
    def get_retriever(cls):
        """
        Get the initialized retriever.
        Raises error if not initialized.
        """
        if cls._retriever is None:
            raise RuntimeError(
                "Retriever not initialized! Call RetrieverManager.initialize() first"
            )
        return cls._retriever
    
    @classmethod
    def reset(cls):
        """Reset the retriever (for testing or re-initialization)."""
        cls._retriever = None
        print("✅ Retriever reset")

