# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
StateLM Tools - Tools for document manipulation, note-taking, and context management.

These tools provide stateful operations for long-context agent workflows, including:
- Document loading and analysis
- Chunk-based document indexing with Elasticsearch
- Note-taking and knowledge management
- Context budget tracking
"""

import asyncio
import copy
import json
import logging
import os
import threading
import uuid
from contextlib import ExitStack
from typing import Any, Optional

import ray
from elasticsearch import Elasticsearch, helpers

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DocStateManager:
    """
    Document state manager for StateLM tools.
    
    This manager maintains the shared document state across all StateLM tool instances,
    including document content, index, and search history.
    """
    
    def __init__(self, tokenizer, document_content: str = ""):
        self.tokenizer = tokenizer
        self.document_content = document_content
        self.index = []
        self.keywords_searched = set()
        self.chunk_pointer = [-1, 0]
        
        # Elasticsearch configuration
        self._es: Optional[Elasticsearch] = None
        self._es_index_name: str = os.getenv('ES_INDEX_NAME', 'lc_agent_document')
        self._es_host: str = os.getenv('ES_HOST', 'http://localhost:9200')
        self._es_user: Optional[str] = os.getenv('ES_USER')
        self._es_pass: Optional[str] = os.getenv('ES_PASS')
        self._es_api_key: Optional[str] = os.getenv('ES_API_KEY')
        self._es_ca_cert: Optional[str] = os.getenv('ES_CA_CERT')
        self._doc_id: Optional[str] = None
        
        # Tokenize document once
        if document_content:
            self.encoded_doc = self.tokenizer(
                self.document_content,
                return_offsets_mapping=True,
                add_special_tokens=False
            )
        else:
            self.encoded_doc = {"input_ids": [], "offset_mapping": []}
        
        logger.info(f"DocStateManager initialized with document length: {len(self.encoded_doc['input_ids'])} tokens")
    
    def _get_es(self) -> Elasticsearch:
        """Get or create Elasticsearch client."""
        if self._es is None:
            kwargs = {}
            # Only add authentication if explicitly provided
            if self._es_api_key:
                kwargs['api_key'] = self._es_api_key
            elif self._es_user and self._es_pass:
                kwargs['basic_auth'] = (self._es_user, self._es_pass)
            # Only add SSL settings if CA cert is provided (for HTTPS connections)
            if self._es_ca_cert:
                kwargs['ca_certs'] = self._es_ca_cert
            # Create client - will use HTTP or HTTPS based on URL scheme
            self._es = Elasticsearch(self._es_host, **kwargs)
        return self._es
    
    def _ensure_es_index(self):
        """Ensure Elasticsearch index exists."""
        es = self._get_es()
        if es.indices.exists(index=self._es_index_name):
            return
        es.indices.create(
            index=self._es_index_name,
            settings={
                'index': {
                    'analysis': {
                        'analyzer': {
                            'default': {'type': 'standard'}
                        }
                    }
                }
            },
            mappings={
                'properties': {
                    'doc_id': {'type': 'keyword'},
                    'chunk_id': {'type': 'integer'},
                    'content': {'type': 'text'},
                    'start_pos': {'type': 'integer'},
                    'end_pos': {'type': 'integer'}
                }
            }
        )
        logger.info(f"Created Elasticsearch index '{self._es_index_name}'.")
    
    def _bulk_index_chunks(self):
        """Bulk index chunks to Elasticsearch."""
        es = self._get_es()
        actions = ({
            '_op_type': 'index',
            '_index': self._es_index_name,
            '_id': f"{self._doc_id}:{c['chunk_id']}",
            'doc_id': self._doc_id,
            'chunk_id': c['chunk_id'],
            'content': c['content'],
            'start_pos': c['start_pos'],
            'end_pos': c['end_pos'],
        } for c in self.index)
        helpers.bulk(es, actions)
        es.indices.refresh(index=self._es_index_name)
        logger.info(f"Indexed {len(self.index)} chunks into Elasticsearch.")
    
    def clear_current_document(self):
        """Clear current document from Elasticsearch and reset state."""
        if not self._doc_id:
            return {"message": "No active document to clear."}
        es = self._get_es()
        es.delete_by_query(
            index=self._es_index_name,
            query={"term": {"doc_id": self._doc_id}},
            refresh=True,
        )
        self.index = []
        self.keywords_searched = set()
        self._doc_id = None
        self.chunk_pointer = [-1, 0]
        return {"message": "Cleared current document."}


class AnalyzeTextTool(BaseTool):
    """Analyze the length of the attached document content."""
    
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}
    
    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid.uuid4())
        self._instance_dict[instance_id] = {}
        return instance_id, ToolResponse()
    
    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        """Analyze document and return token count."""
        try:
            # Get state manager from kwargs (passed by agent loop)
            doc_state_manager = kwargs.get('doc_state_manager')
            if not doc_state_manager:
                return ToolResponse(text=json.dumps({"error": "Document State Manager not available"})), 0.0, {}
            
            total_tokens = len(doc_state_manager.encoded_doc["input_ids"])
            result = {
                "file_name": "attached_document.txt",
                "total_tokens": total_tokens
            }
            
            return ToolResponse(text=json.dumps(result, ensure_ascii=False)), 0.0, {}
        except Exception as e:
            logger.error(f"Error in analyzeText: {e}")
            return ToolResponse(text=json.dumps({"error": str(e)})), 0.0, {}
    
    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]


class LoadDocumentTool(BaseTool):
    """Load the full document content."""
    
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}
    
    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid.uuid4())
        self._instance_dict[instance_id] = {}
        return instance_id, ToolResponse()
    
    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        """Load and return full document content."""
        try:
            doc_state_manager = kwargs.get('doc_state_manager')
            if not doc_state_manager:
                return ToolResponse(text=json.dumps({"error": "Document State Manager not available"})), 0.0, {}
            
            if not doc_state_manager.document_content:
                return ToolResponse(text=json.dumps({"error": "Document content is empty."})), 0.0, {}
            
            result = {"document_content": doc_state_manager.document_content}
            return ToolResponse(text=json.dumps(result, ensure_ascii=False)), 0.0, {}
        except Exception as e:
            logger.error(f"Error in loadDocument: {e}")
            return ToolResponse(text=json.dumps({"error": str(e)})), 0.0, {}
    
    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]


class BuildIndexTool(BaseTool):
    """Split document into fixed-size chunks and build a searchable index."""
    
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}
    
    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid.uuid4())
        self._instance_dict[instance_id] = {}
        return instance_id, ToolResponse()
    
    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        """Build document index with specified chunk size and overlap."""
        try:
            doc_state_manager = kwargs.get('doc_state_manager')
            if not doc_state_manager:
                return ToolResponse(text=json.dumps({"error": "Document State Manager not available"})), 0.0, {}
            
            chunk_size = parameters.get("chunk_size", 8000)
            overlap = parameters.get("overlap", 0)
            
            if chunk_size <= 0:
                return ToolResponse(text=json.dumps({"error": "chunk_size must be > 0"})), 0.0, {}
            if overlap < 0:
                return ToolResponse(text=json.dumps({"error": "overlap must be >= 0"})), 0.0, {}
            if overlap >= chunk_size:
                return ToolResponse(text=json.dumps({"error": "overlap must be < chunk_size"})), 0.0, {}
            
            input_ids = doc_state_manager.encoded_doc["input_ids"]
            offsets = doc_state_manager.encoded_doc["offset_mapping"]
            
            doc_state_manager.index = []
            doc_state_manager._doc_id = uuid.uuid4().hex
            
            start_token = 0
            chunk_id = 0
            while start_token < len(input_ids):
                end_token = min(start_token + chunk_size, len(input_ids))
                
                chunk_offsets = offsets[start_token:end_token]
                char_start = chunk_offsets[0][0]
                char_end = chunk_offsets[-1][1]
                
                chunk_content = doc_state_manager.document_content[char_start:char_end]
                chunk_data = {
                    "chunk_id": chunk_id,
                    "content": chunk_content,
                    "start_pos": start_token,
                    "end_pos": end_token
                }
                doc_state_manager.index.append(chunk_data)
                
                chunk_id += 1
                start_token += chunk_size - overlap
            
            doc_state_manager.keywords_searched = set()
            doc_state_manager.chunk_pointer = [-1, 0]
            
            # Index into Elasticsearch (run in executor to avoid blocking event loop)
            try:
                # Run blocking Elasticsearch calls in thread pool with timeout
                def _index_to_es():
                    doc_state_manager._ensure_es_index()
                    doc_state_manager._bulk_index_chunks()
                
                # Get event loop (use get_event_loop() as fallback for compatibility)
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = asyncio.get_event_loop()
                
                await asyncio.wait_for(
                    loop.run_in_executor(None, _index_to_es),
                    timeout=60.0  # Longer timeout for bulk indexing
                )
            except asyncio.TimeoutError:
                logger.warning("Elasticsearch indexing timed out after 60 seconds")
                # Continue without ES, using local index
            except Exception as e:
                logger.warning(f"Elasticsearch indexing failed: {e}")
                # Continue without ES, using local index
            
            result = {
                "index_id": "document_index",
                "total_chunks": len(doc_state_manager.index),
                "first_chunk_id": 0,
                "last_chunk_id": len(doc_state_manager.index) - 1,
            }
            
            return ToolResponse(text=json.dumps(result, ensure_ascii=False)), 0.0, {}
        except Exception as e:
            logger.error(f"Error in buildIndex: {e}")
            return ToolResponse(text=json.dumps({"error": str(e)})), 0.0, {}
    
    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]


class ReadChunkTool(BaseTool):
    """Retrieve the full text of a chunk by its chunk_id."""
    
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}
    
    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid.uuid4())
        self._instance_dict[instance_id] = {}
        return instance_id, ToolResponse()
    
    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        """Read a specific chunk by ID."""
        try:
            doc_state_manager = kwargs.get('doc_state_manager')
            if not doc_state_manager:
                return ToolResponse(text=json.dumps({"error": "Document State Manager not available"})), 0.0, {}
            
            chunk_id = parameters.get("chunk_id", 0)
            try:
                chunk_id = int(chunk_id)
            except (ValueError, TypeError):
                return ToolResponse(text=json.dumps({"error": "chunk_id must be an integer."})), 0.0, {}
            
            if chunk_id < 0 or chunk_id >= len(doc_state_manager.index):
                return ToolResponse(text=json.dumps({
                    "error": f"Chunk_id: {chunk_id} is out of range. It must be between 0 and {len(doc_state_manager.index)-1}."
                })), 0.0, {}
            
            result = {
                "retrieved_chunk": [doc_state_manager.index[chunk_id]],
                "chunk_id": chunk_id
            }
            return ToolResponse(text=json.dumps(result, ensure_ascii=False)), 0.0, {}
        except Exception as e:
            logger.error(f"Error in readChunk: {e}")
            return ToolResponse(text=json.dumps({"error": str(e)})), 0.0, {}
    
    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]


class SearchEngineTool(BaseTool):
    """Search the document by keywords and return relevant chunks."""
    
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}
    
    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid.uuid4())
        self._instance_dict[instance_id] = {}
        return instance_id, ToolResponse()
    
    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        """Search document using keywords with BM25 ranking."""
        try:
            doc_state_manager = kwargs.get('doc_state_manager')
            if not doc_state_manager:
                return ToolResponse(text=json.dumps({"error": "Document State Manager not available"})), 0.0, {}
            
            raw_kw = parameters.get("keyword", "")
            if not doc_state_manager.index:
                return ToolResponse(text=json.dumps({"error": "Index not built. Please call buildIndex first."})), 0.0, {}
            if not doc_state_manager._doc_id:
                return ToolResponse(text=json.dumps({"error": "No active document. Please call buildIndex first."})), 0.0, {}
            
            # Normalize keywords
            if isinstance(raw_kw, list):
                keywords = [str(k).strip() for k in raw_kw if str(k).strip()]
            else:
                keywords = [k.strip() for k in str(raw_kw).split(",") if k.strip()]
            
            if not keywords:
                return ToolResponse(text=json.dumps({"error": "keyword cannot be empty."})), 0.0, {}
            
            doc_state_manager.keywords_searched.update(keywords)
            
            # Search parameters
            mode = parameters.get("mode", "or").lower()
            size = int(parameters.get("size", 50))
            fragment_size = int(parameters.get("fragment_size", 180))
            num_frags = int(parameters.get("num_fragments", 3))
            no_match_size = int(parameters.get("no_match_size", 120))
            min_should = parameters.get("minimum_should_match", "1")
            
            es = doc_state_manager._get_es()
            
            def _clause(kw: str):
                if " " in kw:
                    return {"match_phrase": {"content": {"query": kw, "slop": 2}}}
                return {"match": {"content": {"query": kw, "operator": "and"}}}
            
            # Build query
            if mode == "and":
                query = {
                    "bool": {
                        "must": [_clause(kw) for kw in keywords],
                        "filter": [{"term": {"doc_id": doc_state_manager._doc_id}}],
                    }
                }
            elif mode == "or":
                query = {
                    "bool": {
                        "should": [_clause(kw) for kw in keywords],
                        "minimum_should_match": min_should,
                        "filter": [{"term": {"doc_id": doc_state_manager._doc_id}}],
                    }
                }
            else:
                return ToolResponse(text=json.dumps({"error": f"Search mode '{mode}' not supported."})), 0.0, {}
            
            # Execute search (run in executor to avoid blocking event loop)
            try:
                # Create a proper function instead of lambda to avoid closure issues
                def _execute_search():
                    return es.search(
                        index=doc_state_manager._es_index_name,
                        query=query,
                        highlight={
                            "pre_tags": ["<em>"],
                            "post_tags": ["</em>"],
                            "fields": {
                                "content": {
                                    "type": "unified",
                                    "fragment_size": fragment_size,
                                    "number_of_fragments": num_frags,
                                    "no_match_size": no_match_size
                                }
                            }
                        },
                        _source=["chunk_id"],
                        size=size,
                        track_total_hits=False
                    )
                
                # Get event loop (use get_event_loop() as fallback for compatibility)
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = asyncio.get_event_loop()
                
                # Use executor to run blocking call, with 30 second timeout
                res = await asyncio.wait_for(
                    loop.run_in_executor(None, _execute_search),
                    timeout=30.0
                )
            except asyncio.TimeoutError:
                logger.error("Elasticsearch query timed out after 30 seconds")
                return ToolResponse(text=json.dumps({"error": "Elasticsearch query timed out"})), 0.0, {}
            except Exception as e:
                logger.error(f"Elasticsearch query failed: {e}")
                return ToolResponse(text=json.dumps({"error": f"Elasticsearch query failed: {e}"})), 0.0, {}
            
            hits = res.get("hits", {}).get("hits", [])
            if not hits:
                result = {
                    "retrieved_chunks": [],
                    "message": "No matching content found.",
                    "keywords": keywords
                }
                return ToolResponse(text=json.dumps(result, ensure_ascii=False)), 0.0, {}
            
            items = []
            for h in hits:
                src = h.get("_source", {})
                chunk_id = src.get("chunk_id")
                score = h.get("_score", 0.0)
                highlights = h.get("highlight", {}).get("content", [])
                items.append({
                    "chunk_id": chunk_id,
                    "relevance_score": round(float(score), 3),
                    "highlights": highlights
                })
            
            items.sort(key=lambda x: x["relevance_score"], reverse=True)
            total = len(items)
            if len(items) > 20:
                items = items[:20]
                result = {
                    "retrieved_chunks": items,
                    "message": f"Showing the most relevant 20/{total} chunks.",
                    "keywords": keywords
                }
            else:
                result = {"retrieved_chunks": items, "keywords": keywords}
            
            return ToolResponse(text=json.dumps(result, ensure_ascii=False)), 0.0, {}
        except Exception as e:
            logger.error(f"Error in searchEngine: {e}")
            return ToolResponse(text=json.dumps({"error": str(e)})), 0.0, {}
    
    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]


class NoteTool(BaseTool):
    """Record key information in a note."""
    
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}
    
    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid.uuid4())
        self._instance_dict[instance_id] = {}
        return instance_id, ToolResponse()
    
    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        """Create a new note with key, summary, and content."""
        try:
            agent_data = kwargs.get('agent_data')
            if not agent_data:
                return ToolResponse(text=json.dumps({"error": "Agent data not available"})), 0.0, {}
            
            key = parameters.get('key')
            summary = parameters.get('summary')
            content = parameters.get('content')
            
            if not key:
                return ToolResponse(text=json.dumps({"error": "Note key is required"})), 0.0, {}
            
            # Store note in agent_data
            agent_data.notes[str(key)] = {
                "summary": str(summary) if summary else "",
                "full_content": json.dumps(content, ensure_ascii=False) if isinstance(content, dict) else str(content)
            }
            
            result = {"status": "success", "key": key}
            return ToolResponse(text=json.dumps(result, ensure_ascii=False)), 0.0, {}
        except Exception as e:
            logger.error(f"Error in note: {e}")
            return ToolResponse(text=json.dumps({"error": str(e)})), 0.0, {}
    
    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]


class ReadNoteTool(BaseTool):
    """Read the full content of a specified note."""
    
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}
    
    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid.uuid4())
        self._instance_dict[instance_id] = {}
        return instance_id, ToolResponse()
    
    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        """Read a note by key."""
        try:
            agent_data = kwargs.get('agent_data')
            if not agent_data:
                return ToolResponse(text=json.dumps({"error": "Agent data not available"})), 0.0, {}
            
            key = parameters.get('key')
            if not key:
                return ToolResponse(text=json.dumps({"error": "Note key is required"})), 0.0, {}
            
            note = agent_data.notes.get(str(key))
            if note is None:
                return ToolResponse(text=json.dumps({"error": f"Note '{key}' not found!"})), 0.0, {}
            
            result = copy.deepcopy(note)
            return ToolResponse(text=json.dumps(result, ensure_ascii=False)), 0.0, {}
        except Exception as e:
            logger.error(f"Error in readNote: {e}")
            return ToolResponse(text=json.dumps({"error": str(e)})), 0.0, {}
    
    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]


class UpdateNoteTool(BaseTool):
    """Overwrite/Append/Delete an existing note."""
    
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}
    
    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid.uuid4())
        self._instance_dict[instance_id] = {}
        return instance_id, ToolResponse()
    
    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        """Update an existing note."""
        try:
            agent_data = kwargs.get('agent_data')
            if not agent_data:
                return ToolResponse(text=json.dumps({"error": "Agent data not available"})), 0.0, {}
            
            key = parameters.get('key')
            mode = parameters.get('mode', '').lower()
            new_content = parameters.get('new_content')
            new_summary = parameters.get('new_summary')
            
            if not key:
                return ToolResponse(text=json.dumps({"error": "Note key is required"})), 0.0, {}
            
            k = str(key)
            if k not in agent_data.notes:
                return ToolResponse(text=json.dumps({"error": f"Note '{key}' not found!"})), 0.0, {}
            
            if mode == "delete":
                del agent_data.notes[k]
                result = {"status": "success", "key": key, "message": f"Note '{key}' deleted."}
                return ToolResponse(text=json.dumps(result, ensure_ascii=False)), 0.0, {}
            
            if new_content is None:
                return ToolResponse(text=json.dumps({"error": "New note content is required."})), 0.0, {}
            
            existing_content = agent_data.notes[k]["full_content"]
            new_content_str = json.dumps(new_content, ensure_ascii=False) if isinstance(new_content, dict) else str(new_content)
            
            if mode == "append":
                full_content = existing_content + "\n" + new_content_str
                msg = f"Note '{key}' appended."
            elif mode == "overwrite":
                full_content = new_content_str
                msg = f"Note '{key}' overwritten."
            else:
                return ToolResponse(text=json.dumps({"error": f"Invalid mode '{mode}'. Use 'append', 'overwrite', or 'delete'."})), 0.0, {}
            
            agent_data.notes[k]["full_content"] = full_content
            if new_summary is not None:
                agent_data.notes[k]["summary"] = str(new_summary)
            
            result = {"status": "success", "key": key, "message": msg}
            return ToolResponse(text=json.dumps(result, ensure_ascii=False)), 0.0, {}
        except Exception as e:
            logger.error(f"Error in updateNote: {e}")
            return ToolResponse(text=json.dumps({"error": str(e)})), 0.0, {}
    
    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]


class MergeNotesTool(BaseTool):
    """Merge multiple notes into one."""
    
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}
    
    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid.uuid4())
        self._instance_dict[instance_id] = {}
        return instance_id, ToolResponse()
    
    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        """Merge multiple notes into a single note."""
        try:
            agent_data = kwargs.get('agent_data')
            if not agent_data:
                return ToolResponse(text=json.dumps({"error": "Agent data not available"})), 0.0, {}
            
            keys = parameters.get('keys', [])
            new_key = parameters.get('new_key')
            new_summary = parameters.get('new_summary')
            
            if not keys:
                return ToolResponse(text=json.dumps({"error": "At least one key is required"})), 0.0, {}
            
            notes_to_merge = []
            for key in keys:
                k = str(key)
                if k in agent_data.notes:
                    notes_to_merge.append((k, agent_data.notes[k]["summary"], agent_data.notes[k]["full_content"]))
                    del agent_data.notes[k]
            
            if not notes_to_merge:
                return ToolResponse(text=json.dumps({"error": "No notes found to merge."})), 0.0, {}
            
            merged_key = new_key or "_".join([note[0] for note in notes_to_merge])
            merged_summary = new_summary or "  ".join([note[1] for note in notes_to_merge])
            merged_content = "\n".join([note[2] for note in notes_to_merge])
            
            agent_data.notes[merged_key] = {
                "summary": str(merged_summary),
                "full_content": str(merged_content)
            }
            
            result = {
                "status": "success",
                "new_key": merged_key,
                "merged_from": [note[0] for note in notes_to_merge]
            }
            return ToolResponse(text=json.dumps(result, ensure_ascii=False)), 0.0, {}
        except Exception as e:
            logger.error(f"Error in mergeNotes: {e}")
            return ToolResponse(text=json.dumps({"error": str(e)})), 0.0, {}
    
    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]


class CheckBudgetTool(BaseTool):
    """Check the remaining context budget."""
    
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}
    
    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid.uuid4())
        self._instance_dict[instance_id] = {}
        return instance_id, ToolResponse()
    
    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        """Calculate remaining context budget."""
        try:
            agent_data = kwargs.get('agent_data')
            tokenizer = kwargs.get('tokenizer')
            tool_schemas = kwargs.get('tool_schemas', [])
            max_context_exp = kwargs.get('max_context_exp', 8192)
            max_output_tokens = kwargs.get('max_output_tokens', 4096)
            max_turns = kwargs.get('max_turns', 50)
            
            if not agent_data or not tokenizer:
                return ToolResponse(text=json.dumps({"error": "Required context not available"})), 0.0, {}
            
            # Build message view and tokenize
            messages = agent_data._render_message_view()
            tokenized_messages = tokenizer.apply_chat_template(
                messages,
                tools=tool_schemas,
                add_generation_prompt=False,
                tokenize=True
            )
            
            conv_rounds = (agent_data.user_turns + agent_data.assistant_turns) // 2
            message_len = len(tokenized_messages)
            
            result = {
                "conv_rounds": conv_rounds,
                "available_tokens": max(max_context_exp - message_len - max_output_tokens, 0),
                "available_rounds": max(max_turns - conv_rounds, 0),
            }
            
            return ToolResponse(text=json.dumps(result, ensure_ascii=False)), 0.0, {}
        except Exception as e:
            logger.error(f"Error in checkBudget: {e}")
            return ToolResponse(text=json.dumps({"error": str(e)})), 0.0, {}
    
    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]


class GetContextStatsTool(BaseTool):
    """Get full statistics about the current working context and stored notes."""
    
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}
    
    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid.uuid4())
        self._instance_dict[instance_id] = {}
        return instance_id, ToolResponse()
    
    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        """Get comprehensive context statistics."""
        try:
            agent_data = kwargs.get('agent_data')
            doc_state_manager = kwargs.get('doc_state_manager')
            tokenizer = kwargs.get('tokenizer')
            tool_schemas = kwargs.get('tool_schemas', [])
            max_context_exp = kwargs.get('max_context_exp', 8192)
            max_output_tokens = kwargs.get('max_output_tokens', 4096)
            max_turns = kwargs.get('max_turns', 50)
            
            if not agent_data or not doc_state_manager or not tokenizer:
                return ToolResponse(text=json.dumps({"error": "Required context not available"})), 0.0, {}
            
            # Calculate budget info
            messages = agent_data._render_message_view()
            tokenized_messages = tokenizer.apply_chat_template(
                messages,
                tools=tool_schemas,
                add_generation_prompt=False,
                tokenize=True
            )
            
            conv_rounds = (agent_data.user_turns + agent_data.assistant_turns) // 2
            message_len = len(tokenized_messages)
            
            result = {
                "total_notes": len(agent_data.notes),
                "notes_keys": list(agent_data.notes.keys()),
                "index_chunks": len(doc_state_manager.index),
                "document_size": len(doc_state_manager.encoded_doc["input_ids"]),
                "searched_keywords": list(doc_state_manager.keywords_searched),
                "conv_rounds": conv_rounds,
                "available_tokens": max(max_context_exp - message_len - max_output_tokens, 0),
                "available_rounds": max(max_turns - conv_rounds, 0),
            }
            
            return ToolResponse(text=json.dumps(result, ensure_ascii=False)), 0.0, {}
        except Exception as e:
            logger.error(f"Error in getContextStats: {e}")
            return ToolResponse(text=json.dumps({"error": str(e)})), 0.0, {}
    
    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]


class DeleteContextTool(BaseTool):
    """Delete an assistant or tool message by its message ID to save tokens."""
    
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}
    
    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid.uuid4())
        self._instance_dict[instance_id] = {}
        return instance_id, ToolResponse()
    
    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        """Delete a message by its message ID."""
        try:
            agent_data = kwargs.get('agent_data')
            if not agent_data:
                return ToolResponse(text=json.dumps({"error": "Agent data not available. deleteContext requires agent_data in kwargs."})), 0.0, {}
            
            msg_id = parameters.get("msg_id")
            
            if msg_id is not None:
                try:
                    msg_ids_to_delete = int(msg_id)
                except ValueError:
                    return ToolResponse(text=json.dumps({"error": "Invalid message ID format"})), 0.0, {}
            else:
                return ToolResponse(text=json.dumps({"error": "'msg_id' is required"})), 0.0, {}
            
            def _resolve_msg_entry(full_history: list[dict], msg_id: int):
                for i, m in enumerate(full_history):
                    if m.get("msg_id") == msg_id:
                        return i, m
                return None, None
            
            idx, entry = _resolve_msg_entry(agent_data.full_history, msg_ids_to_delete)

            if entry is not None:
                # Add the actual msg_id (not the index!) to deleted_msg_ids
                agent_data.deleted_msg_ids.add(msg_ids_to_delete)
                role = entry.get("role", "unknown")
            else:
                logger.warning(f"Message ID {msg_ids_to_delete} not found in full_history, skipping")
                return ToolResponse(text=json.dumps({"error": f"Message ID {msg_ids_to_delete} not found"})), 0.0, {}
            
            result = {
                "status": "success",
                "deleted_msg_id": msg_ids_to_delete,
                "deleted_role": role,
            }
            # Return deleted_msg_ids as a list for compatibility with tool_agent_loop.py
            return ToolResponse(text=json.dumps(result, ensure_ascii=False)), 0.0, {"deleted_msg_ids": [msg_ids_to_delete]}
        except Exception as e:
            logger.error(f"Error in deleteContext: {e}")
            return ToolResponse(text=json.dumps({"error": str(e)})), 0.0, {}
    
    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]


class FinishTool(BaseTool):
    """Submit the final answer."""
    
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}
    
    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid.uuid4())
        self._instance_dict[instance_id] = {}
        return instance_id, ToolResponse()
    
    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        """Submit the final answer and mark completion."""
        try:
            answer = parameters.get("answer", "No final answer provided.")
            result = {"final_answer": answer}
            return ToolResponse(text=json.dumps(result, ensure_ascii=False)), 0.0, {}
        except Exception as e:
            logger.error(f"Error in finish: {e}")
            return ToolResponse(text=json.dumps({"error": str(e)})), 0.0, {}
    
    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]

