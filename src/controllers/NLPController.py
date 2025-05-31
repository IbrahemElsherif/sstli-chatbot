from .BaseController import BaseController
from models.db_schemes import Project, DataChunk
from stores.llm.LLMEnums import DocumentTypeEnum
from typing import List, Dict, Any, Optional
import json
import time
import logging
import asyncio
import concurrent.futures
import hashlib

class NLPController(BaseController):

    def __init__(self, vectordb_client, generation_client, 
                 embedding_client, template_parser):
        super().__init__()

        self.vectordb_client = vectordb_client
        self.generation_client = generation_client
        self.embedding_client = embedding_client
        self.template_parser = template_parser
        self.logger = logging.getLogger(__name__)

    def create_collection_name(self, project_id: str) -> str:
        return f"collection_{project_id}".strip()
    
    def reset_vector_db_collection(self, project: Project) -> bool:
        collection_name = self.create_collection_name(project_id=project.project_id)
        return self.vectordb_client.delete_collection(collection_name=collection_name)
    
    def get_collection_info(self, project_id: str) -> dict:
        try:
            collection_name = self.create_collection_name(project_id=project_id)
            
            # Get collection info from vector database
            collection_info = self.vectordb_client.get_collection_info(collection_name)
            
            if collection_info is None:
                return {
                    "exists": False,
                    "message": f"Collection {collection_name} does not exist"
                }
            
            # Ensure the response is JSON serializable
            result = {
                "exists": True,
                "collection_name": collection_name,
                "status": collection_info.get("status", "unknown"),
                "points_count": collection_info.get("points_count", 0),
                "vectors_count": collection_info.get("vectors_count", 0)
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error getting collection info for project {project_id}: {e}")
            return {
                "exists": False,
                "error": str(e),
                "message": f"Error retrieving collection info: {str(e)}"
            }
    
    def _fallback_embed(self, text: str) -> List[float]:
        """Generate a simple deterministic vector as fallback when API fails.
        This is not a proper embedding but allows the application to continue functioning."""
        
        # Handle empty text
        if not text or len(text.strip()) == 0:
            text = "empty_text"
        
        # Create a simple hash-based embedding (not for production use)
        hash_obj = hashlib.md5(text.encode('utf-8', errors='replace'))
        digest = hash_obj.digest()
        
        # Convert to a normalized vector of the right size
        # Default to 1536 (OpenAI text-embedding-3-small size) if not available
        embedding_size = 1536
        try:
            # Try to get size from embedding client if available
            if hasattr(self.embedding_client, 'embedding_size'):
                embedding_size = self.embedding_client.embedding_size
            elif hasattr(self.embedding_client, 'model_size'):
                embedding_size = self.embedding_client.model_size
        except:
            pass  # Use default
            
        vector = []
        
        # Expand the 16-byte digest to fill the embedding size
        for i in range(embedding_size):
            byte_val = digest[i % 16]
            vector.append((byte_val / 255.0) * 2 - 1)  # Scale to [-1, 1]
        
        return vector

    def index_into_vector_db(self, project: Project, chunks: List[DataChunk],
                         do_reset: bool = False, chunks_ids: List[int] = None) -> bool:
        
        # step1: get collection name
        collection_name = self.create_collection_name(project_id=project.project_id)

        # step2: manage items
        texts = [c.chunk_text for c in chunks]
        metadata = [c.chunk_metadata for c in chunks]

        if not texts:
            return True

        # Test if API is working, otherwise use fallback
        use_fallback = False
        try:
            self.logger.info("Testing OpenAI API connection...")
            test_embedding = self.embedding_client.embed_text(text="test")
            if test_embedding is None or len(test_embedding) == 0:
                use_fallback = True
        except Exception as e:
            self.logger.warning(f"OpenAI API failed ({str(e)}). Using fallback embeddings.")
            use_fallback = True

        # Create collection with proper embedding size
        embedding_size = 1536  # Default OpenAI size
        try:
            self.vectordb_client.create_collection(
                collection_name=collection_name,
                embedding_size=embedding_size,
                do_reset=do_reset
            )
        except Exception as e:
            self.logger.error(f"Failed to create vector collection: {str(e)}")
            return False

        self.logger.info(f"Starting indexing of {len(texts)} chunks using {'fallback' if use_fallback else 'API'} embeddings")
        start_time = time.time()
        
        if use_fallback:
            # Generate fallback embeddings
            all_embeddings = []
            for i, text in enumerate(texts):
                try:
                    vector = self._fallback_embed(text)
                    all_embeddings.append(vector)
                    if i % 10 == 0:  # Log progress every 10 items
                        self.logger.info(f"Generated fallback embeddings: {i+1}/{len(texts)}")
                except Exception as e:
                    self.logger.error(f"Error generating fallback embedding for text {i+1}: {str(e)}")
                    return False
            
            self.logger.info(f"Generated {len(all_embeddings)} fallback embeddings successfully")
            
        else:
            # Use API embeddings
            all_embeddings = []
            for i, text in enumerate(texts):
                try:
                    vector = self.embedding_client.embed_text(text=text)
                    if vector:
                        all_embeddings.append(vector)
                    else:
                        # Fallback for this specific text
                        vector = self._fallback_embed(text)
                        all_embeddings.append(vector)
                        
                    if i % 10 == 0:  # Log progress every 10 items
                        self.logger.info(f"Generated API embeddings: {i+1}/{len(texts)}")
                        
                except Exception as e:
                    self.logger.warning(f"API embedding failed for text {i+1}, using fallback: {str(e)}")
                    vector = self._fallback_embed(text)
                    all_embeddings.append(vector)

        # Insert embeddings into vector database
        if all_embeddings and len(all_embeddings) == len(texts):
            self.logger.info(f"Inserting {len(all_embeddings)} vectors into database...")
            
            success = self.vectordb_client.insert_many(
                collection_name=collection_name,
                texts=texts,
                vectors=all_embeddings,
                metadata=metadata if metadata else None,
                record_ids=chunks_ids if chunks_ids else None,
                batch_size=50
            )
            
            elapsed_time = time.time() - start_time
            
            if success:
                self.logger.info(f"✅ Indexing completed successfully! Processed {len(all_embeddings)} vectors in {elapsed_time:.2f} seconds")
                return True
            else:
                self.logger.error("❌ Failed to insert vectors into database")
                return False
        else:
            self.logger.error(f"❌ Embedding generation failed. Expected {len(texts)} embeddings, got {len(all_embeddings) if all_embeddings else 0}")
            return False

    def search_vector_db_collection(self, project: Project, text: str, limit: int = 10):

        # step1: get collection name
        collection_name = self.create_collection_name(project_id=project.project_id)

        # step2: get text embedding vector
        try:
            vector = self.embedding_client.embed_text(text=text, 
                                                    document_type=DocumentTypeEnum.QUERY.value)
            
            if not vector or len(vector) == 0:
                self.logger.warning("API embedding failed, using fallback embedding")
                vector = self._fallback_embed(text)
        except Exception as e:
            self.logger.warning(f"Error generating embedding: {str(e)}, using fallback")
            vector = self._fallback_embed(text)

        if not vector or len(vector) == 0:
            return False

        # step3: do semantic search
        results = self.vectordb_client.search_by_vector(
            collection_name=collection_name,
            vector=vector,
            limit=limit
        )

        if not results:
            return False

        return results
    
    def answer_rag_question(self, project: Project, query: str, limit: int = 10):
        
        answer, full_prompt, chat_history = None, None, None

        # step1: retrieve related documents
        retrieved_documents = self.search_vector_db_collection(
            project=project,
            text=query,
            limit=limit,
        )

        if not retrieved_documents or len(retrieved_documents) == 0:
            return answer, full_prompt, chat_history
        
        # step2: Construct LLM prompt
        system_prompt = self.template_parser.get("rag", "system_prompt")

        documents_prompts = "\n".join([
            self.template_parser.get("rag", "document_prompt", {
                    "doc_num": idx + 1,
                    "chunk_text": doc.text,
            })
            for idx, doc in enumerate(retrieved_documents)
        ])

        footer_prompt = self.template_parser.get("rag", "footer_prompt", {
            "query": query
        })

        # step3: Construct Generation Client Prompts
        chat_history = [
            self.generation_client.construct_prompt(
                prompt=system_prompt,
                role=self.generation_client.enums.SYSTEM.value,
            )
        ]

        full_prompt = "\n\n".join([ documents_prompts,  footer_prompt])

        # step4: Retrieve the Answer
        answer = self.generation_client.generate_text(
            prompt=full_prompt,
            chat_history=chat_history
        )

        return answer, full_prompt, chat_history

