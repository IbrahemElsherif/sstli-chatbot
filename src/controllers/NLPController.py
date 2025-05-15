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
    
    def get_vector_db_collection_info(self, project: Project) -> Dict[str, Any]:
        collection_name = self.create_collection_name(project_id=project.project_id)
        collection_info = self.vectordb_client.get_collection_info(collection_name=collection_name)

        return json.loads(
            json.dumps(collection_info, default=lambda x: x.__dict__)
        )
    
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
        embedding_size = self.app_settings.EMBEDDING_MODEL_SIZE
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

        # Check if embedding client is properly initialized
        use_fallback = False
        if not hasattr(self.embedding_client, 'client') or self.embedding_client.client is None:
            self.logger.error("Embedding client is not properly initialized. Falling back to simple embeddings.")
            use_fallback = True

        # Create collection
        try:
            self.vectordb_client.create_collection(
                collection_name=collection_name,
                embedding_size=self.app_settings.EMBEDDING_MODEL_SIZE,
                do_reset=do_reset
            )
        except Exception as e:
            self.logger.error(f"Failed to create/access vector collection: {str(e)}")
            return False

        self.logger.info(f"Starting indexing of {len(texts)} chunks into vector DB")
        start_time = time.time()
        
        # If using fallback, generate embeddings without API calls
        if use_fallback:
            self.logger.info("Using fallback embedding method")
            all_embeddings = []
            for text in texts:
                vector = self._fallback_embed(text)
                all_embeddings.append(vector)
            
            # Insert into vector DB
            self.logger.info(f"Inserting {len(all_embeddings)} vectors into vector DB")
            inserted = self.vectordb_client.insert_many(
                collection_name=collection_name,
                texts=texts,
                vectors=all_embeddings,
                metadata=metadata if metadata else None,
                record_ids=chunks_ids if chunks_ids else None,
                batch_size=len(all_embeddings)
            )
            
            elapsed_time = time.time() - start_time
            self.logger.info(f"Vector DB indexing complete. Processed {len(all_embeddings)} vectors in {elapsed_time:.2f} seconds")
            
            return inserted

        # Configuration for processing - optimized for OpenAI
        embedding_batch_size = 100  # OpenAI can handle larger batches
        vector_db_batch_size = 200  # Larger batches for faster vector DB insertion
        
        # Determine processing approach based on dataset size
        small_batch = len(texts) <= 200  # For very small batches, optimize differently
        
        try:
            # For small batches, process everything in one go
            if small_batch:
                self.logger.info(f"Small batch detected, processing all {len(texts)} chunks at once")
                
                # Get all embeddings at once
                start_embed = time.time()
                if hasattr(self.embedding_client, 'embed_batch'):
                    vectors = self.embedding_client.embed_batch(texts)
                else:
                    # Fallback to individual embedding
                    vectors = []
                    for text in texts:
                        vectors.append(self.embedding_client.embed_text(text=text))
                embed_time = time.time() - start_embed
                self.logger.info(f"Embedding completed in {embed_time:.2f} seconds")
                
                # Filter valid embeddings
                valid_embeddings = []
                valid_texts = []
                valid_metadata = []
                valid_ids = []
                
                for i, vector in enumerate(vectors):
                    if vector is not None:
                        valid_embeddings.append(vector)
                        valid_texts.append(texts[i])
                        
                        if metadata:
                            valid_metadata.append(metadata[i])
                        
                        if chunks_ids:
                            valid_ids.append(chunks_ids[i])
                
                success_rate = len(valid_embeddings)/len(texts)*100 if texts else 0
                self.logger.info(f"Successfully embedded {len(valid_embeddings)} out of {len(texts)} texts ({success_rate:.1f}%)")
                
                # Insert all vectors at once
                if valid_embeddings:
                    start_insert = time.time()
                    self.vectordb_client.insert_many(
                        collection_name=collection_name,
                        texts=valid_texts,
                        vectors=valid_embeddings,
                        metadata=valid_metadata if valid_metadata else None,
                        record_ids=valid_ids if valid_ids else None,
                        batch_size=len(valid_embeddings)
                    )
                    insert_time = time.time() - start_insert
                    self.logger.info(f"Vector DB insertion completed in {insert_time:.2f} seconds")
            
            else:
                # Configuration for larger batches
                embedding_batch_size = 100  # OpenAI can handle larger batches
                vector_db_batch_size = 200  # Larger batches for faster vector DB insertion
                
                # Phase 1: Generate all embeddings with better batching
                all_embeddings = []
                processed_count = 0
                
                # Use batch embedding if available
                if hasattr(self.embedding_client, 'embed_batch'):
                    self.logger.info("Using batch embedding for faster processing")
                    
                    for i in range(0, len(texts), embedding_batch_size):
                        batch_end = min(i + embedding_batch_size, len(texts))
                        batch_texts = texts[i:batch_end]
                        batch_size = len(batch_texts)
                        
                        try:
                            # Track progress
                            self.logger.info(f"Generating embeddings batch {i//embedding_batch_size + 1} of {(len(texts) + embedding_batch_size - 1)//embedding_batch_size} ({processed_count}/{len(texts)} total)")
                            
                            # Get embeddings for batch
                            vectors = self.embedding_client.embed_batch(batch_texts)
                            if vectors:
                                all_embeddings.extend(vectors)
                                processed_count += batch_size
                            
                            # Add minimal delay to avoid rate limits
                            if batch_end < len(texts):
                                time.sleep(0.1)  # OpenAI can handle higher request rates
                                
                        except Exception as e:
                            self.logger.error(f"Error in batch embedding: {str(e)}")
                            
                            # Fallback to individual processing if batch fails
                            self.logger.info("Falling back to individual embedding")
                            for j, text in enumerate(batch_texts):
                                try:
                                    vector = self.embedding_client.embed_text(text=text)
                                    all_embeddings.append(vector)
                                    processed_count += 1
                                    
                                    # Small delay for rate limiting
                                    time.sleep(0.1)
                                except Exception as inner_e:
                                    self.logger.error(f"Error embedding text: {str(inner_e)}")
                                    all_embeddings.append(None)
                                    processed_count += 1
                            
                            # Add longer delay after an error
                            time.sleep(1)  # Reduced delay for OpenAI
                else:
                    # Fallback to individual embedding
                    self.logger.info("Using individual embedding process")
                    
                    for i in range(0, len(texts), embedding_batch_size):
                        batch_end = min(i + embedding_batch_size, len(texts))
                        batch_texts = texts[i:batch_end]
                        
                        # Track progress
                        self.logger.info(f"Processing batch {i//embedding_batch_size + 1} of {(len(texts) + embedding_batch_size - 1)//embedding_batch_size} ({processed_count}/{len(texts)} total)")
                        
                        for text in batch_texts:
                            try:
                                vector = self.embedding_client.embed_text(text=text)
                                all_embeddings.append(vector)
                                processed_count += 1
                            except Exception as e:
                                self.logger.error(f"Error embedding individual text: {str(e)}")
                                all_embeddings.append(None)
                                processed_count += 1
                                time.sleep(1)  # Reduced delay for OpenAI after error
                        
                        # Add controlled delay between batches
                        if batch_end < len(texts):
                            time.sleep(0.5)  # Reduced delay for OpenAI
                            
                # Phase 2: Filter valid embeddings
                self.logger.info(f"Embedding phase complete. Processing successful embeddings.")
                valid_embeddings = []
                valid_texts = []
                valid_metadata = []
                valid_ids = []
                
                for i, vector in enumerate(all_embeddings):
                    if vector is not None:
                        valid_embeddings.append(vector)
                        valid_texts.append(texts[i])
                        
                        if metadata:
                            valid_metadata.append(metadata[i])
                        
                        if chunks_ids:
                            valid_ids.append(chunks_ids[i])
                
                self.logger.info(f"Successfully embedded {len(valid_embeddings)} out of {len(texts)} texts ({len(valid_embeddings)/len(texts)*100:.1f}%)")
                
                # Phase 3: Insert all valid vectors to the vector DB in larger batches
                if valid_embeddings:
                    inserted_count = 0
                    for i in range(0, len(valid_embeddings), vector_db_batch_size):
                        batch_end = min(i + vector_db_batch_size, len(valid_embeddings))
                        batch_size = batch_end - i
                        
                        try:
                            self.logger.info(f"Inserting vector batch {i//vector_db_batch_size + 1} of {(len(valid_embeddings) + vector_db_batch_size - 1)//vector_db_batch_size} ({inserted_count}/{len(valid_embeddings)} total)")
                            
                            self.vectordb_client.insert_many(
                                collection_name=collection_name,
                                texts=valid_texts[i:batch_end],
                                vectors=valid_embeddings[i:batch_end],
                                metadata=valid_metadata[i:batch_end] if valid_metadata else None,
                                record_ids=valid_ids[i:batch_end] if valid_ids else None,
                                batch_size=vector_db_batch_size
                            )
                            
                            inserted_count += batch_size
                        except Exception as e:
                            self.logger.error(f"Error inserting vectors batch: {str(e)}")
                            time.sleep(0.5)  # Reduced wait time
            
            elapsed_time = time.time() - start_time
            self.logger.info(f"Vector DB indexing complete. Processed {len(valid_embeddings if small_batch else valid_embeddings)} vectors in {elapsed_time:.2f} seconds")
            
            return True
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            self.logger.error(f"Error in index_into_vector_db: {str(e)}, elapsed time: {elapsed_time:.2f} seconds")
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

