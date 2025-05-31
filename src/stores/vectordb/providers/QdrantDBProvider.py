from qdrant_client import models, QdrantClient
from ..VectorDBInterface import VectorDBInterface
from ..VectorDBEnums import DistanceMethodEnums
import logging
from typing import List
from models.db_schemes import RetrievedDocument

class QdrantDBProvider(VectorDBInterface):

    def __init__(self, db_path: str, distance_method: str):

        self.client = None
        self.db_path = db_path
        self.distance_method = None

        if distance_method == DistanceMethodEnums.COSINE.value:
            self.distance_method = models.Distance.COSINE
        elif distance_method == DistanceMethodEnums.DOT.value:
            self.distance_method = models.Distance.DOT

        self.logger = logging.getLogger(__name__)

    def connect(self):
        self.client = QdrantClient(path=self.db_path)

    def disconnect(self):
        self.client = None

    def is_collection_existed(self, collection_name: str) -> bool:
        return self.client.collection_exists(collection_name=collection_name)
    
    def list_all_collections(self) -> List:
        return self.client.get_collections()
    
    def get_collection_info(self, collection_name: str) -> dict:
        try:
            if not self.is_collection_existed(collection_name):
                return None
            
            # Simple approach: try to scroll through the collection to count points
            total_count = 0
            next_page_token = None
            
            try:
                while True:
                    # Use scroll to get points in batches
                    if next_page_token:
                        scroll_result = self.client.scroll(
                            collection_name=collection_name,
                            limit=100,
                            offset=next_page_token,
                            with_payload=False,
                            with_vectors=False
                        )
                    else:
                        scroll_result = self.client.scroll(
                            collection_name=collection_name,
                            limit=100,
                            with_payload=False,
                            with_vectors=False
                        )
                    
                    # Extract points and next page token
                    points = []
                    next_page_token = None
                    
                    if hasattr(scroll_result, 'points'):
                        points = scroll_result.points or []
                        next_page_token = getattr(scroll_result, 'next_page_offset', None)
                    elif isinstance(scroll_result, tuple) and len(scroll_result) >= 2:
                        points = scroll_result[0] or []
                        next_page_token = scroll_result[1]
                    elif isinstance(scroll_result, tuple) and len(scroll_result) >= 1:
                        points = scroll_result[0] or []
                    
                    # Add count from this batch
                    batch_count = len(points)
                    total_count += batch_count
                    
                    self.logger.info(f"Scrolled batch: {batch_count} points, total so far: {total_count}")
                    
                    # If no more points or no next page, break
                    if batch_count == 0 or not next_page_token:
                        break
                
                self.logger.info(f"Collection {collection_name} has {total_count} total points")
                
                return {
                    "status": "green",
                    "points_count": total_count,
                    "vectors_count": total_count
                }
                
            except Exception as scroll_error:
                self.logger.error(f"Error during scroll counting for {collection_name}: {scroll_error}")
                
                # Ultra-simple fallback: just try to get one point to see if collection has data
                try:
                    simple_scroll = self.client.scroll(collection_name=collection_name, limit=1)
                    has_data = False
                    
                    if hasattr(simple_scroll, 'points') and simple_scroll.points:
                        has_data = len(simple_scroll.points) > 0
                    elif isinstance(simple_scroll, tuple) and len(simple_scroll) >= 1:
                        points = simple_scroll[0]
                        has_data = points and len(points) > 0
                    
                    return {
                        "status": "green",
                        "points_count": "1+" if has_data else 0,
                        "vectors_count": "1+" if has_data else 0
                    }
                except Exception as e:
                    self.logger.error(f"Even simple scroll failed: {e}")
                    return {
                        "status": "unknown",
                        "points_count": 0,
                        "vectors_count": 0
                    }
                        
        except Exception as e:
            self.logger.error(f"Error getting collection info for {collection_name}: {e}")
            return None
    
    def delete_collection(self, collection_name: str):
        if self.is_collection_existed(collection_name):
            return self.client.delete_collection(collection_name=collection_name)
        
    def create_collection(self, collection_name: str, 
                                embedding_size: int,
                                do_reset: bool = False):
        if do_reset:
            _ = self.delete_collection(collection_name=collection_name)
        
        if not self.is_collection_existed(collection_name):
            _ = self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=embedding_size,
                    distance=self.distance_method
                )
            )

            return True
        
        return False
    
    def insert_one(self, collection_name: str, text: str, vector: list,
                         metadata: dict = None, 
                         record_id: str = None):
        
        if not self.is_collection_existed(collection_name):
            self.logger.error(f"Can not insert new record to non-existed collection: {collection_name}")
            return False
        
        try:
            _ = self.client.upload_records(
                collection_name=collection_name,
                records=[
                    models.Record(
                        id=[record_id],
                        vector=vector,
                        payload={
                            "text": text, "metadata": metadata
                        }
                    )
                ]
            )
        except Exception as e:
            self.logger.error(f"Error while inserting batch: {e}")
            return False

        return True
    
    def insert_many(self, collection_name: str, texts: list, 
                          vectors: list, metadata: list = None, 
                          record_ids: list = None, batch_size: int = 50):
        
        if metadata is None:
            metadata = [None] * len(texts)

        if record_ids is None:
            record_ids = list(range(0, len(texts)))

        self.logger.info(f"Starting insertion of {len(texts)} records into {collection_name}")
        total_inserted = 0

        for i in range(0, len(texts), batch_size):
            batch_end = i + batch_size

            batch_texts = texts[i:batch_end]
            batch_vectors = vectors[i:batch_end]
            batch_metadata = metadata[i:batch_end]
            batch_record_ids = record_ids[i:batch_end]

            batch_records = [
                models.Record(
                    id=batch_record_ids[x],
                    vector=batch_vectors[x],
                    payload={
                        "text": batch_texts[x], "metadata": batch_metadata[x]
                    }
                )

                for x in range(len(batch_texts))
            ]

            try:
                self.logger.info(f"Inserting batch {i//batch_size + 1}: {len(batch_records)} records")
                result = self.client.upload_records(
                    collection_name=collection_name,
                    records=batch_records,
                )
                total_inserted += len(batch_records)
                self.logger.info(f"Successfully inserted batch {i//batch_size + 1}. Total so far: {total_inserted}")
                
                # Log the result from Qdrant
                if result:
                    self.logger.info(f"Qdrant upload result: {result}")
                    
            except Exception as e:
                self.logger.error(f"Error while inserting batch {i//batch_size + 1}: {e}")
                return False

        self.logger.info(f"Completed insertion. Total records inserted: {total_inserted}")
        return True
        
    def search_by_vector(self, collection_name: str, vector: list, limit: int = 5):

        results = self.client.search(
            collection_name=collection_name,
            query_vector=vector,
            limit=limit
        )

        if not results or len(results) == 0:
            return None
        
        return [
            RetrievedDocument(**{
                "score": result.score,
                "text": result.payload["text"],
            })
            for result in results
        ]

