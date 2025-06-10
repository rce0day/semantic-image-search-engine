import uuid
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from qdrant_client.http.exceptions import UnexpectedResponse

from semantic_img.indexing.index_base import VectorIndex
from semantic_img import config


class QdrantIndex(VectorIndex):

    def __init__(self, 
                 host: str = config.QDRANT_HOST,
                 port: int = config.QDRANT_PORT,
                 grpc_port: int = config.QDRANT_GRPC_PORT,
                 api_key: Optional[str] = config.QDRANT_API_KEY,
                 prefer_grpc: bool = config.QDRANT_PREFER_GRPC,
                 timeout: float = config.QDRANT_TIMEOUT):

        self.client = QdrantClient(
            host=host,
            port=port,
            grpc_port=grpc_port,
            api_key=api_key,
            prefer_grpc=prefer_grpc,
            timeout=timeout
        )
        
    def create_collection(self, collection_name: str, vector_size: int, **kwargs) -> bool:
        try:
            distance = kwargs.get('distance', rest.Distance.COSINE)
            on_disk = kwargs.get('on_disk', False)
            
            vectors_config = rest.VectorParams(
                size=vector_size,
                distance=distance,
                hnsw_config=rest.HnswConfigDiff(
                    m=32,
                    ef_construct=256,
                    full_scan_threshold=16384
                )
            )
            
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=vectors_config,
                on_disk_payload=on_disk,
                optimizers_config=rest.OptimizersConfigDiff(
                    default_segment_number=4,
                    max_segment_size=200000,
                    memmap_threshold=50000,
                    indexing_threshold=50000,
                    flush_interval_sec=30,
                    max_optimization_threads=2
                )
            )
            
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name="image_path",
                field_schema=rest.PayloadSchemaType.KEYWORD
            )
            
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name="parent_id",
                field_schema=rest.PayloadSchemaType.KEYWORD
            )
            
            return True
        except Exception as e:
            return False
    
    def drop_collection(self, collection_name: str) -> bool:
        try:
            self.client.delete_collection(collection_name=collection_name)
            return True
        except Exception as e:
            return False
    
    def list_collections(self) -> List[str]:
        collections = self.client.get_collections().collections
        return [collection.name for collection in collections]
    
    def collection_info(self, collection_name: str) -> Dict[str, Any]:
        try:
            info = self.client.get_collection(collection_name=collection_name)
            return {
                "name": info.config.params.collection_name,
                "vector_size": info.config.params.vectors.size,
                "distance": info.config.params.vectors.distance,
                "points_count": info.vectors_count,
                "on_disk_payload": info.config.params.on_disk_payload,
            }
        except Exception as e:
            return {}
    
    def add_vectors(self, 
                   collection_name: str, 
                   vectors: np.ndarray, 
                   ids: Optional[List[str]] = None,
                   metadata: Optional[List[Dict[str, Any]]] = None) -> List[str]:

        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(len(vectors))]
        
        if metadata is None:
            metadata = [{} for _ in range(len(vectors))]
        
        points = []
        for i, (vec, id, meta) in enumerate(zip(vectors, ids, metadata)):
            points.append(
                rest.PointStruct(
                    id=id,
                    vector=vec.tolist(),
                    payload=meta
                )
            )
        
        self.client.upsert(
            collection_name=collection_name,
            points=points
        )
        
        return ids
    
    def get_vector(self, collection_name: str, id: str) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]]]:
        try:
            point = self.client.retrieve(
                collection_name=collection_name,
                ids=[id],
                with_vectors=True,
                with_payload=True
            )
            
            if point and len(point) > 0:
                return np.array(point[0].vector), point[0].payload
            return None, None
        except Exception as e:
            return None, None
    
    def delete_vectors(self, collection_name: str, ids: List[str]) -> int:
        try:
            result = self.client.delete(
                collection_name=collection_name,
                points_selector=rest.PointIdsList(
                    points=ids
                )
            )
            return result.status.acknowledged
        except Exception as e:
            return 0
    
    def search(self, 
              collection_name: str, 
              query_vector: np.ndarray, 
              limit: int = 10,
              filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        try:
            qdrant_filter = None
            if filter:
                qdrant_filter = self._convert_filter(filter)
            
            print(f"DEBUG: Qdrant search in collection '{collection_name}', limit: {limit}")
            print(f"DEBUG: Vector shape: {query_vector.shape}, filter: {qdrant_filter}")
            
            results = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector.tolist(),
                limit=limit,
                query_filter=qdrant_filter,
                with_payload=True
            )
            
            print(f"DEBUG: Qdrant raw results count: {len(results)}")
            if results:
                print(f"DEBUG: First result ID: {results[0].id}, score: {results[0].score}")
            
            formatted_results = [
                {
                    "id": str(hit.id),
                    "score": hit.score,
                    "metadata": hit.payload
                }
                for hit in results
            ]
            
            return formatted_results
        except Exception as e:
            print(f"DEBUG: Qdrant search error: {e}")
            return []
    
    def batch_search(self,
                    collection_name: str,
                    query_vectors: np.ndarray,
                    limit: int = 10,
                    filter: Optional[Dict[str, Any]] = None) -> List[List[Dict[str, Any]]]:
        try:
            qdrant_filter = None
            if filter:
                qdrant_filter = self._convert_filter(filter)
            
            requests = []
            for vec in query_vectors:
                requests.append(
                    rest.SearchRequest(
                        vector=vec.tolist(),
                        filter=qdrant_filter,
                        limit=limit,
                        with_payload=True
                    )
                )
            
            results = self.client.search_batch(
                collection_name=collection_name,
                requests=requests
            )
            
            formatted_results = []
            for batch in results:
                formatted_batch = [
                    {
                        "id": str(hit.id),
                        "score": hit.score,
                        "metadata": hit.payload
                    }
                    for hit in batch
                ]
                formatted_results.append(formatted_batch)
            
            return formatted_results
        except Exception as e:
            return [[] for _ in range(len(query_vectors))]
    
    def count(self, collection_name: str, filter: Optional[Dict[str, Any]] = None) -> int:
        try:
            qdrant_filter = None
            if filter:
                qdrant_filter = self._convert_filter(filter)
            
            result = self.client.count(
                collection_name=collection_name,
                count_filter=qdrant_filter
            )
            
            return result.count
        except Exception as e:
            return 0
    
    def update_metadata(self, collection_name: str, id: str, metadata: Dict[str, Any]) -> bool:
        try:
            self.client.set_payload(
                collection_name=collection_name,
                payload=metadata,
                points=[id]
            )
            return True
        except Exception as e:
            return False
    
    def _convert_filter(self, filter: Dict[str, Any]) -> rest.Filter:
        must = []
        
        for key, value in filter.items():
            if isinstance(value, str):
                must.append(
                    rest.FieldCondition(
                        key=key,
                        match=rest.MatchValue(value=value)
                    )
                )
            elif isinstance(value, list):
                must.append(
                    rest.FieldCondition(
                        key=key,
                        match=rest.MatchAny(any=value)
                    )
                )
            elif isinstance(value, (int, float)):
                must.append(
                    rest.FieldCondition(
                        key=key,
                        range=rest.Range(gte=value, lte=value)
                    )
                )
        
        if must:
            return rest.Filter(must=must)
        return None 