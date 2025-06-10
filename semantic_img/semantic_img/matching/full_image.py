import os
import uuid
from typing import List, Dict, Any, Union, Optional
from pathlib import Path
import PIL.Image
import numpy as np

from semantic_img.matching.matching_base import ImageMatcher
from semantic_img.embeddings.clip_encoder import ClipEncoder
from semantic_img.indexing.qdrant_index import QdrantIndex
from semantic_img import config


class FullImageMatcher(ImageMatcher):
    
    def __init__(self, 
                 collection_name: str = "full_cats",
                 encoder: Optional[ClipEncoder] = None,
                 index: Optional[QdrantIndex] = None):

        self.collection_name = collection_name
        
        self.encoder = encoder if encoder else ClipEncoder()
        
        self.index = index if index else QdrantIndex()
        
        if collection_name not in self.index.list_collections():
            self.index.create_collection(
                collection_name=collection_name,
                vector_size=self.encoder.embedding_dim
            )
    
    def index_image(self, 
                   image_path: Union[str, Path], 
                   image_id: str = None,
                   metadata: Dict[str, Any] = None) -> str:

        if image_id is None:
            image_id = str(uuid.uuid4())
            
        if metadata is None:
            metadata = {}
            
        image_path_str = str(image_path)
        metadata["image_path"] = image_path_str
        
        embedding = self.encoder.encode([image_path])[0]
        
        self.index.add_vectors(
            collection_name=self.collection_name,
            vectors=np.array([embedding]),
            ids=[image_id],
            metadata=[metadata]
        )
        
        return image_id
    
    def batch_index_images(self,
                           image_paths: List[Union[str, Path]],
                           image_ids: List[str] = None,
                           metadatas: List[Dict[str, Any]] = None) -> List[str]:

        if image_ids is None:
            image_ids = [str(uuid.uuid4()) for _ in range(len(image_paths))]
            
        if metadatas is None:
            metadatas = [{} for _ in range(len(image_paths))]
            
        for i, image_path in enumerate(image_paths):
            metadatas[i]["image_path"] = str(image_path)
            
        embeddings = self.encoder.encode(image_paths)
        
        self.index.add_vectors(
            collection_name=self.collection_name,
            vectors=embeddings,
            ids=image_ids,
            metadata=metadatas
        )
        
        return image_ids
    
    def match(self,
             query_image: Union[str, Path, PIL.Image.Image],
             limit: int = 10,
             filter: Dict[str, Any] = None) -> List[Dict[str, Any]]:

        query_embedding = self.encoder.encode([query_image])[0]
        print(f"DEBUG: Generated embedding for full image match, shape: {query_embedding.shape}")
        
        print(f"DEBUG: Searching collection '{self.collection_name}' with limit {limit}, threshold {config.FULL_MATCH_THRESHOLD}")
        results = self.index.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=limit,
            filter=filter
        )
        print(f"DEBUG: Raw search results: {len(results)} items found")
        
        # NOTE: Removed threshold filtering to allow all results
        # threshold = config.FULL_MATCH_THRESHOLD
        # results = [r for r in results if r["score"] >= threshold]
        # print(f"DEBUG: After threshold filtering ({threshold}): {len(results)} items remain")
        
        print(f"DEBUG: Returning all {len(results)} results regardless of score")
        return results
    
    def delete_image(self, image_id: str) -> bool:
        deleted = self.index.delete_vectors(
            collection_name=self.collection_name,
            ids=[image_id]
        )
        
        return deleted > 0
    
    def count(self, filter: Dict[str, Any] = None) -> int:
        return self.index.count(
            collection_name=self.collection_name,
            filter=filter
        ) 