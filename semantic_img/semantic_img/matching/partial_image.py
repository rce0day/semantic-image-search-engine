import os
import uuid
from typing import List, Dict, Any, Union, Optional, Tuple
from pathlib import Path
import PIL.Image
import numpy as np
from collections import defaultdict

from semantic_img.matching.matching_base import ImageMatcher
from semantic_img.embeddings.clip_encoder import ClipEncoder
from semantic_img.indexing.qdrant_index import QdrantIndex
from semantic_img import config


class PartialImageMatcher(ImageMatcher):
    
    def __init__(self, 
                 collection_name: str = "partial_cats",
                 encoder: Optional[ClipEncoder] = None,
                 index: Optional[QdrantIndex] = None,
                 window_sizes: List[Tuple[int, int]] = None,
                 stride: int = None):

        self.collection_name = collection_name
        self.window_sizes = window_sizes if window_sizes else config.SLIDING_WINDOW_SIZES
        self.stride = stride if stride else config.SLIDING_WINDOW_STRIDE
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
        metadata["parent_id"] = image_id
        
        window_embeddings = self.encoder.encode_sliding_windows(
            image=image_path,
            window_sizes=self.window_sizes,
            stride=self.stride
        )
        
        if not window_embeddings:
            return image_id
        
        window_ids = []
        window_metadatas = []
        window_vectors = []
        
        for window_pos, embedding in window_embeddings.items():
            window_id = str(uuid.uuid4())
            
            window_metadata = metadata.copy()
            window_metadata["window_position"] = {
                "x": window_pos[0],
                "y": window_pos[1],
                "width": window_pos[2],
                "height": window_pos[3]
            }
            window_metadata["parent_id"] = image_id
            
            window_ids.append(window_id)
            window_metadatas.append(window_metadata)
            window_vectors.append(embedding)
        
        self.index.add_vectors(
            collection_name=self.collection_name,
            vectors=np.array(window_vectors),
            ids=window_ids,
            metadata=window_metadatas
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
        
        for i, (image_path, image_id, metadata) in enumerate(zip(image_paths, image_ids, metadatas)):
            self.index_image(image_path, image_id, metadata)
            
        return image_ids
    
    def match(self,
              query_image: Union[str, Path, PIL.Image.Image],
              limit: int = 10,
              filter: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        query_embedding = self.encoder.encode([query_image])[0]
        
        query_windows = self.encoder.encode_sliding_windows(
            image=query_image,
            window_sizes=self.window_sizes,
            stride=self.stride
        )
        
        all_window_results = []
        
        search_limit = limit * 10
        print(f"DEBUG: Searching collection '{self.collection_name}' with full image")
        full_query_results = self.index.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=search_limit,
            filter=filter
        )
        all_window_results.extend(full_query_results)
        
        if query_windows:
            print(f"DEBUG: Searching with {len(query_windows)} query windows")
            for window_embedding in list(query_windows.values())[:5]:
                window_results = self.index.search(
                    collection_name=self.collection_name,
                    query_vector=window_embedding,
                    limit=search_limit // 2,
                    filter=filter
                )
                all_window_results.extend(window_results)
        
        print(f"DEBUG: Total window results collected: {len(all_window_results)}")
        
        if not all_window_results:
            return []
        
        parent_scores = defaultdict(list)
        parent_metadatas = {}
        parent_best_windows = {}
        
        for result in all_window_results:
            parent_id = result["metadata"].get("parent_id")
            if parent_id:
                parent_scores[parent_id].append(result["score"])
                if parent_id not in parent_metadatas or result["score"] > max(parent_scores[parent_id][:-1], default=0):
                    parent_metadatas[parent_id] = result["metadata"]
                    parent_best_windows[parent_id] = result["metadata"].get("window_position", {})
        
        aggregated_results = []
        for parent_id, scores in parent_scores.items():
            max_score = max(scores)
            avg_score = sum(scores) / len(scores)
            weighted_score = max_score * 0.7 + avg_score * 0.3 + (len(scores) / search_limit) * 0.1
            
            metadata = parent_metadatas[parent_id].copy()
            if "window_position" in metadata:
                del metadata["window_position"]
            
            aggregated_results.append({
                "id": parent_id,
                "score": weighted_score,
                "max_score": max_score,
                "metadata": metadata,
                "matched_regions": len(scores),
                "avg_score": avg_score,
                "best_window": parent_best_windows.get(parent_id, {}),
                "match_type": "partial_to_full"
            })
        
        aggregated_results.sort(key=lambda x: x["score"], reverse=True)
        final_results = aggregated_results[:limit]
        
        print(f"DEBUG: Returning {len(final_results)} full images from partial matching")
        return final_results
    
    def delete_image(self, image_id: str) -> bool:
        filter = {"parent_id": image_id}
        
        count_before = self.index.count(
            collection_name=self.collection_name,
            filter=filter
        )
        
        if count_before == 0:
            return False
        
        search_results = self.index.search(
            collection_name=self.collection_name,
            query_vector=np.zeros(self.encoder.embedding_dim),
            limit=count_before,
            filter=filter
        )
        
        if not search_results:
            return False
        
        window_ids = [result["id"] for result in search_results]
        
        deleted = self.index.delete_vectors(
            collection_name=self.collection_name,
            ids=window_ids
        )
        
        return deleted > 0
    
    def count(self, filter: Dict[str, Any] = None) -> int:
        if filter is None:
            return self.index.count(self.collection_name) // 10
        
        return self.index.count(
            collection_name=self.collection_name,
            filter=filter
        ) 