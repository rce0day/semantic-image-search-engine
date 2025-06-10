from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np


class VectorIndex(ABC):

    @abstractmethod
    def create_collection(self, collection_name: str, vector_size: int, **kwargs) -> bool:
        pass
    
    @abstractmethod
    def drop_collection(self, collection_name: str) -> bool:
        pass
    
    @abstractmethod
    def list_collections(self) -> List[str]:
        pass
    
    @abstractmethod
    def collection_info(self, collection_name: str) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def add_vectors(self, 
                    collection_name: str, 
                    vectors: np.ndarray, 
                    ids: Optional[List[str]] = None,
                    metadata: Optional[List[Dict[str, Any]]] = None) -> List[str]:
        pass
    
    @abstractmethod
    def get_vector(self, collection_name: str, id: str) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]]]:
        pass
    
    @abstractmethod
    def delete_vectors(self, collection_name: str, ids: List[str]) -> int:
        pass
    
    @abstractmethod
    def search(self, 
               collection_name: str, 
               query_vector: np.ndarray, 
               limit: int = 10,
               filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        pass
    
    @abstractmethod
    def batch_search(self,
                     collection_name: str,
                     query_vectors: np.ndarray,
                     limit: int = 10,
                     filter: Optional[Dict[str, Any]] = None) -> List[List[Dict[str, Any]]]:
        pass
    
    @abstractmethod
    def count(self, collection_name: str, filter: Optional[Dict[str, Any]] = None) -> int:
        pass
    
    @abstractmethod
    def update_metadata(self, collection_name: str, id: str, metadata: Dict[str, Any]) -> bool:
        pass 