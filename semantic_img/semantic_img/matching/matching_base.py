from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union
from pathlib import Path
import PIL.Image


class ImageMatcher(ABC):
    
    @abstractmethod
    def index_image(self, 
                    image_path: Union[str, Path], 
                    image_id: str = None,
                    metadata: Dict[str, Any] = None) -> str:
        pass
    
    @abstractmethod
    def batch_index_images(self,
                           image_paths: List[Union[str, Path]],
                           image_ids: List[str] = None,
                           metadatas: List[Dict[str, Any]] = None) -> List[str]:
        pass
    
    @abstractmethod
    def match(self,
              query_image: Union[str, Path, PIL.Image.Image],
              limit: int = 10,
              filter: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        pass
    
    @abstractmethod
    def delete_image(self, image_id: str) -> bool:
        pass
    
    @abstractmethod
    def count(self, filter: Dict[str, Any] = None) -> int:
        pass 