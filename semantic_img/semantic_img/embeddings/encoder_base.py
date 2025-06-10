from abc import ABC, abstractmethod
from typing import List, Union, Dict, Any
import numpy as np
from pathlib import Path
import PIL.Image


class ImageEncoder(ABC):

    @abstractmethod
    def encode(self, images: List[Union[str, Path, PIL.Image.Image]]) -> np.ndarray:
        pass
    
    @abstractmethod
    def encode_sliding_windows(self, 
                               image: Union[str, Path, PIL.Image.Image],
                               window_sizes: List[tuple] = None,
                               stride: int = None) -> Dict[tuple, np.ndarray]:
        pass
    
    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        pass
    
    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        pass 