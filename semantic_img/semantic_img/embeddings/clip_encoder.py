import os
import logging
from typing import List, Union, Dict, Any, Tuple
from pathlib import Path

import numpy as np
import PIL.Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import open_clip

from semantic_img.embeddings.encoder_base import ImageEncoder
from semantic_img import config


if torch.cuda.is_available():
    os.environ["PYTORCH_NO_CUDA_MEMORY_EFFICIENCY_WARNING"] = "1"
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.cuda.set_per_process_memory_fraction(0.85)


class ImageDataset(Dataset):
    
    def __init__(self, images: List[Union[str, Path, PIL.Image.Image]], transform):
        self.images = images
        self.transform = transform
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = self.images[idx]
        if not isinstance(img, PIL.Image.Image):
            img = PIL.Image.open(img).convert('RGB')
        return self.transform(img)


_GLOBAL_ENCODER_INSTANCE = None

class ClipEncoder:
    def __init__(self, 
                 model_name: str = None,
                 device: str = None,
                 use_gpu: bool = None,
                 batch_size: int = None,
                 **kwargs):
        global _GLOBAL_ENCODER_INSTANCE
        
        if _GLOBAL_ENCODER_INSTANCE is not None:
            print("DEBUG: Reusing existing CLIP encoder instance for consistent embeddings")
            self.model = _GLOBAL_ENCODER_INSTANCE.model
            self.preprocess = _GLOBAL_ENCODER_INSTANCE.preprocess
            self.device = _GLOBAL_ENCODER_INSTANCE.device
            self._embedding_dim = _GLOBAL_ENCODER_INSTANCE._embedding_dim
            self.batch_size = _GLOBAL_ENCODER_INSTANCE.batch_size
            self.use_gpu = _GLOBAL_ENCODER_INSTANCE.use_gpu
            return
            
        if model_name is None:
            from semantic_img import config
            model_name = config.EMBEDDING_MODEL
            
        from semantic_img import config
        self.batch_size = batch_size if batch_size is not None else config.EMBEDDING_BATCH_SIZE
        self.use_gpu = use_gpu if use_gpu is not None else config.USE_GPU
        
        self.device = device
        if self.device is None:
            self.device = "cuda" if self.use_gpu and torch.cuda.is_available() else "cpu"
            
        logging.info(f"Loaded {model_name} model config.")
        
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name=model_name,
            pretrained="openai",
            device=self.device
        )
        
        self._embedding_dim = self.model.visual.output_dim
        
        _GLOBAL_ENCODER_INSTANCE = self
        
    def encode(self, images: List[Union[str, Path, PIL.Image.Image]]) -> np.ndarray:
        if not images:
            return np.array([])
            
        dataset = ImageDataset(images, self.preprocess)
        
        batch_size = self.batch_size
        if self.device == 'cuda':
            batch_size = config.GPU_INFERENCE_BATCH_SIZE if hasattr(config, 'GPU_INFERENCE_BATCH_SIZE') else batch_size
        
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size,
            num_workers=2 if self.device == 'cuda' and os.name != 'nt' else 0,  # Optimized for GPU
            pin_memory=True if self.device == 'cuda' else False,
            persistent_workers=True if self.device == 'cuda' and os.name != 'nt' else False
        )
        
        embeddings = []
        
        use_amp = (self.device == 'cuda' and 
                  hasattr(config, 'GPU_MIXED_PRECISION') and 
                  config.GPU_MIXED_PRECISION and
                  hasattr(torch, 'amp'))
        
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(self.device, non_blocking=True if self.device == 'cuda' else False)
                
                if use_amp:
                    with torch.amp.autocast(device_type='cuda'):
                        batch_embeddings = self.model.visual(batch)
                        batch_embeddings = batch_embeddings / batch_embeddings.norm(dim=1, keepdim=True)
                else:
                    batch_embeddings = self.model.visual(batch)
                    batch_embeddings = batch_embeddings / batch_embeddings.norm(dim=1, keepdim=True)
                
                embeddings.append(batch_embeddings.cpu().numpy())
                
                if self.device == 'cuda' and len(embeddings) % 10 == 0:
                    torch.cuda.empty_cache()
                
        return np.vstack(embeddings)
    
    def encode_sliding_windows(self, 
                               image: Union[str, Path, PIL.Image.Image],
                               window_sizes: List[tuple] = None,
                               stride: int = None) -> Dict[Tuple[int, int, int, int], np.ndarray]:

        if window_sizes is None:
            window_sizes = config.SLIDING_WINDOW_SIZES
        if stride is None:
            stride = config.SLIDING_WINDOW_STRIDE
            
        if not isinstance(image, PIL.Image.Image):
            image = PIL.Image.open(image).convert('RGB')
            
        width, height = image.size
        
        normalize = T.Normalize(mean=config.IMAGE_MEAN, std=config.IMAGE_STD)
        to_tensor = T.ToTensor()
        
        windows = []
        window_positions = []
        
        for win_width, win_height in window_sizes:
            if win_width > width or win_height > height:
                continue
                
            for y in range(0, height - win_height + 1, stride):
                for x in range(0, width - win_width + 1, stride):
                    window = image.crop((x, y, x + win_width, y + win_height))
                    
                    window = window.resize((config.IMAGE_SIZE, config.IMAGE_SIZE), PIL.Image.LANCZOS)
                    
                    windows.append(normalize(to_tensor(window)))
                    window_positions.append((x, y, win_width, win_height))
        
        if not windows:
            return {}
            
        batch_size = config.GPU_BATCH_SIZE if self.device == 'cuda' and hasattr(config, 'GPU_BATCH_SIZE') else 32
        window_embeddings_dict = {}
        
        use_amp = (self.device == 'cuda' and 
                  hasattr(config, 'GPU_MIXED_PRECISION') and 
                  config.GPU_MIXED_PRECISION and
                  hasattr(torch, 'amp'))
        
        for i in range(0, len(windows), batch_size):
            batch_windows = windows[i:i+batch_size]
            batch_positions = window_positions[i:i+batch_size]
            
            windows_tensor = torch.stack(batch_windows).to(self.device, non_blocking=True if self.device == 'cuda' else False)
            
            with torch.no_grad():
                if use_amp:
                    with torch.amp.autocast(device_type='cuda'):
                        window_embeddings = self.model.visual(windows_tensor)
                        window_embeddings = window_embeddings / window_embeddings.norm(dim=1, keepdim=True)
                else:
                    window_embeddings = self.model.visual(windows_tensor)
                    window_embeddings = window_embeddings / window_embeddings.norm(dim=1, keepdim=True)
                
                window_embeddings = window_embeddings.cpu().numpy()
                
            for pos, emb in zip(batch_positions, window_embeddings):
                window_embeddings_dict[pos] = emb
                
            if self.device == 'cuda' and i % (batch_size * 5) == 0:
                torch.cuda.empty_cache()
            
        return window_embeddings_dict
    
    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim
    
    @property
    def model_name(self) -> str:
        return self.model.name
    
    def get_metadata(self) -> Dict[str, Any]:
        return {
            'model_name': self.model.name,
            'embedding_dim': self._embedding_dim,
            'device': self.device,
            'batch_size': self.batch_size,
            'use_gpu': self.use_gpu
        } 