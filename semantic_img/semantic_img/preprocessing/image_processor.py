import os
from typing import Union, Tuple, List, Optional
from pathlib import Path
import PIL.Image
import numpy as np
import cv2

from semantic_img import config


class ImageProcessor:
    
    def __init__(self, target_size: Tuple[int, int] = (config.IMAGE_SIZE, config.IMAGE_SIZE)):
        self.target_size = target_size
    
    def load_image(self, image_path: Union[str, Path]) -> PIL.Image.Image:
        return PIL.Image.open(image_path).convert('RGB')
    
    def resize_image(self, 
                     image: PIL.Image.Image, 
                     size: Tuple[int, int] = None, 
                     keep_aspect_ratio: bool = True) -> PIL.Image.Image:

        if size is None:
            size = self.target_size
            
        if keep_aspect_ratio:
            width, height = image.size
            ratio = min(size[0] / width, size[1] / height)
            new_size = (int(width * ratio), int(height * ratio))
            
            image = image.resize(new_size, PIL.Image.LANCZOS)
            
            new_image = PIL.Image.new('RGB', size, (0, 0, 0))
            
            paste_x = (size[0] - new_size[0]) // 2
            paste_y = (size[1] - new_size[1]) // 2
            new_image.paste(image, (paste_x, paste_y))
            
            return new_image
        else:
            return image.resize(size, PIL.Image.LANCZOS)
    
    def normalize_image(self, 
                        img_array: np.ndarray, 
                        mean: List[float] = config.IMAGE_MEAN, 
                        std: List[float] = config.IMAGE_STD) -> np.ndarray:

        img_array = img_array.astype(np.float32) / 255.0
        
        for i in range(3):
            img_array[:, :, i] = (img_array[:, :, i] - mean[i]) / std[i]
            
        return img_array
    
    def extract_sliding_windows(self, 
                               image: PIL.Image.Image, 
                               window_sizes: List[Tuple[int, int]] = None,
                               stride: int = None) -> List[Tuple[Tuple[int, int, int, int], PIL.Image.Image]]:

        if window_sizes is None:
            window_sizes = config.SLIDING_WINDOW_SIZES
        if stride is None:
            stride = config.SLIDING_WINDOW_STRIDE
            
        width, height = image.size
        windows = []
        
        for win_width, win_height in window_sizes:
            if win_width > width or win_height > height:
                continue
                
            for y in range(0, height - win_height + 1, stride):
                for x in range(0, width - win_width + 1, stride):
                    window = image.crop((x, y, x + win_width, y + win_height))
                    windows.append(((x, y, win_width, win_height), window))
        
        return windows
    
    def remove_text_overlays(self, 
                            image: PIL.Image.Image, 
                            min_contour_area: int = 100) -> PIL.Image.Image:

        img = np.array(image)
        img = img[:, :, ::-1].copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(gray)

        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            
            if (area > min_contour_area and 
                area < 0.01 * gray.shape[0] * gray.shape[1] and 
                0.1 < aspect_ratio < 10):
                
                cv2.drawContours(mask, [contour], 0, 255, -1)
        
        dst = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
        
        result = PIL.Image.fromarray(dst[:, :, ::-1])
        
        return result
    
    def enhance_image(self, 
                     image: PIL.Image.Image, 
                     contrast_factor: float = 1.2, 
                     brightness_factor: float = 1.1, 
                     sharpness_factor: float = 1.1) -> PIL.Image.Image:
        from PIL.ImageEnhance import Contrast, Brightness, Sharpness
        
        image = Contrast(image).enhance(contrast_factor)
        image = Brightness(image).enhance(brightness_factor)
        image = Sharpness(image).enhance(sharpness_factor)
        
        return image
    
    def process_image(self, 
                     image: Union[str, Path, PIL.Image.Image], 
                     resize: bool = True,
                     remove_text: bool = False,
                     enhance: bool = False) -> PIL.Image.Image:

        if not isinstance(image, PIL.Image.Image):
            image = self.load_image(image)
            
        if remove_text:
            image = self.remove_text_overlays(image)
            
        if enhance:
            image = self.enhance_image(image)
            
        if resize:
            image = self.resize_image(image)
            
        return image 