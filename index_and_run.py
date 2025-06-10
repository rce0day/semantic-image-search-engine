import os
import sys
import time
import uuid
import logging
from pathlib import Path
from typing import List, Dict, Any
import concurrent.futures
from tqdm import tqdm

current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(current_dir, 'semantic_img'))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_image_files(directory: Path, extensions: List[str] = None) -> List[Path]:
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.tiff']
    
    image_files = []
    for ext in extensions:
        image_files.extend(directory.rglob(f'*{ext}'))
        image_files.extend(directory.rglob(f'*{ext.upper()}'))
    
    return list(set(image_files))

def batch_index_images(image_paths: List[Path], batch_size: int = 100) -> Dict[str, Any]:
    from semantic_img.matching.full_image import FullImageMatcher
    from semantic_img.matching.partial_image import PartialImageMatcher
    
    logger.info("Initializing GPU-optimized matchers...")
    full_matcher = FullImageMatcher(collection_name="full_cats")
    partial_matcher = PartialImageMatcher(collection_name="partial_cats")
    
    initial_full_count = full_matcher.count()
    initial_partial_count = partial_matcher.count()
    
    logger.info(f"Initial counts - Full: {initial_full_count}, Partial: {initial_partial_count}")
    
    total_processed = 0
    successful_full = 0
    successful_partial = 0
    errors = []
    
    batches = [image_paths[i:i + batch_size] for i in range(0, len(image_paths), batch_size)]
    
    for batch_idx, batch in enumerate(tqdm(batches, desc="Processing batches")):
        logger.info(f"Processing batch {batch_idx + 1}/{len(batches)} ({len(batch)} images)")
        
        try:
            batch_paths = []
            batch_ids = []
            batch_metadatas = []
            
            for img_path in batch:
                if img_path.exists() and img_path.is_file():
                    batch_paths.append(img_path)
                    batch_ids.append(str(uuid.uuid4()))
                    batch_metadatas.append({
                        "filename": img_path.name,
                        "original_path": str(img_path),
                        "file_size": img_path.stat().st_size,
                        "indexed_at": time.time(),
                        "image_stem": img_path.stem
                    })
            
            if not batch_paths:
                continue
            
            logger.info(f"Indexing {len(batch_paths)} images for full matching...")
            start_time = time.time()
            full_ids = full_matcher.batch_index_images(
                image_paths=batch_paths,
                image_ids=batch_ids.copy(),
                metadatas=[m.copy() for m in batch_metadatas]
            )
            full_time = time.time() - start_time
            logger.info(f"Full indexing completed in {full_time:.2f}s")
            
            logger.info(f"Indexing {len(batch_paths)} images for partial matching...")
            start_time = time.time()
            partial_ids = partial_matcher.batch_index_images(
                image_paths=batch_paths,
                image_ids=batch_ids.copy(),
                metadatas=[m.copy() for m in batch_metadatas]
            )
            partial_time = time.time() - start_time
            logger.info(f"Partial indexing completed in {partial_time:.2f}s")
            
            successful_full += len(full_ids)
            successful_partial += len(partial_ids)
            total_processed += len(batch_paths)
            
        except Exception as e:
            error_msg = f"Error processing batch {batch_idx + 1}: {str(e)}"
            logger.error(error_msg)
            errors.append(error_msg)
            continue
    
    final_full_count = full_matcher.count()
    final_partial_count = partial_matcher.count()
    
    results = {
        "total_processed": total_processed,
        "successful_full": successful_full,
        "successful_partial": successful_partial,
        "initial_counts": {"full": initial_full_count, "partial": initial_partial_count},
        "final_counts": {"full": final_full_count, "partial": final_partial_count},
        "errors": errors
    }
    
    return results

def optimize_gpu_settings():
    import torch
    
    if torch.cuda.is_available():
        logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA Version: {torch.version.cuda}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.cuda.set_per_process_memory_fraction(0.85)
        
        torch.cuda.empty_cache()
        
        logger.info("GPU optimization settings applied")
    else:
        logger.warning("No GPU detected! Performance will be significantly slower.")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="GPU-optimized image indexing for semantic search")
    parser.add_argument("--images-dir", type=str, default="images", help="Directory containing images to index")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size for processing")
    parser.add_argument("--extensions", nargs="+", default=['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'], 
                       help="Image file extensions to process")
    parser.add_argument("--dry-run", action="store_true", help="Just count images without indexing")
    
    args = parser.parse_args()
    
    optimize_gpu_settings()
    
    images_dir = Path(args.images_dir)
    if not images_dir.exists():
        logger.error(f"Images directory does not exist: {images_dir}")
        return
    
    logger.info(f"Scanning for images in: {images_dir}")
    image_files = get_image_files(images_dir, args.extensions)
    logger.info(f"Found {len(image_files)} image files")
    
    if args.dry_run:
        logger.info("Dry run mode - not indexing images")
        for img in image_files[:10]:
            logger.info(f"  {img}")
        if len(image_files) > 10:
            logger.info(f"  ... and {len(image_files) - 10} more")
        return
    
    if not image_files:
        logger.warning("No image files found!")
        return
    
    logger.info(f"Starting GPU-accelerated indexing of {len(image_files)} images...")
    start_time = time.time()
    
    results = batch_index_images(image_files, batch_size=args.batch_size)
    
    total_time = time.time() - start_time
    
    logger.info("=== INDEXING RESULTS ===")
    logger.info(f"Total time: {total_time:.2f}s")
    logger.info(f"Images processed: {results['total_processed']}")
    logger.info(f"Full images indexed: {results['successful_full']}")
    logger.info(f"Partial images indexed: {results['successful_partial']}")
    logger.info(f"Processing rate: {results['total_processed'] / total_time:.2f} images/second")
    
    if results['errors']:
        logger.warning(f"Errors encountered: {len(results['errors'])}")
        for error in results['errors'][:5]:
            logger.warning(f"  {error}")
    
    logger.info("=== COLLECTION COUNTS ===")
    logger.info(f"Full images: {results['initial_counts']['full']} -> {results['final_counts']['full']}")
    logger.info(f"Partial images: {results['initial_counts']['partial']} -> {results['final_counts']['partial']}")
    
    logger.info("Indexing completed!")

if __name__ == "__main__":
    main() 