#!/usr/bin/env python
"""
Script for batch indexing images.
"""
import os
import sys
import argparse
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Set
import json
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from semantic_img.matching.full_image import FullImageMatcher
from semantic_img.matching.partial_image import PartialImageMatcher
from semantic_img import config


# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_image_files(directory: Path, extensions: Set[str] = None) -> List[Path]:
    """
    Get all image files in a directory.
    
    Args:
        directory: Directory to scan
        extensions: Set of allowed file extensions
        
    Returns:
        List of image file paths
    """
    if extensions is None:
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
    
    # Get all files recursively
    all_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if os.path.splitext(file)[1].lower() in extensions:
                all_files.append(Path(root) / file)
    
    return all_files


def index_image(
    image_path: Path,
    full_matcher: FullImageMatcher,
    partial_matcher: PartialImageMatcher,
    use_full_matching: bool,
    use_partial_matching: bool,
    metadata: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Index a single image.
    
    Args:
        image_path: Path to the image file
        full_matcher: Full image matcher
        partial_matcher: Partial image matcher
        use_full_matching: Whether to use full image matching
        use_partial_matching: Whether to use partial image matching
        metadata: Optional metadata to store with the image
        
    Returns:
        Dict with indexing result information
    """
    try:
        # Prepare metadata
        if metadata is None:
            metadata = {}
        
        # Add image path to metadata
        metadata["original_path"] = str(image_path)
        metadata["filename"] = image_path.name
        
        # Generate a consistent ID from the path
        import hashlib
        image_id = hashlib.md5(str(image_path).encode()).hexdigest()
        
        # Index the image with both matchers
        start_time = time.time()
        
        if use_full_matching:
            full_matcher.index_image(
                image_path=image_path,
                image_id=image_id,
                metadata=metadata
            )
        
        if use_partial_matching:
            partial_matcher.index_image(
                image_path=image_path,
                image_id=image_id,
                metadata=metadata
            )
        
        processing_time = time.time() - start_time
        
        return {
            "image_id": image_id,
            "path": str(image_path),
            "success": True,
            "processing_time": processing_time
        }
    except Exception as e:
        logger.error(f"Failed to index {image_path}: {e}")
        return {
            "path": str(image_path),
            "success": False,
            "error": str(e)
        }


def main():
    """Main function for the script."""
    # Define command-line arguments
    parser = argparse.ArgumentParser(description="Batch index images for semantic search.")
    parser.add_argument("--image-dir", type=str, required=True, help="Directory containing images to index")
    parser.add_argument("--collection-name", type=str, default="images", help="Name for the collection")
    parser.add_argument("--extensions", type=str, default=".jpg,.jpeg,.png", help="Comma-separated list of file extensions")
    parser.add_argument("--use-full-matching", action="store_true", default=True, help="Use full image matching")
    parser.add_argument("--use-partial-matching", action="store_true", default=True, help="Use partial image matching")
    parser.add_argument("--max-workers", type=int, default=config.NUM_WORKERS, help="Maximum number of worker threads")
    parser.add_argument("--metadata-file", type=str, help="JSON file with metadata for images")
    
    args = parser.parse_args()
    
    # Process arguments
    image_dir = Path(args.image_dir)
    if not image_dir.exists() or not image_dir.is_dir():
        logger.error(f"Image directory does not exist: {image_dir}")
        sys.exit(1)
        
    extensions = set(args.extensions.split(','))
    
    # Load metadata if provided
    metadata_dict = {}
    if args.metadata_file:
        metadata_path = Path(args.metadata_file)
        if not metadata_path.exists():
            logger.error(f"Metadata file does not exist: {metadata_path}")
            sys.exit(1)
            
        try:
            with open(metadata_path, 'r') as f:
                metadata_dict = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load metadata file: {e}")
            sys.exit(1)
    
    # Initialize matchers with collection name
    full_matcher = FullImageMatcher(collection_name=f"full_{args.collection_name}")
    partial_matcher = PartialImageMatcher(collection_name=f"partial_{args.collection_name}")
    
    # Get all image files
    logger.info(f"Scanning directory: {image_dir}")
    image_files = get_image_files(image_dir, extensions)
    logger.info(f"Found {len(image_files)} images to index")
    
    # Index images
    results = {
        "total": len(image_files),
        "success": 0,
        "failed": 0,
        "skipped": 0,
        "start_time": time.time(),
        "end_time": None,
        "processing_time": None,
        "failed_images": []
    }
    
    # Create a thread pool
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # Submit indexing tasks
        futures = []
        for image_path in image_files:
            # Get metadata for this image if available
            image_metadata = metadata_dict.get(image_path.name, {})
            
            future = executor.submit(
                index_image,
                image_path,
                full_matcher,
                partial_matcher,
                args.use_full_matching,
                args.use_partial_matching,
                image_metadata
            )
            futures.append(future)
        
        # Process results with progress bar
        for future in tqdm(futures, desc="Indexing images", unit="image"):
            result = future.result()
            if result.get("success", False):
                results["success"] += 1
            else:
                results["failed"] += 1
                results["failed_images"].append(result)
    
    # Calculate timing
    results["end_time"] = time.time()
    results["processing_time"] = results["end_time"] - results["start_time"]
    
    # Print summary
    logger.info(f"Indexing complete:")
    logger.info(f"  - Total images: {results['total']}")
    logger.info(f"  - Successfully indexed: {results['success']}")
    logger.info(f"  - Failed: {results['failed']}")
    logger.info(f"  - Total processing time: {results['processing_time']:.2f} seconds")
    
    # Save results to file
    output_file = Path(f"index_results_{args.collection_name}.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {output_file}")


if __name__ == "__main__":
    main() 