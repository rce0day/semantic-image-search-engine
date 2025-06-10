import os
import sys
import logging
from pathlib import Path

current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(current_dir, 'semantic_img'))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_qdrant():
    from semantic_img.indexing.qdrant_index import QdrantIndex
    from semantic_img import config
    
    logger.info("Checking Qdrant connection...")
    try:
        index = QdrantIndex(
            host=config.QDRANT_HOST,
            port=config.QDRANT_PORT,
            grpc_port=config.QDRANT_GRPC_PORT
        )
        collections = index.client.get_collections()
        logger.info(f"Qdrant connection successful. Found collections: {collections}")
        return True
    except Exception as e:
        logger.error(f"Failed to connect to Qdrant: {e}")
        return False

def check_indexed_images():
    try:
        from semantic_img.matching.full_image import FullImageMatcher
        from semantic_img.matching.partial_image import PartialImageMatcher
        
        full_matcher = FullImageMatcher()
        partial_matcher = PartialImageMatcher()
        
        full_count = full_matcher.count()
        partial_count = partial_matcher.count()
        
        logger.info(f"Indexed images count:")
        logger.info(f"  - Full image collection: {full_count}")
        logger.info(f"  - Partial image collection: {partial_count}")
        
        full_collection = full_matcher.collection_name
        partial_collection = partial_matcher.collection_name
        
        logger.info(f"Collection names:")
        logger.info(f"  - Full image collection: {full_collection}")
        logger.info(f"  - Partial image collection: {partial_collection}")
        
        return full_count, partial_count
    except Exception as e:
        logger.error(f"Error checking indexed images: {e}")
        return 0, 0

def check_search_thresholds():
    from semantic_img import config
    
    logger.info("Current search thresholds:")
    logger.info(f"  - Full match threshold: {config.FULL_MATCH_THRESHOLD}")
    logger.info(f"  - Partial match threshold: {config.PARTIAL_MATCH_THRESHOLD}")
    
    if config.FULL_MATCH_THRESHOLD > 0.7:
        logger.warning(f"Full match threshold {config.FULL_MATCH_THRESHOLD} might be too high. Consider lowering it to 0.6-0.7.")
    
    if config.PARTIAL_MATCH_THRESHOLD > 0.7:
        logger.warning(f"Partial match threshold {config.PARTIAL_MATCH_THRESHOLD} might be too high. Consider lowering it to 0.6-0.7.")

def index_test_image(image_path):
    import time
    from semantic_img.matching.full_image import FullImageMatcher
    from semantic_img.matching.partial_image import PartialImageMatcher
    
    image_path = Path(image_path)
    if not image_path.exists():
        logger.error(f"Image file does not exist: {image_path}")
        return False
    
    try:
        full_matcher = FullImageMatcher()
        partial_matcher = PartialImageMatcher()
        
        full_count_before = full_matcher.count()
        partial_count_before = partial_matcher.count()
        
        test_id = f"test_{int(time.time())}"
        
        metadata = {
            "filename": image_path.name,
            "original_path": str(image_path),
            "test": True
        }
        
        logger.info(f"Indexing test image: {image_path}")
        
        full_matcher.index_image(
            image_path=image_path,
            image_id=test_id,
            metadata=metadata
        )
        
        partial_matcher.index_image(
            image_path=image_path,
            image_id=test_id,
            metadata=metadata
        )
        
        full_count_after = full_matcher.count()
        partial_count_after = partial_matcher.count()
        
        logger.info(f"Full image indexing: {full_count_before} -> {full_count_after}")
        logger.info(f"Partial image indexing: {partial_count_before} -> {partial_count_after}")
        
        if full_count_after > full_count_before or partial_count_after > partial_count_before:
            logger.info("Successfully indexed test image")
            return True
        else:
            logger.warning("Could not verify if test image was indexed")
            return False
        
    except Exception as e:
        logger.error(f"Error indexing test image: {e}")
        return False

def test_search(image_path):
    from semantic_img.matching.full_image import FullImageMatcher
    from semantic_img.matching.partial_image import PartialImageMatcher
    from semantic_img.preprocessing.image_processor import ImageProcessor
    from semantic_img import config
    
    image_path = Path(image_path)
    if not image_path.exists():
        logger.error(f"Image file does not exist: {image_path}")
        return
    
    try:
        processor = ImageProcessor()
        full_matcher = FullImageMatcher(collection_name="full_cats")
        partial_matcher = PartialImageMatcher(collection_name="partial_cats")
        
        processed_img = processor.process_image(image_path, resize=True)
        old_full_threshold = config.FULL_MATCH_THRESHOLD
        old_partial_threshold = config.PARTIAL_MATCH_THRESHOLD
        config.FULL_MATCH_THRESHOLD = 0.5
        config.PARTIAL_MATCH_THRESHOLD = 0.5
        
        try:
            logger.info("Testing full image matching...")
            full_results = full_matcher.match(processed_img, limit=10)
            logger.info(f"Full matching found {len(full_results)} results")
            
            for i, result in enumerate(full_results[:3]):
                logger.info(f"  {i+1}. ID: {result['id']}, Score: {result['score']:.4f}")
            
            logger.info("Testing partial image matching...")
            partial_results = partial_matcher.match(processed_img, limit=10)
            logger.info(f"Partial matching found {len(partial_results)} results")
            
            for i, result in enumerate(partial_results[:3]):
                logger.info(f"  {i+1}. ID: {result['id']}, Score: {result['score']:.4f}")
        finally:
            config.FULL_MATCH_THRESHOLD = old_full_threshold
            config.PARTIAL_MATCH_THRESHOLD = old_partial_threshold
        
    except Exception as e:
        logger.error(f"Error testing search: {e}")

def check_system_status():
    logger.info("Checking system status...")
    
    qdrant_ok = check_qdrant()
    if not qdrant_ok:
        logger.error("Qdrant connection failed")
        return False
    
    full_count, partial_count = check_indexed_images()
    if full_count == 0 and partial_count == 0:
        logger.warning("No images indexed in either collection")
    
    check_search_thresholds()
    
    test_image = None
    for path in [
        Path("images/archive/CAT_01/00000296_022.jpg"),
        *Path("images").glob("**/*.jpg")
    ]:
        if path.exists():
            test_image = path
            break
    
    if not test_image:
        logger.error("No test image found")
        return False
    
    logger.info(f"Testing search with image: {test_image}")
    
    old_thresholds = modify_thresholds(full_threshold=0.5, partial_threshold=0.5)
    
    try:
        test_search(test_image)
        logger.info("System check complete")
        return True
    except Exception as e:
        logger.error(f"Error during system check: {e}")
        return False
    finally:
        if old_thresholds:
            modify_thresholds(**old_thresholds)

def modify_thresholds(full_threshold=None, partial_threshold=None):
    import re
    
    config_path = Path("semantic_img/semantic_img/config.py")
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        return False
    
    try:
        content = config_path.read_text()
        
        old_full = None
        old_partial = None
        
        full_match = re.search(r'FULL_MATCH_THRESHOLD\s*=\s*([\d.]+)', content)
        if full_match:
            old_full = float(full_match.group(1))
            
        partial_match = re.search(r'PARTIAL_MATCH_THRESHOLD\s*=\s*([\d.]+)', content)
        if partial_match:
            old_partial = float(partial_match.group(1))
        
        if full_threshold is not None:
            logger.info(f"Changing FULL_MATCH_THRESHOLD to {full_threshold}")
            content = re.sub(
                r'FULL_MATCH_THRESHOLD\s*=\s*[\d.]+', 
                f'FULL_MATCH_THRESHOLD = {full_threshold}',
                content
            )
        
        if partial_threshold is not None:
            logger.info(f"Changing PARTIAL_MATCH_THRESHOLD to {partial_threshold}")
            content = re.sub(
                r'PARTIAL_MATCH_THRESHOLD\s*=\s*[\d.]+', 
                f'PARTIAL_MATCH_THRESHOLD = {partial_threshold}',
                content
            )
        
        config_path.write_text(content)
        logger.info("Thresholds updated successfully")
        
        return {
            "full_threshold": old_full,
            "partial_threshold": old_partial
        }
    
    except Exception as e:
        logger.error(f"Error modifying thresholds: {e}")
        return False

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Troubleshoot semantic image search.")
    parser.add_argument("--check-qdrant", action="store_true", help="Check Qdrant connection")
    parser.add_argument("--check-indexed", action="store_true", help="Check indexed images")
    parser.add_argument("--check-thresholds", action="store_true", help="Check search thresholds")
    parser.add_argument("--index-test", type=str, help="Index a test image")
    parser.add_argument("--test-search", type=str, help="Test search with an image")
    parser.add_argument("--lower-thresholds", action="store_true", help="Lower match thresholds")
    parser.add_argument("--full-threshold", type=float, help="Set full match threshold")
    parser.add_argument("--partial-threshold", type=float, help="Set partial match threshold")
    parser.add_argument("--check-system", action="store_true", help="Check the entire system")
    
    args = parser.parse_args()
    
    if args.check_qdrant:
        check_qdrant()
    
    if args.check_indexed:
        check_indexed_images()
    
    if args.check_thresholds:
        check_search_thresholds()
    
    if args.lower_thresholds:
        modify_thresholds(full_threshold=0.6, partial_threshold=0.6)
    
    if args.full_threshold is not None or args.partial_threshold is not None:
        modify_thresholds(
            full_threshold=args.full_threshold,
            partial_threshold=args.partial_threshold
        )
    
    if args.index_test:
        index_test_image(args.index_test)
    
    if args.test_search:
        test_search(args.test_search)
        
    if args.check_system:
        check_system_status()
    
    if not any(vars(args).values()):
        logger.info("Running all diagnostic checks...")
        check_system_status()

if __name__ == "__main__":
    main() 