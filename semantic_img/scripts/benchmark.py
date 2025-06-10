#!/usr/bin/env python
"""
Benchmarking script for the semantic image search system.
"""
import os
import sys
import argparse
import time
import logging
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from semantic_img.matching.full_image import FullImageMatcher
from semantic_img.matching.partial_image import PartialImageMatcher
from semantic_img.preprocessing.image_processor import ImageProcessor
from semantic_img import config


# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_partial_images(
    image_path: Path, 
    output_dir: Path,
    num_samples: int = 5,
    min_size_ratio: float = 0.2, 
    max_size_ratio: float = 0.8
) -> List[Dict[str, Any]]:
    """
    Create partial images from an original image for benchmarking.
    
    Args:
        image_path: Path to the original image
        output_dir: Directory to save partial images
        num_samples: Number of partial images to create
        min_size_ratio: Minimum size ratio of the partial image
        max_size_ratio: Maximum size ratio of the partial image
        
    Returns:
        List of dicts with info about created partial images
    """
    # Load the image
    img = Image.open(image_path).convert('RGB')
    width, height = img.size
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate partial images
    partial_images = []
    
    for i in range(num_samples):
        # Randomly select size ratio
        size_ratio_w = random.uniform(min_size_ratio, max_size_ratio)
        size_ratio_h = random.uniform(min_size_ratio, max_size_ratio)
        
        # Calculate window size
        window_width = int(width * size_ratio_w)
        window_height = int(height * size_ratio_h)
        
        # Calculate position
        max_x = width - window_width
        max_y = height - window_height
        if max_x < 0 or max_y < 0:
            # Skip if window is larger than image
            continue
            
        x = random.randint(0, max_x)
        y = random.randint(0, max_y)
        
        # Extract window
        window = img.crop((x, y, x + window_width, y + window_height))
        
        # Save window
        window_path = output_dir / f"{image_path.stem}_partial_{i}.jpg"
        window.save(window_path)
        
        # Record information
        partial_images.append({
            "original_image": str(image_path),
            "partial_image": str(window_path),
            "position": {
                "x": x,
                "y": y,
                "width": window_width,
                "height": window_height
            },
            "size_ratio": {
                "width": size_ratio_w,
                "height": size_ratio_h
            }
        })
    
    return partial_images


def benchmark_partial_matching(
    original_images: List[Path],
    matcher: PartialImageMatcher,
    output_dir: Path,
    num_samples_per_image: int = 5
) -> Dict[str, Any]:
    """
    Benchmark partial image matching.
    
    Args:
        original_images: List of original images
        matcher: Partial image matcher
        output_dir: Directory to save partial images and results
        num_samples_per_image: Number of partial images to create per original
        
    Returns:
        Dictionary with benchmark results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create partial images
    all_partial_images = []
    for image_path in tqdm(original_images, desc="Creating partial images"):
        partial_dir = output_dir / "partial_images"
        partial_images = create_partial_images(
            image_path=image_path,
            output_dir=partial_dir,
            num_samples=num_samples_per_image
        )
        all_partial_images.extend(partial_images)
    
    # Save partial image info
    with open(output_dir / "partial_images.json", 'w') as f:
        json.dump(all_partial_images, f, indent=2)
    
    # Benchmark matching
    processor = ImageProcessor()
    results = []
    
    for partial_info in tqdm(all_partial_images, desc="Benchmarking partial matching"):
        original_path = partial_info["original_image"]
        partial_path = partial_info["partial_image"]
        
        # Process partial image
        partial_img = processor.process_image(partial_path)
        
        # Perform search
        start_time = time.time()
        search_results = matcher.match(partial_img, limit=10)
        search_time = time.time() - start_time
        
        # Check if original image is in results
        original_id = None
        for result in search_results:
            metadata = result["metadata"]
            if metadata.get("original_path") == original_path:
                original_id = result["id"]
                break
        
        # Record result
        results.append({
            "original_image": original_path,
            "partial_image": partial_path,
            "position": partial_info["position"],
            "size_ratio": partial_info["size_ratio"],
            "search_time_ms": search_time * 1000,
            "original_found": original_id is not None,
            "original_rank": next(
                (i for i, r in enumerate(search_results) if r["metadata"].get("original_path") == original_path), 
                -1
            ),
            "original_score": next(
                (r["score"] for r in search_results if r["metadata"].get("original_path") == original_path), 
                0
            ),
            "num_results": len(search_results),
            "top_result": {
                "id": search_results[0]["id"] if search_results else None,
                "score": search_results[0]["score"] if search_results else 0,
                "is_original": (search_results[0]["metadata"].get("original_path") == original_path) if search_results else False
            }
        })
    
    # Calculate success rate
    success_count = sum(1 for r in results if r["original_found"])
    success_rate = success_count / len(results) if results else 0
    
    # Calculate average search time
    avg_search_time = sum(r["search_time_ms"] for r in results) / len(results) if results else 0
    
    # Calculate average rank when found
    found_results = [r for r in results if r["original_found"]]
    avg_rank = sum(r["original_rank"] for r in found_results) / len(found_results) if found_results else 0
    
    # Calculate average score when found
    avg_score = sum(r["original_score"] for r in found_results) / len(found_results) if found_results else 0
    
    # Calculate success rate by size ratio
    size_ratio_bins = np.linspace(0, 1, 11)  # 0.0, 0.1, 0.2, ..., 1.0
    size_ratio_success = [0] * (len(size_ratio_bins) - 1)
    size_ratio_counts = [0] * (len(size_ratio_bins) - 1)
    
    for result in results:
        area_ratio = result["size_ratio"]["width"] * result["size_ratio"]["height"]
        bin_index = min(int(area_ratio * 10), 9)  # Ensure it fits in our bins
        
        size_ratio_counts[bin_index] += 1
        if result["original_found"]:
            size_ratio_success[bin_index] += 1
    
    size_ratio_success_rates = [
        (size_ratio_success[i] / size_ratio_counts[i]) if size_ratio_counts[i] > 0 else 0
        for i in range(len(size_ratio_bins) - 1)
    ]
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    # Size ratio vs. success rate
    plt.subplot(2, 2, 1)
    plt.bar(
        [(size_ratio_bins[i] + size_ratio_bins[i+1]) / 2 for i in range(len(size_ratio_bins) - 1)],
        size_ratio_success_rates,
        width=0.08
    )
    plt.xlabel('Size Ratio (area)')
    plt.ylabel('Success Rate')
    plt.title('Success Rate by Size Ratio')
    
    # Size ratio vs. score
    plt.subplot(2, 2, 2)
    area_ratios = [r["size_ratio"]["width"] * r["size_ratio"]["height"] for r in found_results]
    scores = [r["original_score"] for r in found_results]
    plt.scatter(area_ratios, scores, alpha=0.6)
    plt.xlabel('Size Ratio (area)')
    plt.ylabel('Match Score')
    plt.title('Match Score vs. Size Ratio')
    
    # Rank distribution
    plt.subplot(2, 2, 3)
    rank_counts = {}
    for r in found_results:
        rank = r["original_rank"]
        rank_counts[rank] = rank_counts.get(rank, 0) + 1
    
    plt.bar(rank_counts.keys(), rank_counts.values())
    plt.xlabel('Rank')
    plt.ylabel('Count')
    plt.title('Rank Distribution')
    
    # Overall stats
    plt.subplot(2, 2, 4)
    plt.axis('off')
    plt.text(0.1, 0.9, f'Success Rate: {success_rate:.2%}', fontsize=12)
    plt.text(0.1, 0.8, f'Average Search Time: {avg_search_time:.2f} ms', fontsize=12)
    plt.text(0.1, 0.7, f'Average Rank (when found): {avg_rank:.2f}', fontsize=12)
    plt.text(0.1, 0.6, f'Average Score (when found): {avg_score:.4f}', fontsize=12)
    plt.text(0.1, 0.5, f'Total Tests: {len(results)}', fontsize=12)
    plt.text(0.1, 0.4, f'Successful Tests: {success_count}', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_dir / "partial_matching_results.png")
    
    # Save detailed results
    with open(output_dir / "partial_matching_results.json", 'w') as f:
        json.dump({
            "success_rate": success_rate,
            "avg_search_time_ms": avg_search_time,
            "avg_rank_when_found": avg_rank,
            "avg_score_when_found": avg_score,
            "total_tests": len(results),
            "successful_tests": success_count,
            "size_ratio_bins": list(size_ratio_bins),
            "size_ratio_success_rates": size_ratio_success_rates,
            "detailed_results": results
        }, f, indent=2)
    
    return {
        "success_rate": success_rate,
        "avg_search_time_ms": avg_search_time,
        "avg_rank_when_found": avg_rank,
        "avg_score_when_found": avg_score,
        "total_tests": len(results),
        "successful_tests": success_count
    }


def benchmark_full_matching(
    test_images: List[Path],
    matcher: FullImageMatcher,
    output_dir: Path
) -> Dict[str, Any]:
    """
    Benchmark full image matching.
    
    Args:
        test_images: List of test images
        matcher: Full image matcher
        output_dir: Directory to save results
        
    Returns:
        Dictionary with benchmark results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Benchmark matching
    processor = ImageProcessor()
    results = []
    
    for image_path in tqdm(test_images, desc="Benchmarking full matching"):
        # Process image
        processed_img = processor.process_image(image_path)
        
        # Perform search
        start_time = time.time()
        search_results = matcher.match(processed_img, limit=10)
        search_time = time.time() - start_time
        
        # Check if image is in results
        image_id = None
        for result in search_results:
            metadata = result["metadata"]
            if metadata.get("original_path") == str(image_path):
                image_id = result["id"]
                break
        
        # Record result
        results.append({
            "image": str(image_path),
            "search_time_ms": search_time * 1000,
            "image_found": image_id is not None,
            "image_rank": next(
                (i for i, r in enumerate(search_results) if r["metadata"].get("original_path") == str(image_path)), 
                -1
            ),
            "image_score": next(
                (r["score"] for r in search_results if r["metadata"].get("original_path") == str(image_path)), 
                0
            ),
            "num_results": len(search_results),
            "top_result": {
                "id": search_results[0]["id"] if search_results else None,
                "score": search_results[0]["score"] if search_results else 0,
                "is_same": (search_results[0]["metadata"].get("original_path") == str(image_path)) if search_results else False
            }
        })
    
    # Calculate success rate
    success_count = sum(1 for r in results if r["image_found"])
    success_rate = success_count / len(results) if results else 0
    
    # Calculate average search time
    avg_search_time = sum(r["search_time_ms"] for r in results) / len(results) if results else 0
    
    # Calculate average rank when found
    found_results = [r for r in results if r["image_found"]]
    avg_rank = sum(r["image_rank"] for r in found_results) / len(found_results) if found_results else 0
    
    # Calculate average score when found
    avg_score = sum(r["image_score"] for r in found_results) / len(found_results) if found_results else 0
    
    # Plot results
    plt.figure(figsize=(10, 6))
    
    # Score distribution
    plt.subplot(1, 2, 1)
    scores = [r["image_score"] for r in found_results]
    plt.hist(scores, bins=20)
    plt.xlabel('Match Score')
    plt.ylabel('Count')
    plt.title('Score Distribution')
    
    # Overall stats
    plt.subplot(1, 2, 2)
    plt.axis('off')
    plt.text(0.1, 0.9, f'Success Rate: {success_rate:.2%}', fontsize=12)
    plt.text(0.1, 0.8, f'Average Search Time: {avg_search_time:.2f} ms', fontsize=12)
    plt.text(0.1, 0.7, f'Average Rank (when found): {avg_rank:.2f}', fontsize=12)
    plt.text(0.1, 0.6, f'Average Score (when found): {avg_score:.4f}', fontsize=12)
    plt.text(0.1, 0.5, f'Total Tests: {len(results)}', fontsize=12)
    plt.text(0.1, 0.4, f'Successful Tests: {success_count}', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_dir / "full_matching_results.png")
    
    # Save detailed results
    with open(output_dir / "full_matching_results.json", 'w') as f:
        json.dump({
            "success_rate": success_rate,
            "avg_search_time_ms": avg_search_time,
            "avg_rank_when_found": avg_rank,
            "avg_score_when_found": avg_score,
            "total_tests": len(results),
            "successful_tests": success_count,
            "detailed_results": results
        }, f, indent=2)
    
    return {
        "success_rate": success_rate,
        "avg_search_time_ms": avg_search_time,
        "avg_rank_when_found": avg_rank,
        "avg_score_when_found": avg_score,
        "total_tests": len(results),
        "successful_tests": success_count
    }


def main():
    """Main function for the script."""
    # Define command-line arguments
    parser = argparse.ArgumentParser(description="Benchmark semantic image search system.")
    parser.add_argument("--test-dir", type=str, required=True, help="Directory containing test images")
    parser.add_argument("--output-dir", type=str, default="benchmark_results", help="Directory to save benchmark results")
    parser.add_argument("--collection-name", type=str, default="images", help="Name of the collection to use")
    parser.add_argument("--num-samples", type=int, default=5, help="Number of partial samples per image")
    parser.add_argument("--full-matching", action="store_true", help="Benchmark full image matching")
    parser.add_argument("--partial-matching", action="store_true", help="Benchmark partial image matching")
    
    args = parser.parse_args()
    
    # Default to both if none specified
    if not args.full_matching and not args.partial_matching:
        args.full_matching = True
        args.partial_matching = True
    
    # Process arguments
    test_dir = Path(args.test_dir)
    if not test_dir.exists() or not test_dir.is_dir():
        logger.error(f"Test directory does not exist: {test_dir}")
        sys.exit(1)
        
    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Get test images
    test_images = []
    for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']:
        test_images.extend(list(test_dir.glob(f"*{ext}")))
    
    logger.info(f"Found {len(test_images)} test images in {test_dir}")
    
    # Initialize matchers
    full_matcher = FullImageMatcher(collection_name=f"full_{args.collection_name}")
    partial_matcher = PartialImageMatcher(collection_name=f"partial_{args.collection_name}")
    
    # Run benchmarks
    results = {}
    
    if args.full_matching:
        logger.info("Benchmarking full image matching...")
        full_results = benchmark_full_matching(
            test_images=test_images,
            matcher=full_matcher,
            output_dir=output_dir / "full_matching"
        )
        results["full_matching"] = full_results
        logger.info(f"Full matching success rate: {full_results['success_rate']:.2%}")
    
    if args.partial_matching:
        logger.info("Benchmarking partial image matching...")
        partial_results = benchmark_partial_matching(
            original_images=test_images,
            matcher=partial_matcher,
            output_dir=output_dir / "partial_matching",
            num_samples_per_image=args.num_samples
        )
        results["partial_matching"] = partial_results
        logger.info(f"Partial matching success rate: {partial_results['success_rate']:.2%}")
    
    # Save overall results
    with open(output_dir / "benchmark_summary.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Benchmark complete. Results saved to {output_dir}")


if __name__ == "__main__":
    main() 