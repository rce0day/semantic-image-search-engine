import os
from pathlib import Path
import argparse

def count_images(directory, extensions):
    if not isinstance(extensions, set):
        extensions = set(extensions)
    
    image_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in extensions:
                image_files.append(os.path.join(root, file))
    
    return image_files

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count image files recursively.")
    parser.add_argument("--dir", type=str, required=True, help="Directory to scan")
    parser.add_argument("--extensions", type=str, default=".jpg,.jpeg,.png", 
                        help="Comma-separated list of file extensions")
    
    args = parser.parse_args()
    extensions = set(args.extensions.split(','))
    
    image_files = count_images(args.dir, extensions)
    
    print(f"Found {len(image_files)} image files with extensions {extensions} in {args.dir}")
    
    print("\nSample of found files:")
    for file in image_files[:10]:
        print(f"  {file}")
    
    subfolder_counts = {}
    for file in image_files:
        subfolder = os.path.dirname(file)
        subfolder_counts[subfolder] = subfolder_counts.get(subfolder, 0) + 1
    
    print("\nFiles by subfolder:")
    for subfolder, count in sorted(subfolder_counts.items()):
        print(f"  {subfolder}: {count} files") 