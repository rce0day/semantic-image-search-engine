import os
import sys
import time
import uuid
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional

current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(current_dir, 'semantic_img'))

from flask import Flask, request, jsonify, send_from_directory, render_template
from werkzeug.utils import secure_filename
import PIL.Image

from semantic_img.matching.full_image import FullImageMatcher
from semantic_img.matching.partial_image import PartialImageMatcher
from semantic_img.preprocessing.image_processor import ImageProcessor
from semantic_img import config

app = Flask(__name__, 
            static_folder='frontend/static',
            template_folder='frontend/templates')

UPLOAD_FOLDER = Path(tempfile.mkdtemp())
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

config.FULL_MATCH_THRESHOLD = 0.05
config.PARTIAL_MATCH_THRESHOLD = 0.05

processor = ImageProcessor()
full_matcher = FullImageMatcher(collection_name="full_cats")
partial_matcher = PartialImageMatcher(collection_name="partial_cats")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_uploaded_file(file):
    filename = secure_filename(file.filename)
    temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(temp_path)
    return temp_path

def get_valid_uuid():
    return str(uuid.uuid4())

async def download_image(url):
    import requests
    from io import BytesIO
    
    response = requests.get(url, stream=True, timeout=10)
    response.raise_for_status()
    
    suffix = os.path.splitext(url.split('/')[-1])[1]
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    
    with open(temp_file.name, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    return temp_file.name

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/status')
def status():
    try:
        full_count = full_matcher.count()
        partial_count = partial_matcher.count()
        
        return jsonify({
            "status": "ok",
            "indexed_images": {
                "full_images": full_count,
                "partial_images": partial_count,
            }
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/image/<image_id>')
def serve_image(image_id):
    try:
        vector, metadata = full_matcher.index.get_vector(
            collection_name=full_matcher.collection_name,
            id=image_id
        )
        
        if metadata is None:
            vector, metadata = partial_matcher.index.get_vector(
                collection_name=partial_matcher.collection_name,
                id=image_id
            )
        
        if metadata is None:
            return jsonify({"error": "Image not found"}), 404
        
        image_path = metadata.get("original_path") or metadata.get("image_path")
        if not image_path or not os.path.exists(image_path):
            return jsonify({"error": "Image file not found"}), 404
        
        directory = os.path.dirname(image_path)
        filename = os.path.basename(image_path)
        return send_from_directory(directory, filename)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/static/images/<path:filename>')
def serve_static_image(filename):
    try:
        filename = filename.replace('\\', '/')
        image_path = os.path.join('images', filename)
        image_path = os.path.normpath(image_path)
        
        print(f"DEBUG: Serving image request for: {filename}")
        print(f"DEBUG: Full path: {image_path}")
        print(f"DEBUG: Path exists: {os.path.exists(image_path)}")
        
        if not os.path.exists(image_path):
            return jsonify({"error": f"Image not found: {image_path}"}), 404
        
        return send_from_directory(os.path.dirname(image_path), os.path.basename(image_path))
        
    except Exception as e:
        print(f"DEBUG: Error serving image: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/images/<path:filename>')
def serve_images_direct(filename):
    try:
        filename = filename.replace('\\', '/')
        image_path = os.path.join('images', filename)
        image_path = os.path.normpath(image_path)
        
        print(f"DEBUG: Direct image request for: {filename}")
        print(f"DEBUG: Full path: {image_path}")
        
        if not os.path.exists(image_path):
            return jsonify({"error": f"Image not found: {image_path}"}), 404
        
        return send_from_directory(os.path.dirname(image_path), os.path.basename(image_path))
        
    except Exception as e:
        print(f"DEBUG: Error in direct image serve: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/search/by-image', methods=['POST'])
def search_by_image():
    start_time = time.time()
    
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    use_full_matching = request.form.get('use_full_matching', 'true').lower() == 'true'
    use_partial_matching = request.form.get('use_partial_matching', 'true').lower() == 'true'
    remove_text = request.form.get('remove_text', 'false').lower() == 'true'
    enhance_image = request.form.get('enhance_image', 'false').lower() == 'true'
    limit = int(request.form.get('limit', 10))
    
    print(f"DEBUG: Search parameters - full_matching: {use_full_matching}, partial_matching: {use_partial_matching}")
    print(f"DEBUG: Remove text: {remove_text}, enhance: {enhance_image}, limit: {limit}")
    
    if file and allowed_file(file.filename):
        try:
            temp_path = save_uploaded_file(file)
            print(f"DEBUG: Temp file saved at {temp_path}")
            
            processed_img = processor.process_image(
                temp_path,
                resize=True,
                remove_text=remove_text,
                enhance=enhance_image
            )
            print(f"DEBUG: Image processed")
            
            results = []
            
            if use_full_matching:
                print(f"DEBUG: Performing full matching with {full_matcher.collection_name}")
                full_results = full_matcher.match(processed_img, limit=limit)
                print(f"DEBUG: Full matching returned {len(full_results)} results")
                for result in full_results:
                    results.append({
                        "id": result["id"],
                        "score": result["score"],
                        "metadata": result["metadata"]
                    })
            
            if use_partial_matching:
                print(f"DEBUG: Performing partial matching with {partial_matcher.collection_name}")
                partial_results = partial_matcher.match(processed_img, limit=limit)
                print(f"DEBUG: Partial matching returned {len(partial_results)} results")
                for result in partial_results:
                    results.append({
                        "id": result["id"],
                        "score": result["score"],
                        "metadata": result["metadata"],
                        "matched_regions": result.get("matched_regions"),
                        "avg_score": result.get("avg_score")
                    })
            
            unique_results = {}
            for result in results:
                if result["id"] not in unique_results or result["score"] > unique_results[result["id"]]["score"]:
                    unique_results[result["id"]] = result
            
            final_results = sorted(
                list(unique_results.values()),
                key=lambda x: x["score"],
                reverse=True
            )[:limit]
            
            print(f"DEBUG: Final results count: {len(final_results)}")

            try:
                os.unlink(temp_path)
            except:
                pass
                
            return jsonify({
                "results": final_results,
                "processing_time_ms": (time.time() - start_time) * 1000
            })
            
        except Exception as e:
            print(f"DEBUG: Error in search: {e}")
            return jsonify({"error": str(e)}), 500
    
    return jsonify({"error": "File type not allowed"}), 400

@app.route('/api/search/by-url', methods=['POST'])
def search_by_url():
    import asyncio
    
    start_time = time.time()
    data = request.json
    
    if not data or 'url' not in data:
        return jsonify({"error": "No URL provided"}), 400
    
    url = data['url']
    use_full_matching = data.get('use_full_matching', True)
    use_partial_matching = data.get('use_partial_matching', True)
    remove_text = data.get('remove_text', False)
    enhance_image = data.get('enhance_image', False)
    limit = data.get('limit', 10)
    
    try:
        temp_path = asyncio.run(download_image(url))
        processed_img = processor.process_image(
            temp_path,
            resize=True,
            remove_text=remove_text,
            enhance=enhance_image
        )
        
        results = []
        
        if use_full_matching:
            full_results = full_matcher.match(processed_img, limit=limit)
            for result in full_results:
                results.append({
                    "id": result["id"],
                    "score": result["score"],
                    "metadata": result["metadata"]
                })
        
        if use_partial_matching:
            partial_results = partial_matcher.match(processed_img, limit=limit)
            for result in partial_results:
                results.append({
                    "id": result["id"],
                    "score": result["score"],
                    "metadata": result["metadata"],
                    "matched_regions": result.get("matched_regions"),
                    "avg_score": result.get("avg_score")
                })
        
        unique_results = {}
        for result in results:
            if result["id"] not in unique_results or result["score"] > unique_results[result["id"]]["score"]:
                unique_results[result["id"]] = result
        
        final_results = sorted(
            list(unique_results.values()),
            key=lambda x: x["score"],
            reverse=True
        )[:limit]
        
        try:
            os.unlink(temp_path)
        except:
            pass
            
        return jsonify({
            "results": final_results,
            "processing_time_ms": (time.time() - start_time) * 1000
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(host='0.0.0.0', port=8000, debug=True) 