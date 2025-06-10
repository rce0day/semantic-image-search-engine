import os
import tempfile
import logging
import asyncio
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import shutil
import uuid
import io
import urllib.request
from urllib.error import URLError
import PIL.Image
import time
import json

import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, HttpUrl

from semantic_img.matching.full_image import FullImageMatcher
from semantic_img.matching.partial_image import PartialImageMatcher
from semantic_img.preprocessing.image_processor import ImageProcessor
from semantic_img import config


logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Semantic Image Search API",
    description="API for semantic image search with partial image matching capabilities",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMP_DIR = Path(tempfile.mkdtemp())

processor = ImageProcessor()
full_matcher = FullImageMatcher()
partial_matcher = PartialImageMatcher()


class SearchResult(BaseModel):
    id: str
    score: float
    metadata: Dict[str, Any] = {}
    matched_regions: Optional[int] = None
    avg_score: Optional[float] = None


class SearchResponse(BaseModel):
    results: List[SearchResult]
    query_type: str
    processing_time_ms: float


class ImageUrlRequest(BaseModel):
    url: HttpUrl
    limit: int = Field(default=10, ge=1, le=100)
    use_partial_matching: bool = True
    use_full_matching: bool = True
    remove_text: bool = False
    enhance_image: bool = False


class IndexImageRequest(BaseModel):
    image_id: Optional[str] = None
    metadata: Dict[str, Any] = {}
    use_partial_matching: bool = True
    use_full_matching: bool = True


class IndexResponse(BaseModel):
    image_id: str
    success: bool
    processing_time_ms: float


class BatchIndexResponse(BaseModel):
    image_ids: List[str]
    success_count: int
    fail_count: int
    processing_time_ms: float


async def download_image(url: str) -> Path:
    try:
        temp_file = TEMP_DIR / f"{uuid.uuid4()}.jpg"
        
        with urllib.request.urlopen(url, timeout=10) as response:
            with open(temp_file, 'wb') as f:
                shutil.copyfileobj(response, f)
        
        return temp_file
    except (URLError, IOError) as e:
        logger.error(f"Failed to download image from {url}: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to download image: {e}")


async def save_upload_file(upload_file: UploadFile) -> Path:
    try:
        temp_file = TEMP_DIR / f"{uuid.uuid4()}.jpg"
        
        with open(temp_file, 'wb') as f:
            shutil.copyfileobj(upload_file.file, f)
        
        return temp_file
    except Exception as e:
        logger.error(f"Failed to save uploaded file: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to process uploaded file: {e}")


async def cleanup_temp_file(file_path: Path):
    try:
        if file_path.exists():
            os.remove(file_path)
    except Exception as e:
        logger.error(f"Failed to clean up temp file {file_path}: {e}")


@app.get("/")
async def root():
    return {"status": "ok", "version": "1.0.0"}


@app.post("/search/by-image", response_model=SearchResponse)
async def search_by_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    limit: int = Query(10, ge=1, le=100),
    use_partial_matching: bool = Query(True),
    use_full_matching: bool = Query(True),
    remove_text: bool = Query(False),
    enhance_image: bool = Query(False)
):
    start_time = time.time()
    
    try:
        temp_file = await save_upload_file(file)
        background_tasks.add_task(cleanup_temp_file, temp_file)
        
        processed_img = processor.process_image(
            temp_file,
            resize=True,
            remove_text=remove_text,
            enhance=enhance_image
        )
        
        results = []
        
        if use_full_matching:
            full_results = full_matcher.match(processed_img, limit=limit)
            for result in full_results:
                results.append(SearchResult(
                    id=result["id"],
                    score=result["score"],
                    metadata=result["metadata"]
                ))
        
        if use_partial_matching:
            partial_results = partial_matcher.match(processed_img, limit=limit)
            for result in partial_results:
                results.append(SearchResult(
                    id=result["id"],
                    score=result["score"],
                    metadata=result["metadata"],
                    matched_regions=result.get("matched_regions"),
                    avg_score=result.get("avg_score")
                ))
        
        unique_results = {}
        for result in results:
            if result.id not in unique_results or result.score > unique_results[result.id].score:
                unique_results[result.id] = result
        
        final_results = sorted(
            list(unique_results.values()),
            key=lambda x: x.score,
            reverse=True
        )[:limit]
        
        query_type = []
        if use_full_matching:
            query_type.append("full")
        if use_partial_matching:
            query_type.append("partial")
        
        return SearchResponse(
            results=final_results,
            query_type="+".join(query_type),
            processing_time_ms=(time.time() - start_time) * 1000
        )
        
    except Exception as e:
        logger.error(f"Error in search_by_image: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search/by-url", response_model=SearchResponse)
async def search_by_url(
    background_tasks: BackgroundTasks,
    request: ImageUrlRequest
):
    start_time = time.time()
    
    try:
        temp_file = await download_image(str(request.url))
        background_tasks.add_task(cleanup_temp_file, temp_file)
        
        processed_img = processor.process_image(
            temp_file,
            resize=True,
            remove_text=request.remove_text,
            enhance=request.enhance_image
        )
        
        results = []
        
        if request.use_full_matching:
            full_results = full_matcher.match(processed_img, limit=request.limit)
            for result in full_results:
                results.append(SearchResult(
                    id=result["id"],
                    score=result["score"],
                    metadata=result["metadata"]
                ))
        
        if request.use_partial_matching:
            partial_results = partial_matcher.match(processed_img, limit=request.limit)
            for result in partial_results:
                results.append(SearchResult(
                    id=result["id"],
                    score=result["score"],
                    metadata=result["metadata"],
                    matched_regions=result.get("matched_regions"),
                    avg_score=result.get("avg_score")
                ))
        
        unique_results = {}
        for result in results:
            if result.id not in unique_results or result.score > unique_results[result.id].score:
                unique_results[result.id] = result
        
        final_results = sorted(
            list(unique_results.values()),
            key=lambda x: x.score,
            reverse=True
        )[:request.limit]
        
        query_type = []
        if request.use_full_matching:
            query_type.append("full")
        if request.use_partial_matching:
            query_type.append("partial")
        
        return SearchResponse(
            results=final_results,
            query_type="+".join(query_type),
            processing_time_ms=(time.time() - start_time) * 1000
        )
        
    except Exception as e:
        logger.error(f"Error in search_by_url: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/index/image", response_model=IndexResponse)
async def index_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    request: str = Form(...),
):
    start_time = time.time()
    
    try:
        request_data = json.loads(request)
        index_request = IndexImageRequest(**request_data)
        temp_file = await save_upload_file(file)
        processed_img = processor.process_image(temp_file, resize=True)
        image_id = index_request.image_id or str(uuid.uuid4())
        
        if index_request.use_full_matching:
            full_matcher.index_image(
                image_path=temp_file,
                image_id=image_id,
                metadata=index_request.metadata
            )
        
        if index_request.use_partial_matching:
            partial_matcher.index_image(
                image_path=temp_file,
                image_id=image_id,
                metadata=index_request.metadata
            )
        
        if 'original_path' not in index_request.metadata:
            data_dir = config.DATA_DIR / "images"
            os.makedirs(data_dir, exist_ok=True)
            
            stored_path = data_dir / f"{image_id}{os.path.splitext(file.filename)[1]}"
            shutil.copy(temp_file, stored_path)
            
        background_tasks.add_task(cleanup_temp_file, temp_file)
        
        return IndexResponse(
            image_id=image_id,
            success=True,
            processing_time_ms=(time.time() - start_time) * 1000
        )
        
    except Exception as e:
        logger.error(f"Error in index_image: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status")
async def get_status():
    try:
        full_count = full_matcher.count()
        partial_count = partial_matcher.count()
        full_collection_info = full_matcher.index.collection_info(full_matcher.collection_name)
        partial_collection_info = partial_matcher.index.collection_info(partial_matcher.collection_name)
        
        return {
            "status": "ok",
            "indexed_images": {
                "full_images": full_count,
                "partial_images": partial_count,
            },
            "collections": {
                "full_images": full_collection_info,
                "partial_images": partial_collection_info,
            },
            "config": {
                "embedding_model": config.EMBEDDING_MODEL,
                "embedding_dimension": config.EMBEDDING_DIMENSION,
                "sliding_window_sizes": config.SLIDING_WINDOW_SIZES,
                "sliding_window_stride": config.SLIDING_WINDOW_STRIDE,
            }
        }
    except Exception as e:
        logger.error(f"Error in get_status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/index/{image_id}")
async def delete_image(image_id: str):
    try:
        full_deleted = full_matcher.delete_image(image_id)
        partial_deleted = partial_matcher.delete_image(image_id)
        
        if not (full_deleted or partial_deleted):
            raise HTTPException(status_code=404, detail=f"Image with ID {image_id} not found")
        
        data_dir = config.DATA_DIR / "images"
        for ext in ['.jpg', '.jpeg', '.png']:
            image_path = data_dir / f"{image_id}{ext}"
            if image_path.exists():
                os.remove(image_path)
                break
        
        return {"success": True, "image_id": image_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in delete_image: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def run_server():
    uvicorn.run(
        "semantic_img.api.server:app",
        host=config.API_HOST,
        port=config.API_PORT,
        workers=config.API_WORKERS,
        log_level=config.LOG_LEVEL.lower()
    )


if __name__ == "__main__":
    run_server() 