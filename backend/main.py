from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exception_handlers import http_exception_handler
import asyncio
import concurrent.futures
import uuid
import io
import os
import time
import traceback
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from edge_detection import process_image_from_memory

# Initialize FastAPI app
app = FastAPI(
    title="Edge Detection API",
    description="A scalable edge detection service using Pb-lite algorithm",
    version="1.0.0"
)

# Add request size limit middleware
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_size: int = 10 * 1024 * 1024):  # 10MB default
        super().__init__(app)
        self.max_size = max_size

    async def dispatch(self, request: Request, call_next):
        if request.method in ["POST", "PUT", "PATCH"]:
            content_length = request.headers.get("content-length")
            if content_length and int(content_length) > self.max_size:
                return Response(
                    content="Request too large",
                    status_code=413,
                    media_type="text/plain"
                )
        return await call_next(request)

app.add_middleware(RequestSizeLimitMiddleware, max_size=10 * 1024 * 1024)

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle all uncaught exceptions"""
    print(f"Global exception handler caught: {exc}")
    print(f"Traceback: {traceback.format_exc()}")
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred. Please try again."
        }
    )

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:3000",  # Common dev port
        "https://localhost:8000",
        "https://127.0.0.1:8000"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Global storage for processing jobs (in production, use Redis or a database)
processing_jobs = {}

# Thread pool for background processing
executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

# Job cleanup configuration
JOB_CLEANUP_INTERVAL = 300  # 5 minutes
JOB_MAX_AGE = 3600  # 1 hour
MAX_JOBS_IN_MEMORY = 100

# Root endpoint moved to serve_frontend() function

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "active_jobs": len([job for job in processing_jobs.values() if job["status"] == "processing"]),
        "total_jobs": len(processing_jobs)
    }

def cleanup_old_jobs():
    """Clean up old jobs to prevent memory leaks"""
    current_time = time.time()
    jobs_to_remove = []
    
    for job_id, job in processing_jobs.items():
        job_age = current_time - job.get("created_at", 0)
        
        # Remove jobs older than MAX_AGE or if we have too many jobs
        if (job_age > JOB_MAX_AGE or 
            len(processing_jobs) > MAX_JOBS_IN_MEMORY):
            jobs_to_remove.append(job_id)
    
    # Remove oldest jobs first if we're over the limit
    if len(processing_jobs) > MAX_JOBS_IN_MEMORY:
        sorted_jobs = sorted(processing_jobs.items(), 
                           key=lambda x: x[1].get("created_at", 0))
        excess_count = len(processing_jobs) - MAX_JOBS_IN_MEMORY
        jobs_to_remove.extend([job_id for job_id, _ in sorted_jobs[:excess_count]])
    
    for job_id in jobs_to_remove:
        if job_id in processing_jobs:
            del processing_jobs[job_id]
    
    return len(jobs_to_remove)

# Background task for periodic cleanup
async def periodic_cleanup():
    """Periodically clean up old jobs"""
    while True:
        await asyncio.sleep(JOB_CLEANUP_INTERVAL)
        try:
            removed_count = cleanup_old_jobs()
            if removed_count > 0:
                print(f"Cleaned up {removed_count} old jobs")
        except Exception as e:
            print(f"Error during job cleanup: {e}")

# Start cleanup task
@app.on_event("startup")
async def startup_event():
    """Start background tasks"""
    asyncio.create_task(periodic_cleanup())

def process_image_background(job_id: str, pil_image: Image.Image):
    """
    Background function to process image - runs in thread pool
    """
    try:
        # Update job status
        processing_jobs[job_id]["status"] = "processing"
        
        # Process the image using our edge detection algorithm
        results = process_image_from_memory(pil_image)
        
        # Convert the final output to a format we can return
        final_output = results['final_output']
        
        # Convert numpy array to PIL Image
        if final_output.dtype != np.uint8:
            # Ensure values are in [0,1] range, then convert to [0,255]
            final_output_uint8 = (np.clip(final_output, 0, 1) * 255).astype(np.uint8)
        else:
            final_output_uint8 = final_output
            
        # Convert to PIL Image
        if len(final_output_uint8.shape) == 2:  # Grayscale
            result_image = Image.fromarray(final_output_uint8, mode='L')
        else:  # RGB
            result_image = Image.fromarray(final_output_uint8, mode='RGB')
        
        # Save result to memory buffer
        img_buffer = io.BytesIO()
        result_image.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        
        # Update job with success
        processing_jobs[job_id].update({
            "status": "completed",
            "result_data": img_buffer.getvalue(),
            "all_results": results  # Store all intermediate results if needed
        })
        
    except Exception as e:
        # Update job with error
        processing_jobs[job_id].update({
            "status": "error",
            "error_message": str(e)
        })

@app.post("/api/upload")
async def upload_image(file: UploadFile = File(...)):
    """
    Upload an image for processing
    """
    # Configuration
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    ALLOWED_CONTENT_TYPES = {
        'image/jpeg', 'image/jpg', 'image/png', 
        'image/gif', 'image/bmp', 'image/webp'
    }
    
    # Validate file type
    if not file.content_type or file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid file type. Allowed types: {', '.join(ALLOWED_CONTENT_TYPES)}"
        )
    
    try:
        # Read the uploaded file
        contents = await file.read()
        
        # Validate file size
        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413, 
                detail=f"File too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)}MB"
            )
        
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        # Validate that it's actually an image by trying to open it
        try:
            pil_image = Image.open(io.BytesIO(contents))
            # Verify image can be loaded
            pil_image.verify()
            # Re-open since verify() closes the image
            pil_image = Image.open(io.BytesIO(contents))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")
        
        # Generate unique job ID
        job_id = str(uuid.uuid4())
        
        # Initialize job record
        processing_jobs[job_id] = {
            "status": "pending",
            "filename": file.filename,
            "created_at": time.time()
        }
        
        # Submit processing job to thread pool
        loop = asyncio.get_event_loop()
        loop.run_in_executor(executor, process_image_background, job_id, pil_image)
        
        return JSONResponse({
            "job_id": job_id,
            "status": "pending",
            "message": "Image uploaded successfully, processing started"
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing upload: {str(e)}")

@app.get("/api/status/{job_id}")
async def get_job_status(job_id: str):
    """
    Get the status of a processing job
    """
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = processing_jobs[job_id]
    
    response = {
        "job_id": job_id,
        "status": job["status"],
        "filename": job.get("filename", "")
    }
    
    if job["status"] == "completed":
        response["result_url"] = f"/api/result/{job_id}"
    elif job["status"] == "error":
        response["error"] = job.get("error_message", "Unknown error")
    
    return JSONResponse(response)

@app.get("/api/result/{job_id}")
async def get_result(job_id: str):
    """
    Get the processed image result
    """
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = processing_jobs[job_id]
    
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not completed yet")
    
    if "result_data" not in job:
        raise HTTPException(status_code=500, detail="Result data not found")
    
    # Return the processed image
    return StreamingResponse(
        io.BytesIO(job["result_data"]),
        media_type="image/png",
        headers={"Content-Disposition": "inline; filename=edge_detected.png"}
    )

# Mount static files for serving the frontend
import os
from pathlib import Path

# Get the absolute path to the frontend directory
# This works whether you run from backend/ or project root
current_file = Path(__file__).resolve()
backend_dir = current_file.parent
project_root = backend_dir.parent
frontend_dir = project_root / "frontend"

print(f"Current file: {current_file}")
print(f"Backend directory: {backend_dir}")
print(f"Project root: {project_root}")
print(f"Frontend directory: {frontend_dir}")
print(f"Frontend directory exists: {frontend_dir.exists()}")

if frontend_dir.exists():
    print(f"Frontend files: {list(frontend_dir.iterdir())}")
    
    # Serve static files (CSS, JS) from /static path
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

    # Serve the main HTML file at root
    @app.get("/")
    async def serve_frontend():
        """Serve the main frontend HTML file"""
        from fastapi.responses import FileResponse
        html_path = frontend_dir / "index.html"
        if html_path.exists():
            return FileResponse(str(html_path))
        else:
            raise HTTPException(status_code=404, detail=f"Frontend HTML not found at {html_path}")
else:
    print("WARNING: Frontend directory not found!")
    
    @app.get("/")
    async def serve_frontend_error():
        """Show error when frontend is not found"""
        return JSONResponse({
            "error": "Frontend not found",
            "frontend_path": str(frontend_dir),
            "current_working_directory": os.getcwd(),
            "instructions": "Make sure to run the server from the project root or backend directory"
        })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)