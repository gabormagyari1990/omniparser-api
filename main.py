from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import io
from PIL import Image
import base64
from typing import Optional
import json
import logging
from pydantic import BaseModel, Field

from services.omniparser_service import OmniParserService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="OmniParser API",
    description="API for parsing screenshots and annotating them with useful information",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OmniParser service
try:
    omniparser_service = OmniParserService()
    logger.info("OmniParser service initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize OmniParser service: {str(e)}")
    raise

class ParseResponse(BaseModel):
    annotated_image: str = Field(..., description="Base64 encoded annotated image")
    parsed_content: list = Field(..., description="List of parsed text content")
    label_coordinates: dict = Field(..., description="Dictionary of label coordinates")
    raw_text: list = Field(..., description="List of raw text extracted from image")

@app.get("/")
async def root():
    """
    Root endpoint that returns API information
    """
    return {
        "message": "Welcome to OmniParser API",
        "version": "1.0.0",
        "docs_url": "/docs"
    }

@app.post("/parse", response_model=ParseResponse)
async def parse_image(
    file: UploadFile = File(...),
    box_threshold: Optional[float] = Field(0.05, ge=0, le=1, description="Threshold for box detection"),
    iou_threshold: Optional[float] = Field(0.1, ge=0, le=1, description="IOU threshold for box merging"),
    use_paddleocr: Optional[bool] = Field(True, description="Whether to use PaddleOCR for text detection")
):
    """
    Parse an uploaded image and return annotated results
    
    Parameters:
    - file: Image file to be processed
    - box_threshold: Threshold for box detection (default: 0.05)
    - iou_threshold: IOU threshold for box merging (default: 0.1)
    - use_paddleocr: Whether to use PaddleOCR for text detection (default: True)
    
    Returns:
    - annotated_image: Base64 encoded annotated image
    - parsed_content: List of parsed text content
    - label_coordinates: Dictionary of label coordinates
    - raw_text: List of raw text extracted from image
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and process the image
        contents = await file.read()
        try:
            image = Image.open(io.BytesIO(contents))
        except Exception as e:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Process the image using OmniParser
        result = omniparser_service.process_image(
            image,
            box_threshold=box_threshold,
            iou_threshold=iou_threshold,
            use_paddleocr=use_paddleocr
        )
        
        return JSONResponse(content=result)
        
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
