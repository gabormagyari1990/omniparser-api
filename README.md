# OmniParser API

A FastAPI-based REST API for the OmniParser package that parses screenshots and annotates them with useful information.

## Features

- Image parsing and annotation
- Bounding box detection
- Text recognition
- Swagger documentation

## Installation

1. Clone the repository:

```bash
git clone https://github.com/gabormagyari1990/omniparser-api.git
cd omniparser-api
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Download model weights:
   Download the required model weights from https://huggingface.co/microsoft/OmniParser and place them in the `weights` directory.

4. Run the API:

```bash
uvicorn main:app --reload
```

## API Documentation

Once the server is running, visit:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API Endpoints

### POST /parse

Upload and process an image to get annotations and parsed content.

Parameters:

- file: Image file (multipart/form-data)
- box_threshold: Float (default: 0.05)
- iou_threshold: Float (default: 0.1)
- use_paddleocr: Boolean (default: True)

Returns:

- annotated_image: Base64 encoded annotated image
- parsed_content: List of parsed text content
- label_coordinates: Dictionary of label
