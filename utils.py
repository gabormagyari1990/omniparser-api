import torch
from PIL import Image
import numpy as np
import base64
import io
import logging
from typing import Tuple, List, Dict, Any

logger = logging.getLogger(__name__)

def get_yolo_model(model_path: str) -> Any:
    """
    Load YOLO model for icon detection
    
    Args:
        model_path: Path to the YOLO model weights
        
    Returns:
        YOLO model instance
    """
    try:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
        logger.info('YOLO model loaded successfully')
        return model
    except Exception as e:
        logger.error(f"Error loading YOLO model: {str(e)}")
        raise

def get_caption_model_processor(model_name: str, model_name_or_path: str) -> Dict[str, Any]:
    """
    Load caption model and processor
    
    Args:
        model_name: Name of the model to load
        model_name_or_path: Path to the model weights
        
    Returns:
        Dictionary containing model and processor
    """
    try:
        if model_name == "florence2":
            from transformers import Florence2ForCausalLM, Florence2Processor
            processor = Florence2Processor.from_pretrained(model_name_or_path)
            model = Florence2ForCausalLM.from_pretrained(model_name_or_path)
            logger.info('Caption model loaded successfully')
            return {"model": model, "processor": processor}
        else:
            raise ValueError(f"Unsupported model name: {model_name}")
    except Exception as e:
        logger.error(f"Error loading caption model: {str(e)}")
        raise

def check_ocr_box(
    image_path: str,
    display_img: bool = False,
    output_bb_format: str = 'xyxy',
    goal_filtering: Any = None,
    easyocr_args: Dict = None,
    use_paddleocr: bool = True
) -> Tuple[Tuple[List[str], List[List[float]]], bool]:
    """
    Perform OCR on the image
    
    Args:
        image_path: Path to the image file
        display_img: Whether to display the image
        output_bb_format: Format of bounding boxes
        goal_filtering: Goal filtering options
        easyocr_args: Arguments for EasyOCR
        use_paddleocr: Whether to use PaddleOCR
        
    Returns:
        Tuple containing OCR results and filtering flag
    """
    try:
        if use_paddleocr:
            from paddleocr import PaddleOCR
            ocr = PaddleOCR(use_angle_cls=True, lang='en')
            result = ocr.ocr(image_path, cls=True)
            text = [line[1][0] for line in result[0]] if result[0] else []
            boxes = [line[0] for line in result[0]] if result[0] else []
        else:
            import easyocr
            reader = easyocr.Reader(['en'])
            result = reader.readtext(image_path)
            text = [line[1] for line in result]
            boxes = [line[0] for line in result]
        
        logger.info(f'OCR completed successfully with {len(text)} text elements')
        return (text, boxes), False
    except Exception as e:
        logger.error(f"Error performing OCR: {str(e)}")
        raise

def get_som_labeled_img(
    image_path: str,
    yolo_model: Any,
    BOX_TRESHOLD: float = 0.05,
    output_coord_in_ratio: bool = True,
    ocr_bbox: List = None,
    caption_model_processor: Dict = None,
    ocr_text: List = None,
    iou_threshold: float = 0.1
) -> Tuple[str, Dict, List]:
    """
    Get labeled image with bounding boxes and captions
    
    Args:
        image_path: Path to the image file
        yolo_model: YOLO model instance
        BOX_TRESHOLD: Threshold for box detection
        output_coord_in_ratio: Whether to output coordinates in ratio
        ocr_bbox: OCR bounding boxes
        caption_model_processor: Caption model and processor
        ocr_text: OCR text results
        iou_threshold: IOU threshold for box merging
        
    Returns:
        Tuple containing base64 image, label coordinates, and parsed content
    """
    try:
        # Load and process image
        image = Image.open(image_path)
        results = yolo_model(image)
        
        # Convert image to base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        # Process detections
        boxes = results.xyxy[0].cpu().numpy()
        boxes = boxes[boxes[:, 4] > BOX_TRESHOLD]
        
        # Get coordinates and labels
        label_coordinates = {}
        parsed_content = []
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2, conf, cls = box
            label = f"Box {i}: {conf:.2f}"
            label_coordinates[label] = [float(x1), float(y1), float(x2), float(y2)]
            parsed_content.append(f"Text Box ID {i}: {ocr_text[i] if i < len(ocr_text) else ''}")
        
        logger.info(f'Image processed successfully with {len(boxes)} detections')
        return img_str, label_coordinates, parsed_content
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise
