import sys
import os
from PIL import Image
import base64
import io
import torch
from utils import check_ocr_box, get_yolo_model, get_caption_model_processor, get_som_labeled_img
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OmniParserService:
    def __init__(self):
        """
        Initialize OmniParser service with required models and configurations
        """
        try:
            self.yolo_model = get_yolo_model(model_path='weights/icon_detect/best.pt')
            self.caption_model_processor = get_caption_model_processor(
                model_name="florence2", 
                model_name_or_path="weights/icon_caption_florence"
            )
            logger.info("Models loaded successfully")
        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")
            raise RuntimeError(f"Failed to initialize OmniParser service: {str(e)}")
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.yolo_model.to(self.device)
        
    def process_image(self, image: Image.Image, box_threshold: float = 0.05, 
                     iou_threshold: float = 0.1, use_paddleocr: bool = True):
        """
        Process an image using OmniParser
        
        Args:
            image: PIL Image to process
            box_threshold: Threshold for box detection
            iou_threshold: IOU threshold for box merging
            use_paddleocr: Whether to use PaddleOCR
            
        Returns:
            dict: Dictionary containing processed results
        """
        try:
            # Create temp directory if it doesn't exist
            os.makedirs("temp", exist_ok=True)
            
            # Save image to temporary file with unique name
            img_path = os.path.join("temp", f"temp_{os.urandom(4).hex()}.png")
            image.save(img_path)
            
            logger.info(f"Processing image saved at: {img_path}")
            
            # Process with OCR
            ocr_bbox_rslt, is_goal_filtered = check_ocr_box(
                img_path, 
                display_img=False, 
                output_bb_format='xyxy',
                goal_filtering=None,
                easyocr_args={'paragraph': False, 'text_threshold': 0.9},
                use_paddleocr=use_paddleocr
            )
            text, ocr_bbox = ocr_bbox_rslt
            
            # Get labeled image and parsed content
            dino_labled_img, label_coordinates, parsed_content_list = get_som_labeled_img(
                img_path,
                self.yolo_model,
                BOX_TRESHOLD=box_threshold,
                output_coord_in_ratio=True,
                ocr_bbox=ocr_bbox,
                caption_model_processor=self.caption_model_processor,
                ocr_text=text,
                iou_threshold=iou_threshold
            )
            
            # Clean up temporary file
            os.remove(img_path)
            
            return {
                "annotated_image": dino_labled_img,
                "parsed_content": parsed_content_list,
                "label_coordinates": label_coordinates,
                "raw_text": text
            }
            
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            if os.path.exists(img_path):
                os.remove(img_path)
            raise Exception(f"Error processing image: {str(e)}")
