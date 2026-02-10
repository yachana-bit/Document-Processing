import os
import sys
import types
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Ugly fix until paddleocr is langchain 1.0.0 compatible
# Provide old import paths expected by paddlex:
# langchain.docstore.document -> Document
m1 = types.ModuleType("langchain.docstore.document")
m1.Document = Document
sys.modules["langchain.docstore.document"] = m1
m2 = types.ModuleType("langchain.text_splitter")
m2.RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter
sys.modules["langchain.text_splitter"] = m2

os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
from paddleocr import PaddleOCR

from langchain.tools import tool

# Initialize English OCR model
ocr = PaddleOCR(lang='en')

@tool
def paddle_ocr_read_document(image_path: str) -> List[Dict[str, Any]]:
    """
    Reads an image from the given path and returns extracted text
    with bounding boxes.

    Returns a list of dictionaries, each containing:
    - 'text': the recognized text string
    - 'bbox': bounding box coordinates [x_min, y_min, x_max, y_max]
    - 'confidence': recognition confidence score (if available)
    """
    try:
        result = ocr.predict(image_path)
        page = result[0]

        texts = page['rec_texts']
        boxes = page['dt_polys']
        scores = page.get('rec_scores', [None] * len(texts))

        extracted_items = []
        for text, box, score in zip(texts, boxes, scores):
            x_coords = [point[0] for point in box]
            y_coords = [point[1] for point in box]
            bbox = [min(x_coords), min(y_coords), max(x_coords),
                    max(y_coords)]

            item = {
                'text': text,
                'bbox': bbox,
            }
            if score is not None:
                item['confidence'] = score

            extracted_items.append(item)

        return extracted_items
    except Exception as e:
        return [{"error": f"Error reading image: {e}"}]
