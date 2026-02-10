import os
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps

from PIL import Image
import cv2

from paddleocr import PaddleOCR

# Initialize English OCR model
ocr = PaddleOCR(lang='en') 

def run_ocr(image_path, ocr_model = ocr, show_text = True):

    display(Image.open(image_path))
    result = ocr.predict(image_path)

    page = result[0]
    texts  = page['rec_texts']
    scores = page['rec_scores']
    boxes  = page['rec_polys']

    for text, score, box in zip(texts, scores, boxes):
        coords = box.astype(int).tolist()
        print(f"{text:25} | {score:.3f} | {coords}")

    img = page['doc_preprocessor_res']['output_img']
    img_plot = img.copy()

    image_path = Path(image_path)
    output_path = image_path.with_stem(image_path.stem + "_output")
    cv2.imwrite(str(output_path), img)

    for text, box in zip(texts, boxes):
        pts = np.array(box, dtype=int)
        cv2.polylines(img_plot, [pts], True, (0, 255, 0), 2)
        x, y = pts[0]
        if show_text:
            cv2.putText(img_plot, text,
                        (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 0, 0), 2)

    plt.figure(figsize=(8, 10))
    plt.imshow(cv2.cvtColor(img_plot, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("Aligned Bounding Boxes (Processed Image)")
    plt.show()