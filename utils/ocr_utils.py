"""
OCR utilities for extracting math text from images.
Uses EasyOCR for broad language and math symbol support.
"""

import os
from utils.logger import get_logger

logger = get_logger("utils.ocr")


def extract_text_from_image(image_path: str) -> dict:
    """
    Extract text from an image using EasyOCR.

    Args:
        image_path: Path to the image file (JPG/PNG).

    Returns:
        dict with keys:
            - text (str): Extracted text joined into a single string.
            - confidence (float): Average confidence score (0–1).
            - raw_results (list): Per-box results [(bbox, text, conf), ...].
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    try:
        import easyocr

        reader = easyocr.Reader(["en"], gpu=False)
        # Use advanced parameters to help pick up small math text and symbols
        results = reader.readtext(
            image_path, 
            mag_ratio=2.0,           # Magnify image for better small text detection
            contrast_ths=0.1,        # Lower contrast threshold
            adjust_contrast=0.5,     # Aggressive contrast adjustment
            text_threshold=0.5,      # Lower confidence needed to accept text
            low_text=0.4             # Better faint text detection
        )

        if not results:
            logger.warning("OCR returned no text for %s", image_path)
            return {"text": "", "confidence": 0.0, "raw_results": []}

        texts = [r[1] for r in results]
        confidences = [r[2] for r in results]
        avg_conf = sum(confidences) / len(confidences)

        extracted = " ".join(texts)
        logger.info(
            "OCR extracted %d chars (confidence=%.2f) from %s",
            len(extracted),
            avg_conf,
            image_path,
        )

        return {
            "text": extracted,
            "confidence": round(avg_conf, 3),
            "raw_results": [
                {"text": r[1], "confidence": round(r[2], 3)} for r in results
            ],
        }

    except ImportError:
        logger.error("EasyOCR not installed. Run: pip install easyocr")
        return {
            "text": "[EasyOCR not installed]",
            "confidence": 0.0,
            "raw_results": [],
        }
    except Exception as e:
        logger.error("OCR failed: %s", str(e))
        return {"text": f"[OCR Error: {e}]", "confidence": 0.0, "raw_results": []}
