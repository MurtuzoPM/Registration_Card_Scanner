import cv2
import numpy as np
from PIL import Image
import os
import logging

logger = logging.getLogger(__name__)


def validate_image(file_path):
    """Validate that a file exists and is a readable image."""
    if not os.path.exists(file_path):
        return False, 'File not found'

    try:
        with Image.open(file_path) as img:
            img.verify()
        return True, 'Valid image'
    except Exception as e:
        return False, f'Invalid image: {str(e)}'


def _auto_orient_image(pil_img):
    """Auto-orient image based on EXIF data."""
    try:
        from PIL import ExifTags, ImageOps
        pil_img = ImageOps.exif_transpose(pil_img)
    except Exception:
        pass
    return pil_img


def _deskew_image(gray):
    """Deskew a grayscale image by detecting the dominant angle."""
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100,
                            minLineLength=100, maxLineGap=10)
    if lines is None:
        return gray, 0

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(angle) < 15:
            angles.append(angle)

    if not angles:
        return gray, 0

    median_angle = np.median(angles)

    if abs(median_angle) < 0.5:
        return gray, 0

    h, w = gray.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated = cv2.warpAffine(gray, M, (w, h),
                             flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)
    return rotated, median_angle


def preprocess_image(file_path, enable_preprocessing=True):
    """
    Preprocess image for optimal OCR with MINIMAL processing.

    For registration cards, heavy processing (shadow removal, binary thresholding,
    heavy sharpening) tends to destroy handwritten and faint text. Instead, we:
      1. Load and auto-orient via EXIF
      2. Scale to consistent 2560px width
      3. Convert to grayscale
      4. Deskew if rotated
      5. Light CLAHE contrast enhancement
      6. Very gentle denoise
      7. Convert back to RGB for EasyOCR

    Returns the image as a NumPy array (RGB).
    """
    try:
        pil_image = Image.open(file_path)
        pil_image = _auto_orient_image(pil_image)
        pil_image = pil_image.convert('RGB')
        img_np = np.array(pil_image)
    except Exception:
        img_np = cv2.imread(file_path)
        if img_np is None:
            raise ValueError(f'Could not read image: {file_path}')
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

    if not enable_preprocessing:
        return img_np

    # Step 1: Scale to consistent 2560px width
    h, w = img_np.shape[:2]
    TARGET_WIDTH = 2560
    if w != TARGET_WIDTH:
        scale = TARGET_WIDTH / w
        interp = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
        img_np = cv2.resize(img_np, None, fx=scale, fy=scale,
                            interpolation=interp)
        logger.info('Resized image from %dx%d to %dx%d', w, h,
                     img_np.shape[1], img_np.shape[0])

    # Step 2: Convert to grayscale.  Red ink on pink card stock is often too
    # weak in ordinary luminance, so fold in the green/blue channels where red
    # writing appears darker.
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    red_ink_gray = cv2.min(img_np[:, :, 1], img_np[:, :, 2])
    gray = cv2.min(gray, red_ink_gray)

    # Step 3: Deskew
    gray, skew_angle = _deskew_image(gray)
    if abs(skew_angle) > 0.5:
        logger.info('Deskewed image by %.2f degrees', skew_angle)

    # Step 4: CLAHE contrast enhancement (gentle)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Step 5: Very gentle denoise (preserves text detail)
    gray = cv2.fastNlMeansDenoising(gray, h=5, templateWindowSize=7,
                                    searchWindowSize=21)

    # Convert to 3-channel RGB for EasyOCR
    result = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    return result
