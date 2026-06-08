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


def _remove_colored_background(img_rgb, dark_v_threshold=80):
    """
    Remove red background and blue seal/stamp from Tajikistan registration cards.

    Args:
        dark_v_threshold: V (brightness) threshold for preserving text pixels.
            Higher values preserve more faint handwriting but also more noise.
            Use 80 for clean images, 140 for noisy/faint images.
    """
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    mask_red = (
        cv2.inRange(hsv, np.array([0, 15, 15]), np.array([20, 255, 255])) |
        cv2.inRange(hsv, np.array([160, 15, 15]), np.array([180, 255, 255]))
    )

    mask_blue = cv2.inRange(hsv, np.array([85, 15, 15]), np.array([145, 255, 255]))

    mask_dark = cv2.inRange(hsv, np.array([0, 0, 0]),
                            np.array([180, 255, dark_v_threshold]))

    result = img_rgb.copy()
    mask_colored_nontext = (mask_red | mask_blue) & ~mask_dark
    result[mask_colored_nontext > 0] = [255, 255, 255]

    return result


def _detect_noise_level(img_rgb):
    """Estimate noise level based on colored pixel ratio."""
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    mask_red = (
        cv2.inRange(hsv, np.array([0, 30, 30]), np.array([15, 255, 255])) |
        cv2.inRange(hsv, np.array([165, 30, 30]), np.array([180, 255, 255]))
    )
    mask_blue = cv2.inRange(hsv, np.array([90, 30, 30]), np.array([140, 255, 255]))
    colored_ratio = np.sum((mask_red | mask_blue) > 0) / (img_rgb.shape[0] * img_rgb.shape[1])
    return 'noisy' if colored_ratio > 0.15 else 'clean'


def preprocess_image(file_path, enable_preprocessing=True, fast_mode=False):
    """
    Preprocess image for optimal OCR with dual-pipeline approach.

    Returns THREE preprocessed variants:
      1. Aggressive: color removal (V=140) + red-ink grayscale + CLAHE 3.0
      2. Conservative: standard grayscale + CLAHE 2.0
      3. Adaptive binary: deskewed threshold image (best for digits/dates)

    OCR runs on all variants and results are merged.
    Returns a list of NumPy arrays in RGB.
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
        return [img_np]

    h, w = img_np.shape[:2]
    TARGET_WIDTH = 2560
    if w != TARGET_WIDTH:
        scale = TARGET_WIDTH / w
        interp = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
        img_np = cv2.resize(img_np, None, fx=scale, fy=scale, interpolation=interp)
        logger.info('Resized image from %dx%d to %dx%d', w, h,
                     img_np.shape[1], img_np.shape[0])

    results = []

    # Variant 1: Aggressive — color removal + red-ink-aware grayscale
    img_v1 = _remove_colored_background(img_np, dark_v_threshold=140)
    gray_v1 = cv2.cvtColor(img_v1, cv2.COLOR_RGB2GRAY)
    red_ink = cv2.min(img_v1[:, :, 1], img_v1[:, :, 2])
    gray_v1 = cv2.min(gray_v1, red_ink)
    gray_v1, angle1 = _deskew_image(gray_v1)
    if abs(angle1) > 0.5:
        logger.info('Variant 1 deskewed by %.2f degrees', angle1)
    clahe1 = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray_v1 = clahe1.apply(gray_v1)
    if not fast_mode:
        gray_v1 = cv2.fastNlMeansDenoising(gray_v1, h=5, templateWindowSize=7,
                                            searchWindowSize=21)
    results.append(cv2.cvtColor(gray_v1, cv2.COLOR_GRAY2RGB))

    # Variant 2: Conservative — no color removal, standard pipeline
    gray_v2 = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    gray_v2, angle2 = _deskew_image(gray_v2)
    if abs(angle2) > 0.5:
        logger.info('Variant 2 deskewed by %.2f degrees', angle2)
    clahe2 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_v2 = clahe2.apply(gray_v2)
    if not fast_mode:
        gray_v2 = cv2.fastNlMeansDenoising(gray_v2, h=5, templateWindowSize=7,
                                            searchWindowSize=21)
    results.append(cv2.cvtColor(gray_v2, cv2.COLOR_GRAY2RGB))

    if not fast_mode:
        # Variant 3: Adaptive binary — strong for printed digits/dates (Tesseract-friendly)
        gray_v3 = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        gray_v3, _ = _deskew_image(gray_v3)
        gray_v3 = cv2.GaussianBlur(gray_v3, (3, 3), 0)
        binary = cv2.adaptiveThreshold(
            gray_v3, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, 12,
        )
        binary = cv2.bitwise_not(binary)
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        results.append(cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB))

    logger.info('Multi-pipeline: returning %d variants (fast_mode=%s)',
                len(results), fast_mode)
    return results
