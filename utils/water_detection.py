"""
HSV-based water detection helper untuk annotation
"""
import cv2
import numpy as np


def detect_water_hsv(image, sensitivity='medium'):
    """
    Deteksi area air/banjir menggunakan HSV color space
    
    Args:
        image: Input image (numpy array, RGB)
        sensitivity: 'low', 'medium', 'high'
    
    Returns:
        Binary mask (0 dan 255)
    """
    # Convert RGB to BGR for OpenCV
    if len(image.shape) == 3 and image.shape[2] == 3:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        image_bgr = image
    
    # Convert to HSV
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    
    # Define HSV ranges for water detection
    # Water typically has:
    # - Hue: Blue/Cyan (90-130), sometimes brown/muddy (10-30)
    # - Saturation: Variable (30-255)
    # - Value: Medium to high (50-255)
    
    if sensitivity == 'low':
        # More strict - only clear water
        lower_blue = np.array([90, 50, 50])
        upper_blue = np.array([130, 255, 255])
        masks = [cv2.inRange(hsv, lower_blue, upper_blue)]
        
    elif sensitivity == 'medium':
        # Balanced - clear water + slightly muddy
        lower_blue = np.array([85, 40, 40])
        upper_blue = np.array([135, 255, 255])
        lower_cyan = np.array([80, 30, 40])
        upper_cyan = np.array([100, 255, 255])
        
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        mask_cyan = cv2.inRange(hsv, lower_cyan, upper_cyan)
        masks = [mask_blue, mask_cyan]
        
    else:  # high
        # More permissive - includes muddy/brown water
        lower_blue = np.array([80, 30, 30])
        upper_blue = np.array([140, 255, 255])
        lower_cyan = np.array([75, 20, 30])
        upper_cyan = np.array([105, 255, 255])
        lower_brown = np.array([10, 30, 30])
        upper_brown = np.array([30, 200, 200])
        
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        mask_cyan = cv2.inRange(hsv, lower_cyan, upper_cyan)
        mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)
        masks = [mask_blue, mask_cyan, mask_brown]
    
    # Combine all masks
    combined_mask = np.zeros_like(masks[0])
    for mask in masks:
        combined_mask = cv2.bitwise_or(combined_mask, mask)
    
    # Morphological operations to clean up
    kernel = np.ones((5, 5), np.uint8)
    
    # Remove noise
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Fill holes
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Remove small regions
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = (image.shape[0] * image.shape[1]) * 0.001  # 0.1% of image
    
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            cv2.drawContours(combined_mask, [contour], -1, 0, -1)
    
    return combined_mask


def refine_mask_with_watershed(image, initial_mask):
    """
    Refine mask using watershed algorithm
    
    Args:
        image: Original image
        initial_mask: Initial mask from HSV detection
    
    Returns:
        Refined mask
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Sure background area
    sure_bg = cv2.dilate(initial_mask, np.ones((3, 3), np.uint8), iterations=3)
    
    # Sure foreground area
    dist_transform = cv2.distanceTransform(initial_mask, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.3 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    
    # Unknown region
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # Marker labelling
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    
    # Apply watershed
    if len(image.shape) == 2:
        image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        image_color = image.copy()
    
    markers = cv2.watershed(image_color, markers)
    
    # Create refined mask
    refined_mask = np.zeros_like(initial_mask)
    refined_mask[markers > 1] = 255
    
    return refined_mask


def detect_water_edge_based(image):
    """
    Deteksi air berdasarkan edge detection dan texture
    
    Args:
        image: Input image
    
    Returns:
        Binary mask
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    # Water typically has less texture/edges
    # Invert edges (water = fewer edges = darker in edge map)
    water_likelihood = cv2.bitwise_not(edges)
    
    # Blur to get regions
    water_likelihood = cv2.GaussianBlur(water_likelihood, (21, 21), 0)
    
    # Threshold
    _, mask = cv2.threshold(water_likelihood, 200, 255, cv2.THRESH_BINARY)
    
    # Clean up
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    return mask


def combine_detection_methods(image, hsv_weight=0.7, edge_weight=0.3, sensitivity='medium'):
    """
    Kombinasi HSV dan edge-based detection
    
    Args:
        image: Input image
        hsv_weight: Weight untuk HSV detection
        edge_weight: Weight untuk edge detection
        sensitivity: HSV sensitivity level
    
    Returns:
        Combined binary mask
    """
    # Get masks from both methods
    hsv_mask = detect_water_hsv(image, sensitivity)
    edge_mask = detect_water_edge_based(image)
    
    # Normalize to 0-1
    hsv_norm = hsv_mask.astype(float) / 255.0
    edge_norm = edge_mask.astype(float) / 255.0
    
    # Weighted combination
    combined = (hsv_norm * hsv_weight) + (edge_norm * edge_weight)
    
    # Threshold back to binary
    combined_mask = (combined > 0.5).astype(np.uint8) * 255
    
    # Clean up
    kernel = np.ones((5, 5), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    
    return combined_mask
