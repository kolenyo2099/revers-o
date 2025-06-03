import requests
from PIL import Image
from io import BytesIO
import numpy as np
import torch

def download_image(url):
    """Download an image from a URL"""
    response = requests.get(url)
    return Image.open(BytesIO(response.content)).convert("RGB")

def load_local_image(path):
    """Load an image from local storage"""
    return Image.open(path).convert("RGB")

def tokens_to_grid(tokens, model): # model parameter is not used, but kept for compatibility
    """Convert tokens to spatial grid format"""
    B, N_plus_1, D = tokens.shape
    patch_tokens = tokens[:, 1:, :]  # remove CLS
    N = patch_tokens.shape[1]

    # Smallest square that can hold N
    side = int(np.ceil(np.sqrt(N)))
    grid = side * side

    if N < grid:  # pad missing tokens
        pad_len = grid - N
        pad = torch.zeros(B, pad_len, D,
                          dtype=patch_tokens.dtype,
                          device=patch_tokens.device)
        grid_tokens = torch.cat([patch_tokens, pad], dim=1)
    else:  # drop any extra (globals)
        grid_tokens = patch_tokens[:, :grid, :]

    return grid_tokens.reshape(B, side, side, D)  # (B, H, W, D)

def get_detection_property(detections, index, property_name, default=None):
    """
    Safely get a property from a detection object, handling both tuple-style and attribute-style access.

    Args:
        detections: The detections object from GroundedSAM
        index: The index of the detection to access
        property_name: The property name to access (e.g., 'mask', 'confidence', 'class_id')
        default: Default value to return if property can't be accessed

    Returns:
        The property value or default if not found
    """
    try:
        # First try attribute-style access (supervision.detection.Detections object)
        if hasattr(detections, property_name):
            prop = getattr(detections, property_name)
            if prop is not None and index < len(prop):
                return prop[index]

        # Try direct indexing if it's a list/tuple of objects
        elif isinstance(detections, (list, tuple)) and index < len(detections):
            # If it's an object with the attribute
            if hasattr(detections[index], property_name):
                return getattr(detections[index], property_name)
            # If it's a dict with the key
            elif isinstance(detections[index], dict) and property_name in detections[index]:
                return detections[index][property_name]

        # Try accessing specific properties in alternate ways (for different versions)
        if property_name == 'mask' and hasattr(detections, 'masks') and index < len(detections.masks):
            return detections.masks[index]

        if property_name == 'class_id' and hasattr(detections, 'class_ids') and index < len(detections.class_ids):
            return detections.class_ids[index]

    except Exception as e:
        print(f"[DEBUG] Error accessing {property_name} at index {index}: {e}")

    return default
