#!/usr/bin/env python3
"""
Grounded SAM Region Search Application
Main application file containing all core functionality
"""

import argparse
import os
import sys
import time
import gc
import json
import uuid
import math
import io
from contextlib import nullcontext
from pathlib import Path
from typing import List, Optional

# Core imports
import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib import gridspec
from PIL import Image
import requests
from io import BytesIO
from tqdm import tqdm
import gradio as gr

# ML/CV imports
from autodistill_grounded_sam import GroundedSAM
from autodistill.detection import CaptionOntology
from qdrant_client import QdrantClient
from qdrant_client.http import models

# Add perception_models to path
sys.path.append('./perception_models')
import core.vision_encoder.pe as pe
import core.vision_encoder.transforms as transforms

# Global variables for models and device
pe_model = None
pe_vit_model = None
preprocess = None
device = None

# =============================================================================
# APPLICATION STATE MANAGEMENT
# =============================================================================

class AppState:
    """Manages the application state for the multi-mode interface"""
    def __init__(self):
        self.mode = "landing"  # "landing", "building", "searching"
        self.available_databases = []
        self.active_database = None
        print(f"[STATUS] Initializing App State")
        self.update_available_databases(verbose=False)
    
    def update_available_databases(self, verbose=True):
        """Update the list of available databases"""
        if verbose:
            print(f"[STATUS] Refreshing database list")
        
        self.available_databases = list_available_databases()
        
        if verbose:
            print(f"[STATUS] Found {len(self.available_databases)} databases")
            
        return self.available_databases

def list_available_databases():
    """Find all existing Qdrant collections in the project directory"""
    # Check both possible directories - older Qdrant uses 'collection', newer might use 'collections'
    collections = []
    
    # Path 1: Check the 'collection' directory (singular)
    collection_dir = os.path.join("image_retrieval_project", "qdrant_data", "collection")
    if os.path.exists(collection_dir):
        collections.extend([d for d in os.listdir(collection_dir) if os.path.isdir(os.path.join(collection_dir, d))])
        print(f"[DEBUG] Found collections in 'collection' directory: {collections}")
    
    # Path 2: Check the 'collections' directory (plural)
    collections_dir = os.path.join("image_retrieval_project", "qdrant_data", "collections")
    if os.path.exists(collections_dir):
        plural_collections = [d for d in os.listdir(collections_dir) if os.path.isdir(os.path.join(collections_dir, d))]
        collections.extend(plural_collections)
        print(f"[DEBUG] Found collections in 'collections' directory: {plural_collections}")
    
    # Path 3: Direct lookup in qdrant_data if neither subfolder exists
    if not collections:
        qdrant_data_dir = os.path.join("image_retrieval_project", "qdrant_data")
        if os.path.exists(qdrant_data_dir):
            try:
                # Try to connect to Qdrant and get collections directly
                client = QdrantClient(path=qdrant_data_dir)
                collection_list = client.get_collections().collections
                collections = [c.name for c in collection_list]
                client.close()
                print(f"[DEBUG] Found collections via Qdrant API: {collections}")
            except Exception as e:
                print(f"[DEBUG] Error querying Qdrant API: {e}")
    
    # Remove duplicates while preserving order
    unique_collections = []
    for c in collections:
        if c not in unique_collections:
            unique_collections.append(c)
    
    if not unique_collections:
        print(f"[STATUS] No database collections found")
    else:
        print(f"[STATUS] Available collections: {unique_collections}")
    
    return unique_collections

# =============================================================================
# DEVICE SETUP AND MODEL LOADING
# =============================================================================

def setup_device():
    """Setup the appropriate compute device"""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Using MPS device: {device}")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {device}")
    else:
        device = torch.device("cpu")
        print(f"Using CPU device: {device}")
    return device

def load_pe_model(device):
    """Load Perception Encoder model"""
    print("Available PE configurations:", pe.CLIP.available_configs())

    # Try different models in order of preference
    model_name = 'PE-Core-G14-448' 
    if model_name in pe.CLIP.available_configs():
        pe_model = pe.CLIP.from_config(model_name, pretrained=True)
    elif "PE-Spatial-L14-336" in pe.CLIP.available_configs():
        pe_model = pe.CLIP.from_config("PE-Core-G14-448", pretrained=True)
    else:
        available_configs = pe.CLIP.available_configs()
        if available_configs:
            pe_model = pe.CLIP.from_config(available_configs[0], pretrained=True)
        else:
            raise Exception("No PE model configurations available")

    # For Perception Encoder, we should prefer VisionTransformer over CLIP
    # when we need intermediate layer access
    if hasattr(pe, 'VisionTransformer') and pe.VisionTransformer.available_configs():
        print("Available PE VisionTransformer configs:", pe.VisionTransformer.available_configs())
        vit_configs = pe.VisionTransformer.available_configs()
        # Prefer PE-Spatial variants for region work
        spatial_configs = [c for c in vit_configs if 'Spatial' in c]
        if spatial_configs:
            vit_model = pe.VisionTransformer.from_config(spatial_configs[0], pretrained=True)
            print(f"Loaded Vision Transformer model: {spatial_configs[0]} for intermediate layer access")
            vit_model = vit_model.to(device)
            # Keep both models - one for feature extraction, one for final embedding if needed
            preprocess = transforms.get_image_transform(pe_model.image_size)
            return pe_model, vit_model, preprocess
    
    pe_model = pe_model.to(device)
    preprocess = transforms.get_image_transform(pe_model.image_size)
    return pe_model, None, preprocess

# =============================================================================
# IMAGE PROCESSING FUNCTIONS
# =============================================================================

def download_image(url):
    """Download an image from a URL"""
    response = requests.get(url)
    return Image.open(BytesIO(response.content)).convert("RGB")

def load_local_image(path):
    """Load an image from local storage"""
    return Image.open(path).convert("RGB")

def tokens_to_grid(tokens, model):
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
            
        # Try one more way - if detections has a specific index method
        if hasattr(detections, '__getitem__'):
            try:
                detection = detections[index]
                if hasattr(detection, property_name):
                    return getattr(detection, property_name)
                elif isinstance(detection, dict) and property_name in detection:
                    return detection[property_name]
            except:
                pass
                
        # If all else fails, return the default
        return default
        
    except Exception as e:
        print(f"[STATUS] Error accessing {property_name} at index {index}: {e}")
        return default

def extract_region_embeddings_autodistill(
    image_source,
    text_prompt,
    pe_model_param=None,
    pe_vit_model_param=None,
    preprocess_param=None,
    device_param=None,
    min_area_ratio=0.01,
    max_regions=10,
    is_url=False,
    max_image_size=800,
    optimal_layer=40
):
    """
    Extract region embeddings using Autodistill GroundedSAM and Perception Encoder
    
    This function is compatible with both the notebook version (which doesn't require model parameters)
    and the main.py version (which passes model parameters explicitly)
    """
    # Access global models if needed
    global pe_model, pe_vit_model, preprocess, device
    
    # Use provided models or fall back to globals
    pe_model_to_use = pe_model_param if pe_model_param is not None else pe_model
    pe_vit_model_to_use = pe_vit_model_param if pe_vit_model_param is not None else pe_vit_model
    preprocess_to_use = preprocess_param if preprocess_param is not None else preprocess
    device_to_use = device_param if device_param is not None else device
    
    try:
        print(f"[STATUS] Starting region extraction process...")
        # Load the image
        if is_url:
            print(f"[STATUS] Downloading image from URL...")
            pil_image = download_image(image_source)
        else:
            if isinstance(image_source, str):
                print(f"[STATUS] Loading image from path: {os.path.basename(image_source)}")
                pil_image = load_local_image(image_source)
            else:
                print(f"[STATUS] Processing uploaded image...")
                pil_image = image_source

        # Resize image to control memory usage
        width, height = pil_image.size
        if width > max_image_size or height > max_image_size:
            if width > height:
                new_width = max_image_size
                new_height = int(height * (max_image_size / width))
            else:
                new_height = max_image_size
                new_width = int(width * (max_image_size / height))

            print(f"[STATUS] Resizing image from {width}x{height} to {new_width}x{new_height}")
            pil_image = pil_image.resize((new_width, new_height), Image.LANCZOS)

        # Convert to numpy array for processing
        image = np.array(pil_image)
        image_area = image.shape[0] * image.shape[1]
        print(f"[STATUS] Image converted to array, shape: {image.shape}")

        # Save temp image if needed for Autodistill
        temp_image_path = "/tmp/temp_image.jpg"
        pil_image.save(temp_image_path)

        # Create ontology from text prompts
        prompts = [p.strip() for p in text_prompt.split(',')]
        print(f"[STATUS] Using prompts: {prompts}")
        ontology_dict = {prompt: f"class_{i}" for i, prompt in enumerate(prompts)}

        # Initialize GroundedSAM with our ontology
        print(f"[STATUS] Initializing GroundedSAM model...")
        grounded_sam = GroundedSAM(ontology=CaptionOntology(ontology_dict))
        print(f"[STATUS] GroundedSAM initialized with ontology: {ontology_dict}")

        # Get predictions
        print(f"[STATUS] Running GroundedSAM detection on image...")
        detections = grounded_sam.predict(temp_image_path)

        # Debug the detection results
        print(f"[STATUS] Detection complete. Result type: {type(detections)}")

        # Check if detections is empty - handle both object and tuple formats
        is_empty = False
        if hasattr(detections, 'is_empty') and callable(getattr(detections, 'is_empty')):
            is_empty = detections.is_empty()
        elif isinstance(detections, (tuple, list)):
            is_empty = len(detections) == 0
        else:
            is_empty = len(detections) == 0  # Default case
            
        print(f"[STATUS] Is Detections empty: {is_empty}")

        if not is_empty:
            print(f"[STATUS] Number of detections: {len(detections)}")
            # Print additional debug info if available
            if hasattr(detections, 'confidence') and detections.confidence is not None:
                print(f"[STATUS] Detection confidence count: {len(detections.confidence)}")
            if hasattr(detections, 'class_id') and detections.class_id is not None:
                print(f"[STATUS] Detection class IDs count: {len(detections.class_id)}")
            print(f"[STATUS] Has masks: {hasattr(detections, 'mask') and detections.mask is not None}")

        # Extract masks
        masks = []
        labels = []

        # Handle detections - use direct access method like in the notebook version
        print(f"[STATUS] Processing detections...")
        
        # Handle the detections based on available attributes
        if hasattr(detections, 'mask') and detections.mask is not None and len(detections) > 0:
            print(f"[STATUS] Processing {len(detections)} detections with masks")

            for i in range(len(detections)):
                if i >= max_regions:
                    break

                # Get the mask for this detection
                mask = detections.mask[i]

                # Skip if mask is empty or all False
                if mask is None or np.all(mask == False):
                    print(f"Mask {i} is empty or all False, skipping")
                    continue

                # Calculate area ratio
                area_ratio = np.sum(mask) / image_area
                print(f"Mask {i} area ratio: {area_ratio}")

                # Skip if too small
                if area_ratio < min_area_ratio:
                    print(f"Mask {i} too small (area ratio: {area_ratio} < {min_area_ratio}), skipping")
                    continue

                # Get confidence and class ID
                confidence = detections.confidence[i] if hasattr(detections, 'confidence') and detections.confidence is not None else 0.5
                class_id = detections.class_id[i] if hasattr(detections, 'class_id') and detections.class_id is not None else "class_0"

                # Get the original prompt that generated this detection
                original_prompt = next((k for k, v in ontology_dict.items() if v == class_id), str(class_id))

                masks.append(mask)
                labels.append(f"{original_prompt}: {confidence:.2f}")
                print(f"Added mask {i} for '{original_prompt}' with confidence {confidence:.2f}")
                
        # Fallback to bounding boxes if no masks are available
        elif hasattr(detections, 'xyxy') and len(detections.xyxy) > 0:
            print(f"No usable masks found, creating masks from bounding boxes")
            H, W = image.shape[:2]

            for i in range(len(detections)):
                if i >= max_regions:
                    break

                # Get the bounding box
                x1, y1, x2, y2 = map(int, detections.xyxy[i])

                # Create a mask from the bounding box
                mask = np.zeros((H, W), dtype=bool)
                mask[max(0, y1):min(H, y2), max(0, x1):min(W, x2)] = True

                # Calculate area ratio
                area_ratio = np.sum(mask) / image_area
                print(f"Box mask {i} area ratio: {area_ratio}")

                # Skip if too small
                if area_ratio < min_area_ratio:
                    print(f"Box mask {i} too small (area ratio: {area_ratio} < {min_area_ratio}), skipping")
                    continue

                # Get confidence and class ID
                confidence = detections.confidence[i] if hasattr(detections, 'confidence') and detections.confidence is not None else 0.5
                class_id = detections.class_id[i] if hasattr(detections, 'class_id') and detections.class_id is not None else "class_0"

                # Get the original prompt
                original_prompt = next((k for k, v in ontology_dict.items() if v == class_id), str(class_id))

                masks.append(mask)
                labels.append(f"{original_prompt}: {confidence:.2f}")
                print(f"Created box mask {i} for '{original_prompt}' with confidence {confidence:.2f}")

        # No valid masks found
        if len(masks) == 0:
            print(f"No valid masks found with prompt: '{text_prompt}'")

            # Clean up resources before returning
            if torch.backends.mps.is_available():
                # For MPS (Metal), we need explicit synchronization
                torch.mps.synchronize()  # Ensure all MPS operations are complete
            
            del detections  # Delete detection object

            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)

            return image, [], [], [], []

        # === Process each region using intermediate layer features ===
        region_embeddings = []
        region_metadata = []
        valid_masks = []
        valid_labels = []

        print(f"Processing {len(masks)} regions with intermediate layer features (layer {optimal_layer})")

        # Process each region separately
        for i, mask in enumerate(masks):
            try:
                # Create a cropped image for this region using the bbox
                y_indices, x_indices = np.where(mask)
                if len(y_indices) == 0 or len(x_indices) == 0:
                    print(f"Empty mask for region {i}, skipping")
                    continue

                x_min, x_max = x_indices.min(), x_indices.max()
                y_min, y_max = y_indices.min(), y_indices.max()
                
                # Add some padding to the bbox
                padding = 5
                x_min = max(0, x_min - padding)
                y_min = max(0, y_min - padding)
                x_max = min(image.shape[1] - 1, x_max + padding)
                y_max = min(image.shape[0] - 1, y_max + padding)
                
                # Create a cropped image 
                cropped_image = image[y_min:y_max+1, x_min:x_max+1]
                
                # Skip if crop is too small
                if cropped_image.shape[0] < 10 or cropped_image.shape[1] < 10:
                    print(f"Cropped region {i} too small: {cropped_image.shape}, skipping")
                    continue
                
                # Convert to PIL for model input
                cropped_pil = Image.fromarray(cropped_image)
                
                # Process with PE model - use INTERMEDIATE layer features
                with torch.no_grad():
                    # Convert to tensor in Float32 first (MPS compatibility)
                    pe_input = preprocess_to_use(cropped_pil).unsqueeze(0).to(device_to_use)
                    
                    # Try to get intermediate layer features
                    features = None
                    embedding_method = "unknown"
                    
                    # Enable AMP for MPS if using PyTorch 2.0+
                    if torch.backends.mps.is_available() and hasattr(torch.amp, 'autocast'):
                        amp_context = torch.amp.autocast(device_type='mps', dtype=torch.float16)
                    else:
                        # Use a dummy context manager if autocast not available
                        from contextlib import nullcontext
                        amp_context = nullcontext()
                        
                    with amp_context:
                        # Method 1: Try using VisionTransformer if available (preferred)
                        if pe_vit_model_to_use is not None:
                            try:
                                # Use forward_features with layer_idx parameter
                                features = pe_vit_model_to_use.forward_features(pe_input, layer_idx=optimal_layer)
                                embedding_method = "vit_forward_features"
                            except Exception as e:
                                print(f"Error using VisionTransformer forward_features: {e}")
                                
                        # Method 2: Try using model.visual if it exists
                        if features is None and hasattr(pe_model_to_use, 'visual'):
                            try:
                                # Some vision transformers expose intermediate features
                                if hasattr(pe_model_to_use.visual, 'transformer'):
                                    # Run forward pass and capture all intermediate activations
                                    output = pe_model_to_use.visual.transformer(
                                        pe_model_to_use.visual.conv1(pe_input),
                                        output_hidden_states=True
                                    )
                                    # Get the specific layer we want
                                    if isinstance(output, tuple) and len(output) > 1:
                                        # output[1] typically contains all hidden states
                                        hidden_states = output[1]
                                        if isinstance(hidden_states, list) and len(hidden_states) > optimal_layer:
                                            features = hidden_states[optimal_layer]
                                            embedding_method = "visual_transformer_hidden_states"
                                            
                            except Exception as e:
                                print(f"Error accessing visual transformer: {e}")
                        
                        # Method 3: Fallback to using the final output embedding if needed
                        if features is None:
                            print(f"Warning: Could not access intermediate layer {optimal_layer}, using final output")
                            features = pe_model_to_use.encode_image(pe_input)
                            embedding_method = "encode_image_fallback"
                    
                    # Process features based on their shape
                    if len(features.shape) == 3:  # [batch, sequence_length, embedding_dim]
                        # For transformer features, we need to pool the token embeddings
                        region_embedding = features.mean(dim=1)  # Average pooling over sequence
                    else:
                        # If already pooled or a single vector
                        region_embedding = features
                    
                    # Normalize embedding
                    region_embedding = torch.nn.functional.normalize(region_embedding, dim=-1)
                
                # Make sure to move result back to CPU for easier handling
                region_embeddings.append(region_embedding.squeeze(0).cpu())
                
                # Calculate bounding box and other metadata
                bbox = [int(x_min), int(y_min), int(x_max), int(y_max)]

                # Calculate area
                area = np.sum(mask)

                # Generate a unique ID for this region
                region_id = str(uuid.uuid4())

                # Store metadata
                metadata = {
                    "region_id": region_id,
                    "image_source": image_source if isinstance(image_source, str) else "uploaded_image",
                    "bbox": bbox,
                    "area": area,
                    "area_ratio": area / image_area,
                    "phrase": labels[i],
                    "embedding_method": embedding_method,
                    "layer_used": optimal_layer
                }

                region_metadata.append(metadata)
                valid_masks.append(mask)
                valid_labels.append(labels[i])
                print(f"Successfully processed region {i}: {labels[i]} using {embedding_method}")
            except Exception as e:
                print(f"Error processing region {i}: {e}")
                import traceback
                traceback.print_exc()
                continue

        # Remove temp file
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)

        # Explicit synchronization for MPS
        if torch.backends.mps.is_available():
            torch.mps.synchronize()

        print(f"Extracted {len(region_embeddings)} valid region embeddings")
        return image, valid_masks, region_embeddings, region_metadata, valid_labels

    except Exception as e:
        print(f"Error in extract_region_embeddings_autodistill: {e}")
        import traceback
        traceback.print_exc()

        # Clean up
        if 'temp_image_path' in locals() and os.path.exists(temp_image_path):
            os.remove(temp_image_path)

        # Explicit synchronization for MPS
        if torch.backends.mps.is_available():
            torch.mps.synchronize()

        return None, [], [], [], []

def extract_whole_image_embeddings(
    image_source,
    pe_model_param=None,
    pe_vit_model_param=None,
    preprocess_param=None,
    device_param=None,
    max_image_size=800,
    optimal_layer=40
):
    """
    Extract embeddings for whole images using only Perception Encoder without region detection
    """
    # Access global models if needed
    global pe_model, pe_vit_model, preprocess, device
    
    # Use provided models or fall back to globals
    pe_model_to_use = pe_model_param if pe_model_param is not None else pe_model
    pe_vit_model_to_use = pe_vit_model_param if pe_vit_model_param is not None else pe_vit_model
    preprocess_to_use = preprocess_param if preprocess_param is not None else preprocess
    device_to_use = device_param if device_param is not None else device
    
    try:
        print(f"[STATUS] Starting whole image embedding extraction...")
        # Load the image
        if isinstance(image_source, str):
            if image_source.startswith(('http://', 'https://')):
                print(f"[STATUS] Downloading image from URL...")
                pil_image = download_image(image_source)
            else:
                print(f"[STATUS] Loading image from path: {os.path.basename(image_source)}")
                pil_image = load_local_image(image_source)
        else:
            print(f"[STATUS] Processing uploaded image...")
            pil_image = image_source

        # Resize image to control memory usage
        width, height = pil_image.size
        if width > max_image_size or height > max_image_size:
            if width > height:
                new_width = max_image_size
                new_height = int(height * (max_image_size / width))
            else:
                new_height = max_image_size
                new_width = int(width * (max_image_size / height))

            print(f"[STATUS] Resizing image from {width}x{height} to {new_width}x{new_height}")
            pil_image = pil_image.resize((new_width, new_height), Image.LANCZOS)

        # Convert to numpy array for processing
        image = np.array(pil_image)
        print(f"[STATUS] Image converted to array, shape: {image.shape}")

        # Create metadata for the whole image
        image_id = str(uuid.uuid4())
        metadata = {
            "region_id": image_id,
            "image_source": str(image_source) if isinstance(image_source, str) else "uploaded_image",
            "bbox": [0, 0, image.shape[1], image.shape[0]],  # Full image bbox
            "area": image.shape[0] * image.shape[1],
            "area_ratio": 1.0,  # Full image has area ratio of 1
            "processing_type": "whole_image",  # Indicates this is a whole image, not a region
            "embedding_method": "pe_encoder",
            "layer_used": optimal_layer
        }

        # Process the image with PE model - using the SAME approach as in region processing
        print(f"[STATUS] Processing image with Perception Encoder...")
        
        # Convert to PIL for model input
        pil_img = Image.fromarray(image)
        
        # Process with PE model
        with torch.no_grad():
            # Convert to tensor in Float32 first (MPS compatibility)
            pe_input = preprocess_to_use(pil_img).unsqueeze(0).to(device_to_use)
            
            # Try to get intermediate layer features using the same methods as region processing
            features = None
            embedding_method = "unknown"
            
            # Enable AMP for MPS if using PyTorch 2.0+
            if torch.backends.mps.is_available() and hasattr(torch.amp, 'autocast'):
                amp_context = torch.amp.autocast(device_type='mps', dtype=torch.float16)
            else:
                # Use a dummy context manager if autocast not available
                amp_context = nullcontext()
                
            with amp_context:
                # Method 1: Try using VisionTransformer if available (preferred)
                if pe_vit_model_to_use is not None:
                    try:
                        # Use forward_features with layer_idx parameter
                        features = pe_vit_model_to_use.forward_features(pe_input, layer_idx=optimal_layer)
                        embedding_method = "vit_forward_features"
                        print(f"[STATUS] Successfully extracted features using VisionTransformer forward_features")
                    except Exception as e:
                        print(f"[STATUS] Error using VisionTransformer forward_features: {e}")
                        
                # Method 2: Try using model.visual if it exists
                if features is None and hasattr(pe_model_to_use, 'visual'):
                    try:
                        # Some vision transformers expose intermediate features
                        if hasattr(pe_model_to_use.visual, 'transformer'):
                            # Run forward pass and capture all intermediate activations
                            output = pe_model_to_use.visual.transformer(
                                pe_model_to_use.visual.conv1(pe_input),
                                output_hidden_states=True
                            )
                            # Get the specific layer we want
                            if isinstance(output, tuple) and len(output) > 1:
                                # output[1] typically contains all hidden states
                                hidden_states = output[1]
                                if isinstance(hidden_states, list) and len(hidden_states) > optimal_layer:
                                    features = hidden_states[optimal_layer]
                                    embedding_method = "visual_transformer_hidden_states"
                                    print(f"[STATUS] Successfully extracted features using visual transformer hidden states")
                        
                    except Exception as e:
                        print(f"[STATUS] Error accessing visual transformer: {e}")
                
                # Method 3: Fallback to using the final output embedding if needed
                if features is None:
                    print(f"[STATUS] Could not access intermediate layer {optimal_layer}, using final output")
                    features = pe_model_to_use.encode_image(pe_input)
                    embedding_method = "encode_image_fallback"
                    print(f"[STATUS] Successfully extracted features using encode_image fallback")
            
                # Process features based on their shape
                if len(features.shape) == 3:  # [batch, sequence_length, embedding_dim]
                    # For transformer features, we need to pool the token embeddings
                    embedding = features.mean(dim=1)  # Average pooling over sequence
                    print(f"[STATUS] Applied mean pooling over sequence dimension")
                else:
                    # If already pooled or a single vector
                    embedding = features
                
                # Normalize embedding
                embedding = torch.nn.functional.normalize(embedding, dim=-1)
                print(f"[STATUS] Normalized embedding, shape: {embedding.shape}")
            
            # Update metadata with the method used
            metadata["embedding_method"] = embedding_method
            
            # Move embedding to CPU
            embedding_cpu = embedding.cpu()
        
        # Clean up CUDA/MPS memory if needed
        if torch.backends.mps.is_available():
            # For MPS (Metal), we need explicit synchronization
            torch.mps.synchronize()  # Ensure all MPS operations are complete
        
        # Return results
        print(f"[STATUS] Successfully extracted whole image embedding with shape: {embedding_cpu.shape}, method: {embedding_method}")
        return image, embedding_cpu, metadata, "Whole Image"
        
    except Exception as e:
        print(f"[ERROR] Error extracting whole image embedding: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

# =============================================================================
# DATABASE FUNCTIONS
# =============================================================================

def setup_qdrant(collection_name, vector_size):
    """Setup Qdrant collection for storing image region embeddings"""
    try:
        persist_path = "./image_retrieval_project/qdrant_data"
        os.makedirs(persist_path, exist_ok=True)

        client = QdrantClient(path=persist_path)
        print(f"Using local storage at {persist_path}")

        collections = client.get_collections().collections
        collection_names = [collection.name for collection in collections]

        if collection_name not in collection_names:
            # Create the collection with standard vector configuration - no named vectors/multi-vectors
            client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE
                ),
                # Ensure we have proper indexing for fast search
                optimizers_config=models.OptimizersConfigDiff(
                    memmap_threshold=10000  # Use memmap for large collections
                )
            )
            print(f"Created new collection: {collection_name}")
        else:
            print(f"Using existing collection: {collection_name}")
            # Verify the collection has the correct vector size
            collection_info = client.get_collection(collection_name=collection_name)
            existing_vector_size = collection_info.config.params.vectors.size
            if existing_vector_size != vector_size:
                print(f"⚠️ Warning: Collection {collection_name} has vector size {existing_vector_size}, but requested size is {vector_size}")
                print(f"This might cause issues when searching. Consider using a different collection name.")

        return client
    except Exception as e:
        print(f"Error in setup_qdrant: {e}")
        import traceback
        traceback.print_exc()
        return None

def store_embeddings_in_qdrant(client, collection_name, embeddings, metadata_list):
    """Store embeddings in Qdrant collection"""
    try:
        # First, ensure all embeddings are in the correct format
        embedding_vectors = []
        for emb in embeddings:
            # Check if embedding is already a tensor, get numpy array
            if isinstance(emb, torch.Tensor):
                emb_numpy = emb.detach().cpu().numpy()
            else:
                emb_numpy = np.array(emb)
                
            # If we have a 1D array with shape (D,), reshape to (1, D)
            if len(emb_numpy.shape) == 1:
                emb_vector = emb_numpy
            # If we have a 2D array with shape (1, D), extract the inner vector
            elif len(emb_numpy.shape) == 2 and emb_numpy.shape[0] == 1:
                emb_vector = emb_numpy[0]
            else:
                # For any other shapes, flatten to ensure 1D
                emb_vector = emb_numpy.flatten()
                
            embedding_vectors.append(emb_vector)

        point_ids = [metadata["region_id"] for metadata in metadata_list]

        # Convert numpy types in metadata to Python native types
        sanitized_metadata = []
        for metadata in metadata_list:
            sanitized = {}
            for key, value in metadata.items():
                if isinstance(value, np.integer):
                    sanitized[key] = int(value)
                elif isinstance(value, np.floating):
                    sanitized[key] = float(value)
                elif isinstance(value, np.ndarray):
                    sanitized[key] = value.tolist()
                elif isinstance(value, list):
                    sanitized[key] = [int(x) if isinstance(x, np.integer) else
                                     float(x) if isinstance(x, np.floating) else x
                                     for x in value]
                else:
                    sanitized[key] = value
            sanitized_metadata.append(sanitized)

        # Prepare points for upsert - make sure vectors are all the correct shape (1D only)
        points = []
        for i in range(len(embedding_vectors)):
            points.append(models.PointStruct(
                id=point_ids[i],
                vector=embedding_vectors[i],
                payload=sanitized_metadata[i]
            ))

        # Upsert points in batches
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i+batch_size]
            client.upsert(
                collection_name=collection_name,
                points=batch
            )

        print(f"Stored {len(points)} embeddings in Qdrant collection: {collection_name}")
        return point_ids
    except Exception as e:
        print(f"Error in store_embeddings_in_qdrant: {e}")
        import traceback
        traceback.print_exc()
        return []

def visualize_detected_regions(image, masks, labels, title="Detected Regions", figsize=(10, 10)):
    """Create a visualization of detected regions in an image"""
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(image)

    # Draw each mask with a different color
    for i, mask in enumerate(masks):
        color = np.random.random(3)

        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        cv2.drawContours(image, contours, -1, color * 255, 2)

        M = cv2.moments(mask.astype(np.uint8))
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            label_text = labels[i] if i < len(labels) else f"Region {i+1}"
            ax.text(cx, cy, label_text, color='white',
                   bbox=dict(facecolor='black', alpha=0.7))

    ax.set_title(title)
    ax.axis('off')
    plt.tight_layout()

    return fig

# =============================================================================
# MAIN PROCESSING PIPELINE
# =============================================================================

def process_folder_for_region_database(
    folder_path,
    collection_name,
    text_prompts,
    pe_model,
    pe_vit_model, 
    preprocess,
    device,
    optimal_layer=40,
    min_area_ratio=0.01,
    max_regions=10,
    visualize_samples=True,
    sample_count=3,
    checkpoint_interval=10,
    resume_from_checkpoint=True
):
    """Process all images in a folder and build a searchable database"""
    print(f"Starting to process folder: {folder_path}")
    print(f"Looking for objects matching: {text_prompts}")
    print(f"Using embedding layer: {optimal_layer}")

    if not os.path.exists(folder_path):
        print(f"❌ Error: Folder not found at {folder_path}")
        return None

    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_paths = []

    for file in os.listdir(folder_path):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_paths.append(os.path.join(folder_path, file))

    if not image_paths:
        print(f"❌ Error: No images found in {folder_path}")
        return None

    print(f"📁 Found {len(image_paths)} images to process")

    # Initialize tracking variables
    processed_count = 0
    failed_count = 0
    skipped_count = 0
    total_regions = 0

    # Initialize database variables
    client = None
    vector_size = None
    initialization_complete = False
    
    # Include layer in collection name
    region_collection = f"{collection_name}_layer{optimal_layer}"
    print(f"📊 Using collection name with layer: {region_collection}")

    start_time = time.time()

    # Process images
    for idx, image_path in enumerate(image_paths):
        print(f"\nProcessing image {idx+1}/{len(image_paths)}: {os.path.basename(image_path)}")

        try:
            # Extract regions
            image, masks, embeddings, metadata, labels = extract_region_embeddings_autodistill(
                image_path,
                text_prompt=text_prompts,
                pe_model_param=pe_model,
                pe_vit_model_param=pe_vit_model,
                preprocess_param=preprocess,
                device_param=device,
                is_url=False,
                min_area_ratio=min_area_ratio,
                max_regions=max_regions,
                optimal_layer=optimal_layer
            )

            if image is not None and len(embeddings) > 0:
                print(f"✅ Found {len(embeddings)} valid regions in image {idx+1}")

                # Initialize database if not done yet
                if not initialization_complete:
                    vector_size = embeddings[0].shape[0]
                    print(f"📊 Vector dimension: {vector_size}")

                    client = setup_qdrant(region_collection, vector_size)
                    if client is None:
                        print("❌ Failed to initialize Qdrant client.")
                        return None

                    initialization_complete = True

                # Store embeddings
                store_embeddings_in_qdrant(client, region_collection, embeddings, metadata)
                total_regions += len(embeddings)
                processed_count += 1
            else:
                print(f"⚠️ No valid regions found in image {idx+1}, skipping")
                skipped_count += 1

        except Exception as e:
            print(f"❌ Error processing image {idx+1}: {e}")
            failed_count += 1
            continue

        # Print progress
        if (idx+1) % 5 == 0 or idx == len(image_paths) - 1:
            elapsed_time = time.time() - start_time
            print(f"\n📊 Progress: {idx+1}/{len(image_paths)} images, {total_regions} total regions")
            print(f"⏱️ Time elapsed: {elapsed_time:.1f}s")
            print(f"✅ Processed: {processed_count}, ⏭️ Skipped: {skipped_count}, ❌ Failed: {failed_count}")

    # Print final statistics
    elapsed_time = time.time() - start_time
    print("\n📊 Processing Complete!")
    print(f"⏱️ Total time: {elapsed_time:.1f}s")
    print(f"🔍 Images processed: {processed_count}/{len(image_paths)}")
    print(f"⏭️ Images skipped (no regions): {skipped_count}") 
    print(f"❌ Images failed: {failed_count}")
    print(f"📦 Total regions stored: {total_regions}")
    print(f"📊 Embeddings created with layer: {optimal_layer}")

    return client

def process_folder_for_whole_image_database(
    folder_path,
    collection_name,
    pe_model,
    pe_vit_model, 
    preprocess,
    device,
    optimal_layer=40,
    checkpoint_interval=10,
    resume_from_checkpoint=True
):
    """Process all images in a folder and build a searchable database of whole image embeddings"""
    print(f"Starting to process folder for whole image embeddings: {folder_path}")

    if not os.path.exists(folder_path):
        print(f"❌ Error: Folder not found at {folder_path}")
        return None

    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_paths = []

    for file in os.listdir(folder_path):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_paths.append(os.path.join(folder_path, file))

    if not image_paths:
        print(f"❌ Error: No images found in {folder_path}")
        return None

    print(f"📁 Found {len(image_paths)} images to process")

    # Initialize tracking variables
    processed_count = 0
    failed_count = 0
    total_images = 0

    # Initialize database variables
    client = None
    vector_size = None
    initialization_complete = False

    start_time = time.time()

    # Create a special collection name for whole images with layer information
    whole_image_collection = f"{collection_name}_whole_images_layer{optimal_layer}"
    print(f"📊 Using collection name: {whole_image_collection}")

    # Process images
    for idx, image_path in enumerate(image_paths):
        print(f"\nProcessing image {idx+1}/{len(image_paths)}: {os.path.basename(image_path)}")

        try:
            # Extract whole image embedding
            image, embedding, metadata, label = extract_whole_image_embeddings(
                image_path,
                pe_model_param=pe_model,
                pe_vit_model_param=pe_vit_model,
                preprocess_param=preprocess,
                device_param=device,
                optimal_layer=optimal_layer
            )

            if image is not None and embedding is not None:
                print(f"✅ Successfully extracted embedding for image {idx+1}")

                # Initialize database if not done yet
                if not initialization_complete:
                    vector_size = embedding.shape[1]
                    print(f"📊 Vector dimension: {vector_size}")

                    client = setup_qdrant(whole_image_collection, vector_size)
                    if client is None:
                        print("❌ Failed to initialize Qdrant client.")
                        return None

                    initialization_complete = True

                # Store embedding
                store_embeddings_in_qdrant(client, whole_image_collection, [embedding], [metadata])
                total_images += 1
                processed_count += 1
            else:
                print(f"⚠️ Failed to extract embedding for image {idx+1}, skipping")
                failed_count += 1

        except Exception as e:
            print(f"❌ Error processing image {idx+1}: {e}")
            failed_count += 1
            continue

        # Print progress
        if (idx+1) % 5 == 0 or idx == len(image_paths) - 1:
            elapsed_time = time.time() - start_time
            print(f"\n📊 Progress: {idx+1}/{len(image_paths)} images, {total_images} total processed")
            print(f"⏱️ Time elapsed: {elapsed_time:.1f}s")
            print(f"✅ Processed: {processed_count}, ❌ Failed: {failed_count}")

    # Print final statistics
    elapsed_time = time.time() - start_time
    print("\n📊 Processing Complete!")
    print(f"⏱️ Total time: {elapsed_time:.1f}s")
    print(f"🔍 Images processed: {processed_count}/{len(image_paths)}")
    print(f"❌ Images failed: {failed_count}")
    print(f"📊 Embeddings created with layer: {optimal_layer}")

    return client

# =============================================================================
# BATCH PROCESSING FUNCTIONS
# =============================================================================

def process_folder_with_progress(folder_path, prompts, collection_name):
    """Process folder with progress updates for the UI"""
    try:
        print(f"[STATUS] Starting folder processing: {folder_path}")
        print(f"[STATUS] Using collection: {collection_name}")
        print(f"[STATUS] Using prompts: {prompts}")
        
        # Get global model variables
        global pe_model, pe_vit_model, preprocess, device
        
        # Split prompts if provided as comma-separated string
        prompt_list = [p.strip() for p in prompts.split(",")] if prompts else ["person", "building", "car", "text", "object"]
        print(f"[STATUS] Parsed prompts: {prompt_list}")
        
        # Get file list
        image_files = [f for f in os.listdir(folder_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
        total = len(image_files)
        
        print(f"[STATUS] Found {total} images to process in folder")
        
        if total == 0:
            print(f"[STATUS] No images found in folder {folder_path}")
            return "No images found in folder", gr.update(visible=False)
        
        # Process images with progress updates
        client = None
        processed = 0
        skipped = 0
        errors = 0
        total_regions = 0
        
        # Update Gradio with initial status
        yield f"🔍 Starting to process {total} images in {folder_path}", gr.update(visible=False)
        
        print(f"[STATUS] Beginning image processing loop")
        for i, img_file in enumerate(image_files):
            # Process image
            img_path = os.path.join(folder_path, img_file)
            
            # Yield progress update with percentage
            progress_pct = ((i+1) / total) * 100
            yield f"🔄 Processing: {i+1}/{total} images ({progress_pct:.1f}%)\n📄 Current: {img_file}", gr.update(visible=False)
            
            print(f"[STATUS] Processing image {i+1}/{total}: {img_file}")
            
            try:
                # Extract region embeddings
                image, masks, embeddings, metadata, labels = extract_region_embeddings_autodistill(
                    img_path,
                    ",".join(prompt_list),
                    pe_model,
                    pe_vit_model,
                    preprocess,
                    device,
                    min_area_ratio=0.005,
                    max_regions=5
                )
                
                if image is not None and len(embeddings) > 0:
                    print(f"[STATUS] Found {len(embeddings)} regions in {img_file}")
                    # Setup Qdrant client if needed
                    if client is None:
                        vector_size = embeddings[0].shape[0]
                        print(f"[STATUS] Setting up database with vector size {vector_size}")
                        client = setup_qdrant(collection_name, vector_size)
                    
                    # Store embeddings
                    print(f"[STATUS] Storing {len(embeddings)} embeddings in database")
                    store_embeddings_in_qdrant(client, collection_name, embeddings, metadata)
                    processed += 1
                    total_regions += len(embeddings)
                else:
                    print(f"[STATUS] No valid regions found in {img_file}")
                    skipped += 1
            except Exception as e:
                print(f"[ERROR] Error processing {img_path}: {e}")
                import traceback
                traceback.print_exc()
                errors += 1
            
            # Every few images, yield a progress update with detailed stats
            if (i + 1) % 3 == 0 or i == total - 1:
                stats = (
                    f"📊 Progress: {i+1}/{total} images ({(i+1)/total*100:.1f}%)\n"
                    f"✅ Processed: {processed} images\n"
                    f"⏭️ Skipped: {skipped} images (no regions found)\n"
                    f"❌ Errors: {errors} images\n"
                    f"🔍 Total regions found: {total_regions}"
                )
                yield stats, gr.update(visible=i == total-1)
                
        print(f"[STATUS] Processing complete - processed: {processed}, skipped: {skipped}, errors: {errors}, total regions: {total_regions}")
        
        if client is not None:
            print(f"[STATUS] Closing database connection")
            client.close()
        
        final_message = (
            f"✅ Complete! Processing summary:\n\n"
            f"📁 Total images: {total}\n"
            f"✅ Successfully processed: {processed}\n"
            f"⏭️ Skipped (no regions): {skipped}\n"
            f"❌ Errors: {errors}\n"
            f"🔍 Total regions stored: {total_regions}\n\n"
            f"📦 Database collection: '{collection_name}'"
        )
        print(f"[STATUS] {final_message}")
        return final_message, gr.update(visible=True)
    except Exception as e:
        error_message = f"❌ Error processing folder: {str(e)}"
        print(f"[ERROR] {error_message}")
        import traceback
        traceback.print_exc()
        return error_message, gr.update(visible=False)

def process_folder_with_progress_advanced(folder_path, prompts, collection_name, 
                                  optimal_layer=40, min_area_ratio=0.005, max_regions=5, 
                                  resume_from_checkpoint=True):
    """Process folder with progress updates for the UI with advanced parameters"""
    try:
        print(f"[STATUS] Starting folder processing: {folder_path}")
        print(f"[STATUS] Using collection: {collection_name}")
        print(f"[STATUS] Using prompts: {prompts}")
        print(f"[STATUS] Advanced parameters: optimal_layer={optimal_layer}, min_area_ratio={min_area_ratio}, " 
              f"max_regions={max_regions}, resume_from_checkpoint={resume_from_checkpoint}")
        
        # Get global model variables
        global pe_model, pe_vit_model, preprocess, device
        
        # Split prompts if provided as comma-separated string
        prompt_list = [p.strip() for p in prompts.split(",")] if prompts else ["person", "building", "car", "text", "object"]
        print(f"[STATUS] Parsed prompts: {prompt_list}")
        
        # Get file list
        if not os.path.exists(folder_path):
            return f"❌ Error: Folder {folder_path} does not exist", gr.update(visible=False)
            
        image_files = [f for f in os.listdir(folder_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
        total = len(image_files)
        
        print(f"[STATUS] Found {total} images to process in folder")
        
        if total == 0:
            print(f"[STATUS] No images found in folder {folder_path}")
            return "No images found in folder", gr.update(visible=False)
        
        # Process images with progress updates
        client = None
        processed = 0
        skipped = 0
        errors = 0
        total_regions = 0
        
        # Use collection name provided (should already include layer info from caller)
        collection_with_layer = collection_name
        
        # Update Gradio with initial status
        yield (f"🔍 Starting to process {total} images in {folder_path}\n"
              f"📊 Parameters:\n"
              f"  - Semantic Layer: {optimal_layer}\n"
              f"  - Min Region Size: {min_area_ratio}\n"
              f"  - Max Regions: {max_regions}\n"
              f"  - Resume from checkpoint: {resume_from_checkpoint}\n"
              f"  - Collection name: {collection_with_layer}"), gr.update(visible=False)
        
        print(f"[STATUS] Beginning image processing loop")
        for i, img_file in enumerate(image_files):
            # Process image
            img_path = os.path.join(folder_path, img_file)
            
            # Yield progress update with percentage
            progress_pct = ((i+1) / total) * 100
            yield f"🔄 Processing: {i+1}/{total} images ({progress_pct:.1f}%)\n📄 Current: {img_file}", gr.update(visible=False)
            
            print(f"[STATUS] Processing image {i+1}/{total}: {img_file}")
            
            try:
                # Extract region embeddings with custom parameters
                image, masks, embeddings, metadata, labels = extract_region_embeddings_autodistill(
                    img_path,
                    ",".join(prompt_list),
                    pe_model_param=pe_model,
                    pe_vit_model_param=pe_vit_model,
                    preprocess_param=preprocess,
                    device_param=device,
                    min_area_ratio=min_area_ratio,
                    max_regions=max_regions,
                    optimal_layer=optimal_layer
                )
                
                if image is not None and len(embeddings) > 0:
                    print(f"[STATUS] Found {len(embeddings)} regions in {img_file}")
                    # Setup Qdrant client if needed
                    if client is None:
                        vector_size = embeddings[0].shape[0]
                        print(f"[STATUS] Setting up database with vector size {vector_size}")
                        client = setup_qdrant(collection_with_layer, vector_size)
                    
                    # Add layer information to each region's metadata
                    for md in metadata:
                        md["embedding_layer"] = optimal_layer
                    
                    # Store embeddings
                    print(f"[STATUS] Storing {len(embeddings)} embeddings in database")
                    store_embeddings_in_qdrant(client, collection_with_layer, embeddings, metadata)
                    processed += 1
                    total_regions += len(embeddings)
                else:
                    print(f"[STATUS] No valid regions found in {img_file}")
                    skipped += 1
            except Exception as e:
                print(f"[ERROR] Error processing {img_path}: {e}")
                import traceback
                traceback.print_exc()
                errors += 1
            
            # Every few images, yield a progress update with detailed stats
            if (i + 1) % 3 == 0 or i == total - 1:
                stats = (
                    f"📊 Progress: {i+1}/{total} images ({(i+1)/total*100:.1f}%)\n"
                    f"✅ Processed: {processed} images\n"
                    f"⏭️ Skipped: {skipped} images (no regions found)\n"
                    f"❌ Errors: {errors} images\n"
                    f"🔍 Total regions found: {total_regions}\n\n"
                    f"🧠 Using semantic layer: {optimal_layer}\n"
                    f"🔍 Min region size: {min_area_ratio}\n"
                    f"📏 Max regions per image: {max_regions}\n"
                    f"📦 Collection: {collection_with_layer}"
                )
                yield stats, gr.update(visible=i == total-1)
                
        print(f"[STATUS] Processing complete - processed: {processed}, skipped: {skipped}, errors: {errors}, total regions: {total_regions}")
        
        if client is not None:
            print(f"[STATUS] Closing database connection")
            client.close()
        
        final_message = (
            f"✅ Complete! Processing summary:\n\n"
            f"📁 Total images: {total}\n"
            f"✅ Successfully processed: {processed}\n"
            f"⏭️ Skipped (no regions): {skipped}\n"
            f"❌ Errors: {errors}\n"
            f"🔍 Total regions stored: {total_regions}\n\n"
            f"🧠 Semantic layer used: {optimal_layer}\n"
            f"🔍 Min region size: {min_area_ratio}\n"
            f"📏 Max regions per image: {max_regions}\n\n"
            f"📦 Database collection: '{collection_with_layer}'"
        )
        print(f"[STATUS] {final_message}")
        return final_message, gr.update(visible=True)
    except Exception as e:
        error_message = f"❌ Error processing folder: {str(e)}"
        print(f"[ERROR] {error_message}")
        import traceback
        traceback.print_exc()
        return error_message, gr.update(visible=False)

def process_folder_with_progress_whole_images(folder_path, collection_name, 
                                    optimal_layer=40, resume_from_checkpoint=True):
    """Process folder with whole image embeddings for the UI"""
    try:
        print(f"[STATUS] Starting whole image processing for folder: {folder_path}")
        print(f"[STATUS] Using collection: {collection_name}")
        print(f"[STATUS] Using embedding layer: {optimal_layer}")
        
        # Get global model variables
        global pe_model, pe_vit_model, preprocess, device
        
        # Get file list
        image_files = [f for f in os.listdir(folder_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
        total = len(image_files)
        
        print(f"[STATUS] Found {total} images to process in folder")
        
        if total == 0:
            print(f"[STATUS] No images found in folder {folder_path}")
            return "No images found in folder", gr.update(visible=False)
        
        # Process images with progress updates
        client = None
        processed = 0
        errors = 0
        
        # Use collection name provided (should already include layer info from caller)
        whole_image_collection = collection_name
        
        # Update Gradio with initial status
        yield f"🔍 Starting to process {total} whole images in {folder_path}\n📊 Collection: {whole_image_collection}\n🧠 Using embedding layer: {optimal_layer}", gr.update(visible=False)
        
        print(f"[STATUS] Beginning whole image processing loop")
        for i, img_file in enumerate(image_files):
            # Process image
            img_path = os.path.join(folder_path, img_file)
            
            # Yield progress update with percentage
            progress_pct = ((i+1) / total) * 100
            yield f"🔄 Processing: {i+1}/{total} images ({progress_pct:.1f}%)\n📄 Current: {img_file}", gr.update(visible=False)
            
            print(f"[STATUS] Processing image {i+1}/{total}: {img_file}")
            
            try:
                # Extract whole image embedding
                image, embedding, metadata, label = extract_whole_image_embeddings(
                    img_path,
                    pe_model,
                    pe_vit_model,
                    preprocess,
                    device,
                    optimal_layer=optimal_layer
                )
                
                if image is not None and embedding is not None:
                    print(f"[STATUS] Successfully extracted embedding for {img_file}")
                    # Setup Qdrant client if needed
                    if client is None:
                        vector_size = embedding.shape[1]
                        print(f"[STATUS] Setting up database with vector size {vector_size}")
                        client = setup_qdrant(whole_image_collection, vector_size)
                    
                    # Store embedding
                    print(f"[STATUS] Storing embedding in database")
                    store_embeddings_in_qdrant(client, whole_image_collection, [embedding], [metadata])
                    processed += 1
                else:
                    print(f"[STATUS] Failed to extract embedding for {img_file}")
            except Exception as e:
                print(f"[ERROR] Error processing {img_path}: {e}")
                import traceback
                traceback.print_exc()
                errors += 1
                
            # Provide periodic updates
            if (i+1) % 5 == 0 or i == len(image_files) - 1:
                yield f"🔄 Processing: {i+1}/{total} images ({progress_pct:.1f}%)\n✅ Processed: {processed}\n❌ Errors: {errors}\n🧠 Using embedding layer: {optimal_layer}", gr.update(visible=False)
        
        # Final update
        if processed > 0:
            final_message = f"✅ Processing complete!\n📊 Processed {processed}/{total} images\n🗄️ Collection: {whole_image_collection}\n🧠 Using embedding layer: {optimal_layer}"
            print(f"[STATUS] {final_message}")
            return final_message, gr.update(visible=True)
        else:
            error_message = f"❌ Processing failed. No images were successfully processed."
            print(f"[STATUS] {error_message}")
            return error_message, gr.update(visible=False)
        
    except Exception as e:
        error_message = f"❌ Error: {str(e)}"
        print(f"[ERROR] Batch processing error: {e}")
        import traceback
        traceback.print_exc()
        return error_message, gr.update(visible=False)

# =============================================================================
# GRADIO INTERFACE CLASSES
# =============================================================================

class GradioInterface:
    """Gradio interface for the Grounded SAM Region Search application"""
    
    def __init__(self, pe_model, pe_vit_model, preprocess, device):
        """Initialize the interface with required models"""
        self.pe_model = pe_model
        self.pe_vit_model = pe_vit_model
        self.preprocess = preprocess
        self.device = device
        
        # State variables
        self.detected_regions = {
            "image": None,
            "masks": [],
            "embeddings": [],
            "metadata": [],
            "labels": []
        }
        
        # Add whole image state
        self.whole_image = {
            "image": None,
            "embedding": None,
            "metadata": None,
            "label": None
        }
        
        self.image_cache = {}
        self.search_result_images = []
        self.active_client = None
        
        # Get available collections
        self.available_collections = list_available_databases()
        # Set default collection based on available databases
        if self.available_collections:
            self.active_collection = self.available_collections[0]
            print(f"[STATUS] Using default collection: {self.active_collection}")
        else:
            self.active_collection = "grounded_image_regions"
            print(f"[STATUS] No existing collections found. Using default name: {self.active_collection}")
    
    def set_active_collection(self, collection_name):
        """Set the active collection for search operations"""
        if collection_name and collection_name != "No databases found":
            if collection_name == self.active_collection:
                # No change needed
                print(f"[DEBUG] Collection already set to {collection_name}")
                return f"Using database: {collection_name}"
                
            previous = self.active_collection
            self.active_collection = collection_name
            print(f"[STATUS] Changed active collection from {previous} to {collection_name}")
            
            # Close existing client if database changed
            if self.active_client is not None:
                print(f"[STATUS] Closing previous database connection")
                self.active_client.close()
                self.active_client = None
                
            return f"Set active database to: {collection_name}"
        return "No database selected"

    def process_image_with_prompt(self, image, text_prompt, min_area_ratio=0.01):
        """Process an uploaded image with a text prompt to detect regions"""
        if isinstance(image, np.ndarray):
            image_pil = Image.fromarray(image)
        else:
            image_pil = image

        # Extract regions from image
        image_np, masks, embeddings, metadata, labels = extract_region_embeddings_autodistill(
            image_pil,
            text_prompt=text_prompt,
            pe_model_param=self.pe_model,
            pe_vit_model_param=self.pe_vit_model,
            preprocess_param=self.preprocess,
            device_param=self.device,
            is_url=False,
            min_area_ratio=min_area_ratio,
            max_regions=5,
            optimal_layer=40
        )

        # Store results
        self.detected_regions = {
            "image": image_np,
            "masks": masks,
            "embeddings": embeddings,
            "metadata": metadata,
            "labels": labels
        }

        if len(masks) == 0:
            return None, f"No regions found with prompt: '{text_prompt}'", gr.Dropdown(choices=[], value=None), None

        # Create visualization
        fig = visualize_detected_regions(image_np, masks, labels)
        
        # Create dropdown choices
        choices = [f"Region {i+1}: {label}" for i, label in enumerate(labels)]

        # Save figure to buffer
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)

        segmented_image = Image.open(buf)

        # Create preview of first region
        region_preview = None
        if len(masks) > 0:
            region_preview = self.create_region_preview(image_np, masks[0], labels[0])

        return segmented_image, f"Found {len(masks)} regions", gr.Dropdown(choices=choices, value=choices[0] if choices else None), region_preview

    def create_region_preview(self, image, mask, label=None):
        """Creates a visualization of a single region"""
        preview = image.copy()
        mask_3d = np.stack([mask, mask, mask], axis=2)
        highlighted = np.where(mask_3d, np.minimum(preview * 1.5, 255), preview * 0.4)

        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(highlighted, contours, -1, (0, 255, 0), 2)

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(highlighted.astype(np.uint8))
        if label:
            ax.set_title(label)
        ax.axis('off')

        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)

        return Image.open(buf)

    def search_region(self, region_selection, similarity_threshold=0.5, max_results=5, optimal_layer=40):
        """Search for similar regions based on the selected region's embedding"""
        print(f"[STATUS] Starting search for similar regions...")
        if region_selection is None or self.detected_regions["image"] is None:
            print(f"[STATUS] No region selected or no detected regions available")
            return None, "Please select a region first.", gr.update(visible=False), gr.update(choices=[], value=None)

        print(f"[STATUS] Processing region selection: {region_selection}")
        region_idx = int(region_selection.split(":")[0].replace("Region ", "")) - 1

        if region_idx < 0 or region_idx >= len(self.detected_regions["embeddings"]):
            print(f"[STATUS] Invalid region index: {region_idx}")
            return None, "Invalid region selection.", gr.update(visible=False), gr.update(choices=[], value=None)

        # Get the embedding for the selected region
        embedding = self.detected_regions["embeddings"][region_idx]
        print(f"[STATUS] Retrieved embedding for region {region_idx}, shape: {embedding.shape}")
        
        # Connect to Qdrant
        try:
            print(f"[STATUS] Connecting to vector database...")
            if self.active_client is None:
                self.active_client = QdrantClient(path="./image_retrieval_project/qdrant_data")
                print(f"[STATUS] Created new database connection")
            else:
                print(f"[STATUS] Using existing database connection")
            
            # Base collection name
            base_collection_name = self.active_collection
            
            # Define the possible collection names to search
            region_collection_name = f"{base_collection_name}_layer{optimal_layer}"
            whole_image_collection_name = f"{base_collection_name}_whole_images_layer{optimal_layer}"
            
            print(f"[STATUS] Looking for collections to search...")
            
            # Get available collections
            collections = self.active_client.get_collections().collections
            collection_names = [collection.name for collection in collections]
            print(f"[STATUS] Available collections: {collection_names}")
            
            # Check if the specific collections exist
            collections_to_search = []
            collection_types = []
            
            # First, check region collection (primary for region search)
            if region_collection_name in collection_names:
                collections_to_search.append(region_collection_name)
                collection_types.append("region")
                print(f"[STATUS] Will search region collection: {region_collection_name}")
            elif base_collection_name in collection_names:
                collections_to_search.append(base_collection_name)
                collection_types.append("region")
                print(f"[STATUS] Will search base collection: {base_collection_name}")
            
            # Then, check whole image collection (secondary for region search)
            if whole_image_collection_name in collection_names:
                collections_to_search.append(whole_image_collection_name)
                collection_types.append("whole_image")
                print(f"[STATUS] Will also search whole image collection: {whole_image_collection_name}")
            else:
                # Look for any whole image collection as fallback
                whole_image_collections = [c for c in collection_names if "whole_images" in c]
                if whole_image_collections:
                    collections_to_search.append(whole_image_collections[0])
                    collection_types.append("whole_image")
                    print(f"[STATUS] Will also search alternative whole image collection: {whole_image_collections[0]}")
            
            # If no collections found, return error
            if not collections_to_search:
                print(f"[ERROR] No suitable collections found to search")
                return None, f"Error: No suitable collections found to search.", gr.update(visible=False), gr.update(choices=[], value=None)
            
            # Ensure parameters are the correct type
            limit = int(max_results) if isinstance(max_results, str) else max_results
            threshold = float(similarity_threshold) if isinstance(similarity_threshold, str) else similarity_threshold
            
            # Convert embedding to list for Qdrant - properly format as 1D vector
            print(f"[STATUS] Converting embedding to list format...")
            if len(embedding.shape) == 2 and embedding.shape[0] == 1:
                # Extract the inner vector if shape is [1, D]
                embedding_list = embedding[0].cpu().numpy().tolist()
            else:
                # Otherwise convert the tensor to a flattened list
                embedding_list = embedding.cpu().numpy().flatten().tolist()
            
            # Search across all collections and gather results
            all_results = []
            
            for i, collection_name in enumerate(collections_to_search):
                collection_type = collection_types[i]
                print(f"[STATUS] Searching collection: {collection_name} (type: {collection_type})...")
                
                try:
                    # Set a higher limit to ensure we get enough results after combining
                    collection_limit = limit * 2
                    
                    search_results = self.active_client.search(
                        collection_name=collection_name,
                        query_vector=embedding_list,
                        limit=collection_limit,
                        score_threshold=threshold
                    )
                    
                    print(f"[STATUS] Found {len(search_results)} results in {collection_name}")
                    
                    # Tag results with their collection type
                    for result in search_results:
                        # Add collection info directly to payload
                        if hasattr(result, "payload"):
                            result.payload["collection_type"] = collection_type
                            result.payload["collection_name"] = collection_name
                    
                    all_results.extend(search_results)
                except Exception as e:
                    print(f"[WARNING] Error searching collection {collection_name}: {e}")
                    continue
            
            # Sort all results by score and limit to requested number
            all_results.sort(key=lambda x: x.score, reverse=True)
            filtered_results = all_results[:limit]
            
            print(f"[STATUS] Combined search complete, found {len(filtered_results)} results from {len(collections_to_search)} collections")
            
            if not filtered_results:
                print(f"[STATUS] No similar items found above threshold {threshold}")
                return None, f"No similar items found with similarity threshold {threshold}.", gr.update(visible=False), gr.update(choices=[], value=None)
            
            # Create visualization of combined results
            return self.create_unified_search_results_visualization(
                filtered_results, 
                self.detected_regions["image"],
                self.detected_regions["masks"][region_idx],
                self.detected_regions["labels"][region_idx],
                "region"
            )
        
        except Exception as e:
            error_message = f"Error searching for similar regions: {str(e)}"
            print(f"[ERROR] {error_message}")
            import traceback
            traceback.print_exc()
            return None, error_message, gr.update(visible=False), gr.update(choices=[], value=None)

    def search_whole_image(self, similarity_threshold=0.5, max_results=5, optimal_layer=40):
        """Search for similar whole images based on the processed image's embedding"""
        print(f"[STATUS] Starting search for similar whole images...")
        if self.whole_image["image"] is None or self.whole_image["embedding"] is None:
            print(f"[STATUS] No processed image available")
            return None, "Please process an image first.", gr.update(visible=False), gr.update(choices=[], value=None)

        # Check if the processed image used the same layer as requested for search
        actual_layer = self.whole_image.get("layer_used", -1)
        if actual_layer != optimal_layer:
            print(f"[WARNING] Search layer ({optimal_layer}) doesn't match the layer used for processing ({actual_layer})")

        # Get the embedding for the whole image
        embedding = self.whole_image["embedding"]
        print(f"[STATUS] Retrieved embedding for whole image, shape: {embedding.shape}")
        
        # Connect to Qdrant
        try:
            print(f"[STATUS] Connecting to vector database...")
            if self.active_client is None:
                self.active_client = QdrantClient(path="./image_retrieval_project/qdrant_data")
                print(f"[STATUS] Created new database connection")
            else:
                print(f"[STATUS] Using existing database connection")
            
            # Base collection name
            base_collection_name = self.active_collection
            
            # Define the possible collection names to search
            region_collection_name = f"{base_collection_name}_layer{optimal_layer}"
            whole_image_collection_name = f"{base_collection_name}_whole_images_layer{optimal_layer}"
            
            print(f"[STATUS] Looking for collections to search...")
            
            # Get available collections
            collections = self.active_client.get_collections().collections
            collection_names = [collection.name for collection in collections]
            print(f"[STATUS] Available collections: {collection_names}")
            
            # Check if the specific collections exist
            collections_to_search = []
            collection_types = []
            
            # First, check whole image collection (primary for whole image search)
            if whole_image_collection_name in collection_names:
                collections_to_search.append(whole_image_collection_name)
                collection_types.append("whole_image")
                print(f"[STATUS] Will search whole image collection: {whole_image_collection_name}")
            else:
                # Look for any whole image collection as fallback
                whole_image_collections = [c for c in collection_names if "whole_images" in c]
                if whole_image_collections:
                    collections_to_search.append(whole_image_collections[0])
                    collection_types.append("whole_image")
                    print(f"[STATUS] Will search alternative whole image collection: {whole_image_collections[0]}")
            
            # Then, check region collection (secondary for whole image search)
            if region_collection_name in collection_names:
                collections_to_search.append(region_collection_name)
                collection_types.append("region")
                print(f"[STATUS] Will also search region collection: {region_collection_name}")
            elif base_collection_name in collection_names:
                collections_to_search.append(base_collection_name)
                collection_types.append("region")
                print(f"[STATUS] Will also search base collection: {base_collection_name}")
            
            # If no collections found, return error
            if not collections_to_search:
                print(f"[ERROR] No suitable collections found to search")
                return None, f"Error: No suitable collections found to search.", gr.update(visible=False), gr.update(choices=[], value=None)
            
            # Ensure parameters are the correct type
            limit = int(max_results) if isinstance(max_results, str) else max_results
            threshold = float(similarity_threshold) if isinstance(similarity_threshold, str) else similarity_threshold
            
            # Convert embedding to list for Qdrant - properly format as 1D vector
            print(f"[STATUS] Converting embedding to list format...")
            if len(embedding.shape) == 2 and embedding.shape[0] == 1:
                # Extract the inner vector if shape is [1, D]
                embedding_list = embedding[0].cpu().numpy().tolist()
            else:
                # Otherwise convert the tensor to a flattened list
                embedding_list = embedding.cpu().numpy().flatten().tolist()
            
            # Search across all collections and gather results
            all_results = []
            
            for i, collection_name in enumerate(collections_to_search):
                collection_type = collection_types[i]
                print(f"[STATUS] Searching collection: {collection_name} (type: {collection_type})...")
                
                try:
                    # Set a higher limit to ensure we get enough results after combining
                    collection_limit = limit * 2
                    
                    search_results = self.active_client.search(
                        collection_name=collection_name,
                        query_vector=embedding_list,
                        limit=collection_limit,
                        score_threshold=threshold
                    )
                    
                    print(f"[STATUS] Found {len(search_results)} results in {collection_name}")
                    
                    # Tag results with their collection type
                    for result in search_results:
                        # Add collection info directly to payload
                        if hasattr(result, "payload"):
                            result.payload["collection_type"] = collection_type
                            result.payload["collection_name"] = collection_name
                    
                    all_results.extend(search_results)
                except Exception as e:
                    print(f"[WARNING] Error searching collection {collection_name}: {e}")
                    continue
            
            # Sort all results by score and limit to requested number
            all_results.sort(key=lambda x: x.score, reverse=True)
            filtered_results = all_results[:limit]
            
            print(f"[STATUS] Combined search complete, found {len(filtered_results)} results from {len(collections_to_search)} collections")
            
            if not filtered_results:
                print(f"[STATUS] No similar items found above threshold {threshold}")
                return None, f"No similar items found with similarity threshold {threshold}.", gr.update(visible=False), gr.update(choices=[], value=None)
            
            # Create visualization of combined results
            return self.create_unified_search_results_visualization(
                filtered_results, 
                self.whole_image["image"],
                None,  # No mask for whole image
                "Whole Image",  # Label for whole image
                "whole_image"
            )
        
        except Exception as e:
            error_message = f"Error searching for similar images: {str(e)}"
            print(f"[ERROR] {error_message}")
            import traceback
            traceback.print_exc()
            return None, error_message, gr.update(visible=False), gr.update(choices=[], value=None)

    def create_unified_search_results_visualization(self, filtered_results, query_image, query_mask=None, query_label="Query", query_type="region"):
        """Create visualization of search results with unified display for both region and whole image results"""
        print(f"[STATUS] Creating unified search results visualization for {len(filtered_results)} results...")
        self.search_result_images = []

        # Determine optimal grid layout based on number of results
        num_results = len(filtered_results)
        if num_results <= 3:
            grid_rows = 1
            grid_cols = num_results 
        elif num_results <= 6:
            grid_rows = 2
            grid_cols = 3
        else:
            grid_rows = 3
            grid_cols = 4  # Maximum 12 results displayed well
        
        # Calculate figure size (width, height) with better proportions
        fig_width = min(18, 6 + 3 * grid_cols)  # Limit max width
        fig_height = 4 + 3 * grid_rows
        
        # Prepare figure for visualization
        print(f"[STATUS] Setting up visualization figure with {grid_rows}x{grid_cols} grid...")
        fig = plt.figure(figsize=(fig_width, fig_height))
        
        # Create a layout with a dedicated area for query and proper spacing
        gs = gridspec.GridSpec(grid_rows + 1, grid_cols, height_ratios=[1] + [3] * grid_rows)
        
        # Query section (top row, spans all columns)
        ax_query = fig.add_subplot(gs[0, :])
        
        # Display the query differently based on type
        if query_type == "region" and query_mask is not None:
            # For regions, highlight the region in the query image
            highlighted = query_image.copy()
            mask_3d = np.stack([query_mask, query_mask, query_mask], axis=2)
            highlighted = np.where(mask_3d, np.minimum(highlighted * 1.5, 255), highlighted * 0.4)
            
            # Add contour
            contours, _ = cv2.findContours(
                query_mask.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(highlighted, contours, -1, (0, 255, 0), 2)
            
            ax_query.imshow(highlighted.astype(np.uint8))
            ax_query.set_title(f"Query Region: {query_label}", fontsize=14, pad=10)
        else:
            # For whole images, just show the image
            ax_query.imshow(query_image)
            ax_query.set_title(f"Query: {query_label}", fontsize=14, pad=10)
            
        ax_query.axis('off')

        # Create radio choices for result selection
        radio_choices = []
        result_info = ["# Search Results\n\n"]

        # Results section (grid layout below query)
        print(f"[STATUS] Processing {len(filtered_results)} result images...")
        for i, result in enumerate(filtered_results):
            if i >= grid_rows * grid_cols:  # Limit to grid capacity
                print(f"[STATUS] Only displaying first {grid_rows * grid_cols} results due to space constraints")
                break
                
            # Calculate row and column for this result
            row = (i // grid_cols) + 1  # +1 because query is in row 0
            col = i % grid_cols
            
            print(f"[STATUS] Processing result {i+1}/{len(filtered_results)}, placing at position ({row},{col})...")
            ax = fig.add_subplot(gs[row, col])

            try:
                # Extract metadata
                metadata = result.payload
                image_source = metadata.get("image_source", "Unknown")
                score = result.score
                
                # Get collection type (region or whole_image)
                collection_type = metadata.get("collection_type", "unknown")
                collection_name = metadata.get("collection_name", "unknown")
                
                print(f"[STATUS] Result {i+1} - Source: {os.path.basename(image_source) if isinstance(image_source, str) else 'embedded_image'}, "
                      f"Score: {score:.4f}, Type: {collection_type}")
                
                # Get filename for display
                if isinstance(image_source, str):
                    filename = os.path.basename(image_source)
                else:
                    filename = "embedded_image"
                
                # Load image
                if image_source in self.image_cache:
                    print(f"[STATUS] Using cached image for {filename}")
                    img = self.image_cache[image_source]
                else:
                    print(f"[STATUS] Loading image from {filename}")
                    try:
                        img = np.array(load_local_image(image_source))
                        self.image_cache[image_source] = img
                    except Exception as e:
                        print(f"[WARNING] Could not load image {filename}: {e}")
                        ax.text(0.5, 0.5, f"Error loading image", ha='center', va='center')
                        ax.axis('off')
                        continue

                # Display the result differently based on collection type
                if collection_type == "region":
                    # Extract and display region using bbox
                    bbox = metadata.get("bbox", [0, 0, 100, 100])
                    x_min, y_min, x_max, y_max = bbox
                    
                    # Ensure bbox is within image bounds
                    if x_min >= img.shape[1] or y_min >= img.shape[0]:
                        print(f"[WARNING] Invalid bbox for {filename}: {bbox}")
                        region = img  # Show whole image as fallback
                    else:
                        # Clip values to image bounds
                        x_min = max(0, min(x_min, img.shape[1]-1))
                        y_min = max(0, min(y_min, img.shape[0]-1))
                        x_max = max(x_min+1, min(x_max, img.shape[1]))
                        y_max = max(y_min+1, min(y_max, img.shape[0]))
                        
                        print(f"[STATUS] Extracting region from bbox: {[x_min, y_min, x_max, y_max]}")
                        region = img[y_min:y_max, x_min:x_max]
                    
                    # Display the region
                    ax.imshow(region)
                    
                    # Get phrase if available
                    phrase = metadata.get("phrase", "Region")
                    
                    # Create a badge showing it's a region
                    badge_props = dict(boxstyle="round,pad=0.3", fc="#e1f5fe", ec="#01579b", alpha=0.8)
                    badge_text = "REGION"
                    ax.text(0.04, 0.04, badge_text, transform=ax.transAxes, 
                            fontsize=9, weight='bold', color='#01579b',
                            verticalalignment='top', horizontalalignment='left',
                            bbox=badge_props)
                    
                    # Store region for possible enlargement
                    self.search_result_images.append({
                        "image": region,
                        "bbox": bbox,
                        "metadata": {
                            "image_source": image_source,
                            "filename": filename,
                            "phrase": phrase,
                            "score": score,
                            "type": "region",
                            "collection_name": collection_name
                        }
                    })
                    
                    # Create radio choice label
                    choice_label = f"Result {i+1}: {filename} - {phrase} (Region)"
                    
                    # Add to result info text
                    result_info.append(f"**Result {i+1}:** {filename}\n- Type: Region\n- Phrase: {phrase}\n- Score: {score:.4f}\n")
                    
                else:  # whole_image
                    # Display the whole image
                    ax.imshow(img)
                    
                    # Create a badge showing it's a whole image
                    badge_props = dict(boxstyle="round,pad=0.3", fc="#e8f5e9", ec="#2e7d32", alpha=0.8)
                    badge_text = "WHOLE IMAGE"
                    ax.text(0.04, 0.04, badge_text, transform=ax.transAxes, 
                            fontsize=9, weight='bold', color='#2e7d32',
                            verticalalignment='top', horizontalalignment='left',
                            bbox=badge_props)
                    
                    # Store whole image for possible enlargement
                    self.search_result_images.append({
                        "image": img,
                        "metadata": {
                            "image_source": image_source,
                            "filename": filename,
                            "phrase": "Whole Image",
                            "score": score,
                            "type": "whole_image",
                            "collection_name": collection_name
                        }
                    })
                    
                    # Create radio choice label
                    choice_label = f"Result {i+1}: {filename} (Whole Image)"
                    
                    # Add to result info text
                    result_info.append(f"**Result {i+1}:** {filename}\n- Type: Whole Image\n- Score: {score:.4f}\n")
                
                # Add to radio choices
                radio_choices.append(choice_label)
                
                # Create a clearer title with result number and score
                title = f"Result {i+1}: {score:.2f}"
                ax.set_title(title, fontsize=12, pad=5)
                
                # Add filename with better visibility and positioning
                ax.text(0.5, 0.03, f"{filename}", transform=ax.transAxes, 
                        ha='center', va='bottom', fontsize=10, 
                        bbox=dict(facecolor='white', alpha=0.8, pad=3,
                                  edgecolor='lightgray', boxstyle='round'))
                ax.axis('off')

            except Exception as e:
                print(f"[ERROR] Error displaying result {i+1}: {e}")
                import traceback
                traceback.print_exc()
                ax.text(0.5, 0.5, f"Error loading image {i+1}", ha='center', va='center')
                ax.axis('off')
                self.search_result_images.append(None)
                continue

        # Adjust layout to prevent overlapping and provide better spacing
        plt.tight_layout(pad=2.0)
        fig.subplots_adjust(wspace=0.3, hspace=0.3)

        print(f"[STATUS] Generating final visualization image...")
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150)
        buf.seek(0)
        plt.close(fig)

        results_image = Image.open(buf)
        print(f"[STATUS] Search results visualization complete with {len(radio_choices)} result options")
        return results_image, "\n".join(result_info), gr.update(visible=True), gr.update(choices=radio_choices, value=None)

    def display_enlarged_result(self, result_selection):
        """Displays an enlarged version of the selected search result"""
        if result_selection is None or not self.search_result_images:
            return None
            
        try:
            # Extract result index from selection
            result_idx = int(result_selection.split(":")[0].replace("Result ", "")) - 1
            
            if result_idx < 0 or result_idx >= len(self.search_result_images) or self.search_result_images[result_idx] is None:
                return None
                
            # Get the selected result data
            result_data = self.search_result_images[result_idx]
            
            # Check if result_data is a dictionary (expected format)
            if not isinstance(result_data, dict):
                print(f"[WARNING] Unexpected result_data type: {type(result_data)}")
                return None
            
            # Safely get image data
            if "image" not in result_data:
                print(f"[WARNING] No 'image' key in result_data: {list(result_data.keys())}")
                return None
                
            region = result_data["image"]
            
            # Safely handle metadata
            if "metadata" not in result_data:
                print(f"[WARNING] No 'metadata' key in result_data: {list(result_data.keys())}")
                # Create default metadata
                metadata = {
                    "filename": f"Result {result_idx+1}",
                    "phrase": "Unknown",
                    "score": 0.0
                }
            else:
                metadata = result_data["metadata"]
                
            # Get metadata values safely with defaults
            filename = metadata.get("filename", f"Result {result_idx+1}")
            phrase = metadata.get("phrase", "Unknown")
            score = metadata.get("score", 0.0)
            
            # Create a more visually appealing enlarged display
            fig = plt.figure(figsize=(10, 8), constrained_layout=True)
            gs = gridspec.GridSpec(1, 1, figure=fig)
            ax = fig.add_subplot(gs[0, 0])
            
            # Display the image
            ax.imshow(region)
            ax.axis('off')
            
            # Add a comprehensive, well-formatted title
            title = f"Result {result_idx+1}: {filename}"
            ax.set_title(title, fontsize=16, pad=15)
            
            # Add detailed metadata information in a visually appealing box
            info_text = (
                f"Phrase: {phrase}\n"
                f"Score: {score:.4f}" if isinstance(score, (float, int)) else f"Score: {score}"
            )
            
            # Create a text box with better styling
            props = dict(boxstyle='round,pad=1', facecolor='white', alpha=0.8, edgecolor='lightgray')
            ax.text(0.5, 0.03, info_text, transform=ax.transAxes, 
                    ha='center', va='bottom', fontsize=12,
                    bbox=props, linespacing=1.5)
            
            # Convert the figure to an image
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=200)  # Higher DPI for better quality
            buf.seek(0)
            plt.close(fig)
            
            return Image.open(buf)
            
        except Exception as e:
            print(f"[ERROR] Error displaying enlarged result: {e}")
            import traceback
            traceback.print_exc()
            return None

    def build_interface(self):
        """Build the Gradio interface"""
        # Create the interface with tabs for different modes
        with gr.Blocks(title="Grounded SAM Region Search") as demo:
            gr.Markdown("# 🔍 Grounded SAM Region Search")
            gr.Markdown("Upload an image, detect regions, and search for similar regions in your image database.")

            # Processing mode selector (new)
            gr.Markdown("## Processing Mode")
            with gr.Row():
                processing_mode = gr.Radio(
                    choices=["Region Detection (GroundedSAM + PE)", "Whole Image (PE Only)"],
                    value="Region Detection (GroundedSAM + PE)",
                    label="Select Processing Mode"
                )
            
            # Common file upload for both modes
            with gr.Row():
                input_image = gr.Image(type="pil", label="Upload Image")
            
            # Region-based mode components
            with gr.Group(visible=True) as region_mode_group:
                with gr.Row():
                    text_prompt = gr.Textbox(
                        placeholder="person, car, building, sign, text, animal, food",
                        label="Detection Prompts (comma-separated)",
                        value="person, car, building"
                    )
                
                with gr.Row():
                    process_button = gr.Button("Detect Regions", variant="primary")
            
            # Whole-image mode components
            with gr.Group(visible=False) as whole_image_mode_group:
                with gr.Row():
                    process_whole_button = gr.Button("Process Whole Image", variant="primary")
            
            # Status and results area
            with gr.Row():
                with gr.Column():
                    # Results for region-based mode
                    with gr.Group(visible=True) as region_results_group:
                        detection_info = gr.Textbox(label="Detection Status")
                        segmented_output = gr.Image(type="pil", label="Detected Regions")
                        region_dropdown = gr.Dropdown(choices=[], label="Select a Region")
                        region_preview = gr.Image(type="pil", label="Region Preview")
                        
                        with gr.Row():
                            with gr.Column():
                                similarity_slider = gr.Slider(
                                    minimum=0.0, maximum=1.0, value=0.5, step=0.01,
                                    label="Similarity Threshold"
                                )
                            with gr.Column():
                                max_results_dropdown = gr.Dropdown(
                                    choices=["5", "10", "20", "50"], value="5",
                                    label="Max Results"
                                )
                        
                        # Add layer selection for search
                        with gr.Row():
                            embedding_layer = gr.Slider(
                                minimum=1, maximum=50, value=40, step=1,
                                label="Embedding Layer",
                                info="Layer of the Perception Encoder to use for search (should match the database layer)"
                            )
                                
                        search_button = gr.Button("Search Similar Regions", variant="primary")
                    
                    # Results for whole-image mode
                    with gr.Group(visible=False) as whole_image_results_group:
                        whole_image_info = gr.Textbox(label="Processing Status")
                        processed_output = gr.Image(type="pil", label="Processed Image")
                        
                        with gr.Row():
                            with gr.Column():
                                whole_similarity_slider = gr.Slider(
                                    minimum=0.0, maximum=1.0, value=0.5, step=0.01,
                                    label="Similarity Threshold"
                                )
                            with gr.Column():
                                whole_max_results_dropdown = gr.Dropdown(
                                    choices=["5", "10", "20", "50"], value="5",
                                    label="Max Results"
                                )
                        
                        # Add layer selection for whole image search
                        with gr.Row():
                            whole_embedding_layer = gr.Slider(
                                minimum=1, maximum=50, value=40, step=1,
                                label="Embedding Layer",
                                info="Layer of the Perception Encoder to use for search (should match the database layer)"
                            )
                                
                        whole_search_button = gr.Button("Search Similar Images", variant="primary")
                    
                    # Common search results area
                    search_info = gr.Textbox(label="Search Status")
                    search_results_output = gr.Image(type="pil", label="Search Results")
                    
                    with gr.Group(visible=False) as button_section:
                        result_selector = gr.Dropdown(choices=[], label="Select Result to View")
                    
                    enlarged_result = gr.Image(type="pil", label="Enlarged Result")
            
            # Function to toggle visibility based on mode
            def toggle_mode(mode):
                if mode == "Region Detection (GroundedSAM + PE)":
                    return gr.update(visible=True), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
                else:  # Whole Image mode
                    return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=True)
            
            # Connect mode selector to toggle visibility
            processing_mode.change(
                toggle_mode,
                inputs=[processing_mode],
                outputs=[
                    region_mode_group, 
                    whole_image_mode_group,
                    region_results_group,
                    whole_image_results_group
                ]
            )
            
            # Connect buttons to functions for region-based processing
            process_button.click(
                self.process_image_with_prompt,
                inputs=[input_image, text_prompt],
                outputs=[segmented_output, detection_info, region_dropdown, region_preview]
            )

            region_dropdown.change(
                self.update_region_preview,
                inputs=[region_dropdown],
                outputs=[region_preview]
            )

            search_button.click(
                self.search_region,
                inputs=[region_dropdown, similarity_slider, max_results_dropdown, embedding_layer],
                outputs=[search_results_output, search_info, button_section, result_selector]
            )
            
            # Connect buttons to functions for whole-image processing
            process_whole_button.click(
                self.process_whole_image,
                inputs=[input_image, whole_embedding_layer],
                outputs=[processed_output, whole_image_info]
            )
            
            whole_search_button.click(
                self.search_whole_image,
                inputs=[whole_similarity_slider, whole_max_results_dropdown, whole_embedding_layer],
                outputs=[search_results_output, search_info, button_section, result_selector]
            )
            
            # Connect result selector for both modes
            result_selector.change(
                self.display_enlarged_result,
                inputs=[result_selector],
                outputs=[enlarged_result]
            )

        return demo

    def process_whole_image(self, image, optimal_layer=40):
        """Process an uploaded image using only PE Encoder without region detection"""
        if isinstance(image, np.ndarray):
            image_pil = Image.fromarray(image)
        else:
            image_pil = image

        print(f"[STATUS] Processing whole image with embedding layer: {optimal_layer}")
            
        # Extract whole image embedding
        image_np, embedding, metadata, label = extract_whole_image_embeddings(
            image_pil,
            pe_model_param=self.pe_model,
            pe_vit_model_param=self.pe_vit_model,
            preprocess_param=self.preprocess,
            device_param=self.device,
            optimal_layer=optimal_layer
        )

        # Store results
        self.whole_image = {
            "image": image_np,
            "embedding": embedding,
            "metadata": metadata,
            "label": label,
            "layer_used": optimal_layer
        }

        if embedding is None:
            return image_np, "Failed to process image"

        # Create a simple visualization with border
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(image_np)
        ax.set_title(f"Whole Image Processed (Layer {optimal_layer})")
        # Add a green border to show it's been processed
        border_width = 5
        plt.gca().spines['top'].set_linewidth(border_width)
        plt.gca().spines['bottom'].set_linewidth(border_width)
        plt.gca().spines['left'].set_linewidth(border_width)
        plt.gca().spines['right'].set_linewidth(border_width)
        plt.gca().spines['top'].set_color('green')
        plt.gca().spines['bottom'].set_color('green')
        plt.gca().spines['left'].set_color('green')
        plt.gca().spines['right'].set_color('green')
        ax.axis('on')  # Show axis for border visibility
        ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
        
        # Save figure to buffer
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)

        processed_image = Image.open(buf)
        return processed_image, f"Image processed successfully using layer {optimal_layer}"

    def update_region_preview(self, region_selection):
        """Updates the region preview when a region is selected"""
        if region_selection is None or self.detected_regions["image"] is None:
            return None

        region_idx = int(region_selection.split(":")[0].replace("Region ", "")) - 1
        mask = self.detected_regions["masks"][region_idx]
        label = self.detected_regions["labels"][region_idx]

        return self.create_region_preview(self.detected_regions["image"], mask, label)

# =============================================================================
# MULTI-MODE INTERFACE
# =============================================================================

def create_multi_mode_interface(pe_model, pe_vit_model, preprocess, device):
    """Create a multi-mode interface with both region-based and whole-image processing"""
    # Initialize the app state to track databases
    app_state = AppState()
    
    # Initialize the interface with the models
    interface = GradioInterface(pe_model, pe_vit_model, preprocess, device)
    
    # Create a comprehensive multi-mode Gradio interface with tabs
    with gr.Blocks(title="Grounded SAM Region Search") as demo:
        gr.Markdown("# 🔍 Grounded SAM Region Search")
        
        # Create mode tabs
        with gr.Tabs() as tabs:
            # Tab 1: Create Database
            with gr.TabItem("Create Database"):
                gr.Markdown("## Create a New Database")
                gr.Markdown("Process images in a folder to create a searchable database.")
                
                with gr.Row():
                    with gr.Column():
                        db_folder_path = gr.Textbox(label="Image Folder Path", placeholder="/path/to/images")
                        db_collection_name = gr.Textbox(label="Database Name", value="my_image_database", 
                                                      placeholder="Name for your database collection")
                        
                        db_mode = gr.Radio(
                            choices=["Region Detection (GroundedSAM + PE)", "Whole Image (PE Only)"],
                            value="Region Detection (GroundedSAM + PE)",
                            label="Database Processing Mode"
                        )
                        
                        # Region-specific options
                        with gr.Group(visible=True) as region_options_group:
                            region_prompts = gr.Textbox(
                                placeholder="person, car, building, sign, text, animal, food",
                                label="Detection Prompts (comma-separated)",
                                value="person, car, building"
                            )
                            
                        # Common options for both modes
                        with gr.Row():
                            with gr.Column():
                                layer_slider = gr.Slider(
                                    minimum=1, maximum=50, value=40, step=1,
                                    label="Embedding Layer",
                                    info="Layer of the Perception Encoder to use (1-50)"
                                )
                        
                        # Process button
                        create_db_button = gr.Button("Process Folder & Create Database", variant="primary")
                    
                    with gr.Column():
                        db_progress = gr.Textbox(label="Processing Status")
                        db_done_msg = gr.Markdown(visible=False)
                        
                # Function to toggle region options based on mode
                def toggle_region_options(mode):
                    if mode == "Region Detection (GroundedSAM + PE)":
                        return gr.update(visible=True)
                    else:
                        return gr.update(visible=False)
                
                db_mode.change(toggle_region_options, inputs=[db_mode], outputs=[region_options_group])
                
                # Connect build database functions
                def process_folder_with_mode(folder_path, collection_name, prompts, layer, mode):
                    """Process folder based on selected mode"""
                    print(f"[STATUS] Starting folder processing with mode: {mode}")
                    # Set up a generator function to handle progress updates
                    
                    if mode == "Region Detection (GroundedSAM + PE)":
                        # Modify collection name to include mode and layer
                        collection_name = f"{collection_name}_layer{layer}"
                        print(f"[STATUS] Using region collection name: {collection_name}")
                        # Process with region detection
                        gen = process_folder_with_progress_advanced(
                            folder_path, 
                            prompts, 
                            collection_name,
                            optimal_layer=layer,
                            min_area_ratio=0.005,
                            max_regions=5
                        )
                    else:
                        # Modify collection name to include mode and layer
                        collection_name = f"{collection_name}_whole_images_layer{layer}"
                        print(f"[STATUS] Using whole image collection name: {collection_name}")
                        # Process with whole image
                        gen = process_folder_with_progress_whole_images(
                            folder_path,
                            collection_name,
                            optimal_layer=layer
                        )
                    
                    # Return first message
                    result = next(gen)
                    
                    # Process remaining generator values 
                    for message, done_visible in gen:
                        yield message, gr.update(visible=done_visible)
                
                create_db_button.click(
                    process_folder_with_mode,
                    inputs=[db_folder_path, db_collection_name, region_prompts, layer_slider, db_mode],
                    outputs=[db_progress, db_done_msg]
                )
            
            # Tab 2: Search Database
            with gr.TabItem("Search Database"):
                gr.Markdown("## Search Database")
                gr.Markdown("Upload an image, detect regions, and search for similar regions in your database.")
                
                # Database selection
                with gr.Row():
                    with gr.Column():
                        # Refresh button for database list
                        refresh_db_button = gr.Button("Refresh Database List")
                    with gr.Column():
                        db_dropdown = gr.Dropdown(
                            choices=app_state.available_databases if app_state.available_databases else ["No databases found"],
                            value=app_state.available_databases[0] if app_state.available_databases else None,
                            label="Select Database"
                        )
                
                # Processing mode selector
                with gr.Row():
                    processing_mode = gr.Radio(
                        choices=["Region Detection (GroundedSAM + PE)", "Whole Image (PE Only)"],
                        value="Region Detection (GroundedSAM + PE)",
                        label="Select Processing Mode"
                    )
                
                # Common file upload for both modes
                with gr.Row():
                    input_image = gr.Image(type="pil", label="Upload Image")
                
                # Region-based mode components
                with gr.Group(visible=True) as region_mode_group:
                    with gr.Row():
                        text_prompt = gr.Textbox(
                            placeholder="person, car, building, sign, text, animal, food",
                            label="Detection Prompts (comma-separated)",
                            value="person, car, building"
                        )
                    
                    with gr.Row():
                        process_button = gr.Button("Detect Regions", variant="primary")
                
                # Whole-image mode components
                with gr.Group(visible=False) as whole_image_mode_group:
                    with gr.Row():
                        process_whole_button = gr.Button("Process Whole Image", variant="primary")
                
                # Status and results area
                with gr.Row():
                    with gr.Column():
                        # Results for region-based mode
                        with gr.Group(visible=True) as region_results_group:
                            detection_info = gr.Textbox(label="Detection Status")
                            segmented_output = gr.Image(type="pil", label="Detected Regions")
                            region_dropdown = gr.Dropdown(choices=[], label="Select a Region")
                            region_preview = gr.Image(type="pil", label="Region Preview")
                            
                            with gr.Row():
                                with gr.Column():
                                    similarity_slider = gr.Slider(
                                        minimum=0.0, maximum=1.0, value=0.5, step=0.01,
                                        label="Similarity Threshold"
                                    )
                                with gr.Column():
                                    max_results_dropdown = gr.Dropdown(
                                        choices=["5", "10", "20", "50"], value="5",
                                        label="Max Results"
                                    )
                            
                            # Add layer selection for search
                            with gr.Row():
                                embedding_layer = gr.Slider(
                                    minimum=1, maximum=50, value=40, step=1,
                                    label="Embedding Layer",
                                    info="Layer of the Perception Encoder to use for search (should match the database layer)"
                                )
                                    
                            search_button = gr.Button("Search Similar Regions", variant="primary")
                        
                        # Results for whole-image mode
                        with gr.Group(visible=False) as whole_image_results_group:
                            whole_image_info = gr.Textbox(label="Processing Status")
                            processed_output = gr.Image(type="pil", label="Processed Image")
                            
                            with gr.Row():
                                with gr.Column():
                                    whole_similarity_slider = gr.Slider(
                                        minimum=0.0, maximum=1.0, value=0.5, step=0.01,
                                        label="Similarity Threshold"
                                    )
                                with gr.Column():
                                    whole_max_results_dropdown = gr.Dropdown(
                                        choices=["5", "10", "20", "50"], value="5",
                                        label="Max Results"
                                    )
                            
                            # Add layer selection for whole image search
                            with gr.Row():
                                whole_embedding_layer = gr.Slider(
                                    minimum=1, maximum=50, value=40, step=1,
                                    label="Embedding Layer",
                                    info="Layer of the Perception Encoder to use for search (should match the database layer)"
                                )
                                    
                            whole_search_button = gr.Button("Search Similar Images", variant="primary")
                        
                        # Common search results area
                        search_info = gr.Textbox(label="Search Status")
                        search_results_output = gr.Image(type="pil", label="Search Results")
                        
                        with gr.Group(visible=False) as button_section:
                            result_selector = gr.Dropdown(choices=[], label="Select Result to View")
                        
                        enlarged_result = gr.Image(type="pil", label="Enlarged Result")
                
                # Function to toggle visibility based on mode
                def toggle_mode(mode):
                    if mode == "Region Detection (GroundedSAM + PE)":
                        return gr.update(visible=True), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
                    else:  # Whole Image mode
                        return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=True)
                
                # Connect mode selector to toggle visibility
                processing_mode.change(
                    toggle_mode,
                    inputs=[processing_mode],
                    outputs=[
                        region_mode_group, 
                        whole_image_mode_group,
                        region_results_group,
                        whole_image_results_group
                    ]
                )
                
                # Function to refresh database list
                def refresh_databases():
                    app_state.update_available_databases(verbose=True)
                    if app_state.available_databases:
                        print(f"[STATUS] Updated dropdown with {len(app_state.available_databases)} databases")
                        return gr.Dropdown(choices=app_state.available_databases, value=app_state.available_databases[0])
                    else:
                        print(f"[STATUS] No databases found to display in dropdown")
                        return gr.Dropdown(choices=["No databases found"], value=None)
                
                # Connect refresh button
                refresh_db_button.click(refresh_databases, inputs=[], outputs=[db_dropdown])
                
                # Function to set active database
                def set_active_database(collection_name):
                    if collection_name and collection_name != "No databases found":
                        interface.set_active_collection(collection_name)
                        return f"Set active database to: {collection_name}"
                    return "No database selected"
                
                # Connect database dropdown
                db_dropdown.change(set_active_database, inputs=[db_dropdown], outputs=[detection_info])
                
                # Connect buttons to functions for region-based processing
                process_button.click(
                    interface.process_image_with_prompt,
                    inputs=[input_image, text_prompt],
                    outputs=[segmented_output, detection_info, region_dropdown, region_preview]
                )

                region_dropdown.change(
                    interface.update_region_preview,
                    inputs=[region_dropdown],
                    outputs=[region_preview]
                )

                search_button.click(
                    interface.search_region,
                    inputs=[region_dropdown, similarity_slider, max_results_dropdown, embedding_layer],
                    outputs=[search_results_output, search_info, button_section, result_selector]
                )
                
                # Connect buttons to functions for whole-image processing
                process_whole_button.click(
                    interface.process_whole_image,
                    inputs=[input_image, whole_embedding_layer],
                    outputs=[processed_output, whole_image_info]
                )
                
                whole_search_button.click(
                    interface.search_whole_image,
                    inputs=[whole_similarity_slider, whole_max_results_dropdown, whole_embedding_layer],
                    outputs=[search_results_output, search_info, button_section, result_selector]
                )
                
                # Connect result selector for both modes
                result_selector.change(
                    interface.display_enlarged_result,
                    inputs=[result_selector],
                    outputs=[enlarged_result]
                )
            
            # Tab 3: About
            with gr.TabItem("About"):
                gr.Markdown("""
                # Grounded SAM Region Search
                
                This application combines GroundedSAM for region detection and Perception Encoder (PE) for embedding generation to create a powerful semantic image search system.
                
                ## Features
                
                - **Region Detection Mode**: Extract meaningful regions from images using GroundedSAM, then embed them with PE
                - **Whole Image Mode**: Process entire images directly with PE for faster processing
                - **Cross-Compatible Search**: Search across both region and whole image databases
                - **Flexible Database Management**: Create and search multiple databases
                
                ## How to Use
                
                1. **Create Database tab**: Process folders of images to build your search database
                2. **Search Database tab**: Upload and process images to find similar content in your database
                
                ## Technical Details
                
                - Built with GroundedSAM, Perception Encoder, and Qdrant vector database
                - Supports various semantic layer depths for different types of features
                - Cross-compatible between region detection and whole-image processing modes
                """)
    
    # Set the default active database if available
    if app_state.available_databases:
        interface.set_active_collection(app_state.available_databases[0])
    
    return demo

# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    # Parse command line arguments for backward compatibility
    parser = argparse.ArgumentParser(description="Grounded SAM Region Search Application")
    parser.add_argument("--build-database", action="store_true", help="Build database from images folder")
    parser.add_argument("--interface", action="store_true", help="Launch Gradio interface")
    parser.add_argument("--all", action="store_true", help="Build database and launch interface")
    parser.add_argument("--folder", default="./my_images", help="Folder containing images to process")
    parser.add_argument("--prompts", default="person, building, car, text, object", help="Text prompts for detection")
    parser.add_argument("--collection", default="grounded_image_regions", help="Qdrant collection name")
    parser.add_argument("--new-interface", action="store_true", help="Use the new multi-mode interface (deprecated, now the default)")
    parser.add_argument("--legacy-interface", action="store_true", help="Use the legacy single-mode interface")
    # Add mode option
    parser.add_argument("--mode", choices=["region", "whole_image"], default="region", 
                        help="Processing mode: 'region' for GroundedSAM+PE or 'whole_image' for PE only")
    parser.add_argument("--optimal-layer", type=int, default=40, help="Optimal layer for PE model (default: 40)")
    # Add port option
    parser.add_argument("--port", type=int, default=7860, help="Port to use for the Gradio interface")

    args = parser.parse_args()

    # Setup device and load models
    print("🚀 Initializing Grounded SAM Region Search Application")
    print("=" * 50)

    # Setup device and declare globals
    global device, pe_model, pe_vit_model, preprocess
    device = setup_device()

    # Load PE models
    print("\n🧠 Loading Perception Encoder models...")
    try:
        pe_model, pe_vit_model, preprocess = load_pe_model(device)
        print("✅ Models loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load models: {e}")
        print("Make sure you've run setup.py first!")
        return

    # Use multi-mode interface by default unless legacy mode is explicitly requested
    # This ensures both build database and search functionality are always available
    if not args.legacy_interface:
        print("\n🌐 Launching multi-mode interface with both build and search functionality...")
        demo = create_multi_mode_interface(pe_model, pe_vit_model, preprocess, device)
        print("🚀 Interface ready! Opening in browser...")
        
        # Try to launch with different ports to avoid port conflicts
        import socket
        import os
        
        # First attempt to kill any existing process on the port
        os.system(f"lsof -ti:{args.port} | xargs kill -9 2>/dev/null || true")
        
        # Function to check if a port is in use
        def is_port_in_use(port):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                return s.connect_ex(('localhost', port)) == 0
        
        # Try ports sequentially
        max_port = args.port + 10
        current_port = args.port
        
        while current_port <= max_port:
            if not is_port_in_use(current_port):
                try:
                    print(f"Trying to start server on port {current_port}...")
                    demo.launch(share=False, server_name="127.0.0.1", server_port=current_port)
                    return  # If successful, exit the function
                except OSError:
                    print(f"Failed to start on port {current_port}")
            current_port += 1
        
        # If all ports failed, try without specifying a port
        print("All specified ports are in use. Letting Gradio choose an available port...")
        demo.launch(share=False)
        return

    # Legacy command-line mode (only if explicitly requested)
    if args.legacy_interface and not any([args.build_database, args.interface, args.all]):
        parser.print_help()
        return

    # Build database if requested (legacy mode)
    if args.build_database or args.all:
        print(f"\n📁 Building database from folder: {args.folder}")
        
        if not os.path.exists(args.folder):
            print(f"❌ Error: Folder not found at {args.folder}")
            print(f"Please create the folder and add images, or specify a different folder with --folder")
            return

        # Choose processing mode based on argument
        if args.mode == "whole_image":
            print(f"📊 Processing Mode: Whole Image (PE Only)")
            client = process_folder_for_whole_image_database(
                folder_path=args.folder,
                collection_name=args.collection,
                pe_model=pe_model,
                pe_vit_model=pe_vit_model,
                preprocess=preprocess,
                device=device,
                optimal_layer=args.optimal_layer,
                checkpoint_interval=3,
                resume_from_checkpoint=True
            )
        else:  # Default to region mode
            print(f"📊 Processing Mode: Region Detection (GroundedSAM + PE)")
            client = process_folder_for_region_database(
                folder_path=args.folder,
                collection_name=args.collection,
                text_prompts=args.prompts,
                pe_model=pe_model,
                pe_vit_model=pe_vit_model,
                preprocess=preprocess,
                device=device,
                optimal_layer=args.optimal_layer,
                min_area_ratio=0.005,
                max_regions=5,
                visualize_samples=False,  # Disable for non-interactive mode
                sample_count=2,
                checkpoint_interval=3,
                resume_from_checkpoint=True
            )

        if client is not None:
            print("✅ Database built successfully!")
            client.close()
        else:
            print("❌ Failed to build database")
            if not args.interface:
                return

    # Launch interface if requested (legacy mode)
    if args.interface or args.all:
        print("\n🌐 Launching legacy Gradio interface...")
        
        interface = GradioInterface(pe_model, pe_vit_model, preprocess, device)
        demo = interface.build_interface()
        
        print("🚀 Interface ready! Opening in browser...")
        
        # Try to launch with different ports to avoid port conflicts
        import socket
        import os
        
        # First attempt to kill any existing process on the port
        os.system(f"lsof -ti:{args.port} | xargs kill -9 2>/dev/null || true")
        
        # Function to check if a port is in use
        def is_port_in_use(port):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                return s.connect_ex(('localhost', port)) == 0
        
        # Try ports sequentially
        max_port = args.port + 10
        current_port = args.port
        
        while current_port <= max_port:
            if not is_port_in_use(current_port):
                try:
                    print(f"Trying to start server on port {current_port}...")
                    demo.launch(share=False, server_name="127.0.0.1", server_port=current_port)
                    return  # If successful, exit the function
                except OSError:
                    print(f"Failed to start on port {current_port}")
            current_port += 1
        
        # If all ports failed, try without specifying a port
        print("All specified ports are in use. Letting Gradio choose an available port...")
        demo.launch(share=False)

if __name__ == "__main__":
    main()