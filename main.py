#!/usr/bin/env python3
"""
Revers-o: GroundedSAM + Perception Encoders Image Similarity Search Application
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
import tempfile
from contextlib import nullcontext
from pathlib import Path
from typing import List, Optional

# Core imports
import cv2
import numpy as np
import torch
import torchvision # Added for NMS
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
    collections = []
    
    # Always use the Qdrant API as the primary source of truth
    qdrant_data_dir = os.path.join("image_retrieval_project", "qdrant_data")
    if os.path.exists(qdrant_data_dir):
        try:
            # Connect to Qdrant and get collections directly
            client = QdrantClient(path=qdrant_data_dir)
            collection_list = client.get_collections().collections
            collections = [c.name for c in collection_list]
            client.close()
            print(f"[DEBUG] Found collections via Qdrant API: {collections}")
        except Exception as e:
            print(f"[DEBUG] Error querying Qdrant API: {e}")
            
            # Fallback to filesystem checks if API fails
            # Path 1: Check the 'collection' directory (singular)
            collection_dir = os.path.join(qdrant_data_dir, "collection")
            if os.path.exists(collection_dir):
                collections.extend([d for d in os.listdir(collection_dir) if os.path.isdir(os.path.join(collection_dir, d))])
                print(f"[DEBUG] Found collections in 'collection' directory: {collections}")
            
            # Path 2: Check the 'collections' directory (plural)
            collections_dir = os.path.join(qdrant_data_dir, "collections")
            if os.path.exists(collections_dir):
                plural_collections = [d for d in os.listdir(collections_dir) if os.path.isdir(os.path.join(collections_dir, d))]
                collections.extend(plural_collections)
                print(f"[DEBUG] Found collections in 'collections' directory: {plural_collections}")
    
    # Remove duplicates while preserving order
    unique_collections = []
    for c in collections:
        if c not in unique_collections:
            unique_collections.append(c)
    
    # Filter out any collections that start with "." (hidden folders)
    unique_collections = [c for c in unique_collections if not c.startswith(".")]
    
    if not unique_collections:
        print(f"[STATUS] No database collections found")
    else:
        print(f"[STATUS] Available collections: {unique_collections}")
    
    return unique_collections
    
# Add the new function to delete a database collection
def delete_database_collection(collection_name):
    """Properly delete a database collection using the Qdrant API"""
    if not collection_name or collection_name == "No databases found":
        return False, "No valid collection specified"
        
    try:
        print(f"[STATUS] Attempting to delete collection: {collection_name}")
        qdrant_data_dir = os.path.join("image_retrieval_project", "qdrant_data")
        
        # Connect to Qdrant
        client = QdrantClient(path=qdrant_data_dir)
        
        # Check if collection exists
        collections = client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if collection_name not in collection_names:
            client.close()
            return False, f"Collection '{collection_name}' does not exist"
            
        # Delete the collection
        client.delete_collection(collection_name=collection_name)
        client.close()
        
        print(f"[STATUS] Successfully deleted collection: {collection_name}")
        return True, f"Successfully deleted database: {collection_name}"
    except Exception as e:
        print(f"[ERROR] Failed to delete collection {collection_name}: {e}")
        import traceback
        traceback.print_exc()
        return False, f"Error deleting database: {str(e)}"

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
            
    except Exception as e:
        print(f"[DEBUG] Error accessing {property_name} at index {index}: {e}")
        
    return default

def extract_region_embeddings_autodistill(
    image_source,
    text_prompt,
    pe_model_param=None,
    pe_vit_model_param=None,
    preprocess_param=None,
    device_param=None,
    min_area_ratio=0.01, # Default min_area_ratio
    max_regions=10,
    is_url=False,
    max_image_size=800,
    optimal_layer=40,
    pooling_strategy="top_k"
):
    """
    Simplified region detection and embedding extraction using GroundedSAM and Perception Encoder.
    Based on simpler GroundedSAM examples.
    
    Returns:
        tuple: (image_np, masks, embeddings, metadata, labels, error_message)
        - If successful: (image_np, masks, embeddings, metadata, labels, None)
        - If no regions found: (image_np, [], [], [], [], None) 
        - If error occurred: (None, [], [], [], [], error_message)
    """

    # Kept Helper Functions
    def apply_spatial_pooling(masked_features, strategy="top_k", top_k_ratio=0.1):
        """
        Apply different pooling strategies to masked features.
        """
        if strategy == "max":
            pooled = masked_features.max(dim=0, keepdim=True)[0]
        elif strategy == "top_k":
            feature_norms = masked_features.norm(dim=1)
            k = max(1, int(top_k_ratio * len(feature_norms)))
            top_k_indices = torch.topk(feature_norms, k)[1]
            pooled = masked_features[top_k_indices].mean(dim=0, keepdim=True)
        elif strategy == "attention":
            feature_norms = masked_features.norm(dim=1)
            attention_weights = torch.softmax(feature_norms, dim=0)
            pooled = (masked_features * attention_weights.unsqueeze(1)).sum(dim=0, keepdim=True)
        else:  # "average" or fallback
            pooled = masked_features.mean(dim=0, keepdim=True)
        return pooled

    def get_detected_class(detections_obj, det_index, class_names_from_ontology):
        # detections_obj is the Detections object from GroundedSAM
        # det_index is the index of the current detection
        # class_names_from_ontology is the list like ['person', 'rocks'] from ontology.classes()

        if hasattr(detections_obj, 'class_id') and \
           detections_obj.class_id is not None and \
           len(detections_obj.class_id) > det_index:
            
            class_id_val = detections_obj.class_id[det_index]

            if isinstance(class_id_val, torch.Tensor):
                class_id_val = class_id_val.item()

            # Corrected type check to include numpy integers
            if isinstance(class_id_val, (int, np.integer)) and 0 <= class_id_val < len(class_names_from_ontology):
                # This is the primary, expected path
                detected_class_name = class_names_from_ontology[class_id_val]
                print(f"[DEBUG get_class] Using class_id {class_id_val} (type {type(class_id_val)}) to get name '{detected_class_name}' from ontology classes list.")
                return detected_class_name
            else:
                print(f"[DEBUG get_class] class_id_val {class_id_val} (type {type(class_id_val)}) is not a valid index for ontology classes list (len {len(class_names_from_ontology)}).")
        else:
            print(f"[DEBUG get_class] detections_obj.class_id not found or invalid for index {det_index}.")

        # Fallback if the above fails (should ideally not be reached if GroundedSAM and ontology work as expected)
        # If there's only one class in the ontology, it's a safe bet. Otherwise, it's a guess.
        default_class = class_names_from_ontology[0] if class_names_from_ontology else "unknown"
        print(f"[DEBUG get_class] Fallback: returning '{default_class}'")
        return default_class


    def cleanup_temp_resources(temp_file_paths_list=None):
        """Clean up temporary resources, including GPU/MPS memory if applicable."""
        if temp_file_paths_list is None:
            temp_file_paths_list = []
        for temp_file in temp_file_paths_list:
            if temp_file and os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                    print(f"[Cleanup] Removed temp file: {temp_file}")
                except Exception as e_file_remove:
                    print(f"[WARN Cleanup] Failed to remove temp file {temp_file}: {e_file_remove}")
        
        # Explicitly call garbage collector
        gc.collect()
        print(f"[Cleanup] Garbage collection called.")
        
        # Clear CUDA cache if available and in use
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"[Cleanup] Cleared CUDA cache.")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            try:
                torch.mps.empty_cache() # For MPS devices
                print(f"[Cleanup] Cleared MPS cache.")
            except Exception as e_mps_cache:
                print(f"[WARN Cleanup] Error clearing MPS cache: {e_mps_cache}")
    
    # Access global models if needed
    global pe_model, pe_vit_model, preprocess, device
    
    pe_model_to_use = pe_model_param if pe_model_param is not None else pe_model
    pe_vit_model_to_use = pe_vit_model_param if pe_vit_model_param is not None else pe_vit_model
    preprocess_to_use = preprocess_param if preprocess_param is not None else preprocess
    device_to_use = device_param if device_param is not None else device
    
    detections_obj = None 
    embeddings, metadata_list, labels = [], [], []
    temp_image_path = None # Ensure temp_image_path is defined for finally

    try:
        print(f"[STATUS] Starting simplified region extraction process...")
        
        # Load and prepare image
        if is_url:
            pil_image = download_image(image_source)
        else:
            if isinstance(image_source, str):
                print(f"[STATUS] Loading image from path: {os.path.basename(image_source)}")
                pil_image = load_local_image(image_source)
            else:
                pil_image = image_source # Assuming it's already a PIL image

        # Resize image if needed for optimal processing
        original_size = pil_image.size
        if max(pil_image.size) > max_image_size:
            pil_image.thumbnail((max_image_size, max_image_size), Image.Resampling.LANCZOS)
            print(f"[STATUS] Resized image from {original_size} to {pil_image.size}")

        image_np = np.array(pil_image)
        image_area = image_np.shape[0] * image_np.shape[1]

        # Create temporary file for GroundedSAM
        temp_dir = tempfile.gettempdir()
        unique_id = str(uuid.uuid4())[:8]
        
        if isinstance(image_source, str) and os.path.isfile(image_source):
            base_filename = os.path.basename(image_source)
        else:
            base_filename = f"uploaded_image_{unique_id}.jpg" # Handle PIL image inputs
            
        temp_image_path = os.path.join(temp_dir, f"gsam_temp_{unique_id}_{base_filename}")
        pil_image.save(temp_image_path, "JPEG", quality=95)
        print(f"[STATUS] Saved temporary image for GroundedSAM: {temp_image_path}")


        prompts = [p.strip() for p in text_prompt.split('.') if p.strip()]
        if not prompts:
            prompts = ["object", "region"] # Default prompts if none provided
            print(f"[WARN] No text prompt provided. Using default prompts: {prompts}")

        print(f"[DEBUG_PROMPT] Initial text_prompt string: '{text_prompt}'")
        print(f"[DEBUG_PROMPT] Derived prompts list: {prompts}")

        # Improved prompt formatting for GroundedSAM
        # Based on official documentation: period-separated format works best for multiple objects
        # Natural language descriptions work well for single descriptive phrases
        if len(prompts) == 1 and len(prompts[0].split()) > 2:
            # Single descriptive phrase - keep natural language format
            formatted_prompt = prompts[0]
            print(f"[DEBUG_PROMPT] Using natural language format: '{formatted_prompt}'")
        else:
            # Multiple objects or simple terms - use period-separated format
            # Since input is already period-separated, just ensure proper formatting
            formatted_prompt = " . ".join(prompts) + " ."
            print(f"[DEBUG_PROMPT] Using period-separated format: '{formatted_prompt}'")

        ontology_dict = {prompt: prompt for prompt in prompts}
        print(f"[DEBUG_PROMPT] Created ontology_dict: {ontology_dict}")
        
        from autodistill_grounded_sam import GroundedSAM # Ensure this import is here
        from autodistill.detection import CaptionOntology
        
        ontology = CaptionOntology(ontology_dict)
        print(f"[DEBUG_PROMPT] Created ontology with classes: {ontology.classes()}")
        
        grounded_sam = GroundedSAM(
            ontology=ontology,
            box_threshold=0.35,
            text_threshold=0.25
        )
        print(f"[STATUS] Initialized GroundedSAM with box_threshold=0.35, text_threshold=0.25")

        # Use the formatted prompt for detection
        detections = grounded_sam.predict(temp_image_path)
        print(f"[STATUS] GroundedSAM detected {len(detections)} regions using prompt: '{formatted_prompt}'")
        
        # Get the class names list directly from the ontology object used by GroundedSAM
        ontology_class_names = ontology.classes()
        print(f"[DEBUG_PROMPT] ontology.classes() returned: {ontology_class_names}")
        print(f"[DEBUG_PROMPT] Type of ontology_class_names: {type(ontology_class_names)}")
        if isinstance(ontology_class_names, list) and ontology_class_names:
            print(f"[DEBUG_PROMPT] Type of first element in ontology_class_names: {type(ontology_class_names[0])}")


        if len(detections) == 0:
            print(f"[STATUS] No detections found by GroundedSAM.")
            # Call cleanup even if no detections, to remove temp image
            cleanup_temp_resources(temp_file_paths_list=[temp_image_path])
            return image_np if image_np is not None else None, [], [], [], [], None


        # Process detections (simplified)
        masks_cpu_np, metadata_list, labels = [], [], [] # Store CPU/NumPy masks for PE
        
        for i in range(min(len(detections), max_regions)):
            try:
                mask_tensor = detections.mask[i]
                raw_confidence = 0.0 # Default

                # --- Start of detailed logging for this detection ---
                print(f"\\n--- [DEBUG_DETECTION {i}] ---")
                
                # Log class_id
                if hasattr(detections, 'class_id') and detections.class_id is not None and len(detections.class_id) > i:
                    raw_class_id = detections.class_id[i]
                    print(f"[DEBUG_DETECTION {i}] Raw detections.class_id[{i}]: {raw_class_id} (Type: {type(raw_class_id)})")
                    if isinstance(raw_class_id, torch.Tensor):
                        print(f"[DEBUG_DETECTION {i}] Raw class_id tensor details: device={raw_class_id.device}, dtype={raw_class_id.dtype}")
                else:
                    print(f"[DEBUG_DETECTION {i}] detections.class_id not available or index out of bounds.")

                # Log confidence
                if hasattr(detections, 'confidence') and detections.confidence is not None and len(detections.confidence) > i:
                    confidence_val = detections.confidence[i]
                    print(f"[DEBUG_DETECTION {i}] Raw detections.confidence[{i}]: {confidence_val} (Type: {type(confidence_val)})")
                    if isinstance(confidence_val, torch.Tensor):
                        raw_confidence = confidence_val.item()
                    elif isinstance(confidence_val, np.ndarray):
                        raw_confidence = float(confidence_val)
                    else:
                        raw_confidence = float(confidence_val)
                else:
                     print(f"[DEBUG_DETECTION {i}] detections.confidence not available or index out of bounds.")


                # Log other potentially useful attributes from detections object
                if hasattr(detections, 'data') and isinstance(detections.data, dict):
                    print(f"[DEBUG_DETECTION {i}] detections.data keys: {list(detections.data.keys())}")
                    for key, value_list in detections.data.items():
                        if isinstance(value_list, (list, np.ndarray)) and len(value_list) > i:
                            item_value = value_list[i]
                            print(f"[DEBUG_DETECTION {i}] detections.data['{key}'][{i}]: {item_value} (Type: {type(item_value)})")
                        elif not isinstance(value_list, (list, np.ndarray)):
                             print(f"[DEBUG_DETECTION {i}] detections.data['{key}'] is not a list/array: {value_list} (Type: {type(value_list)})")


                # Call get_detected_class (which also has internal prints)
                detected_class = get_detected_class(detections, i, ontology_class_names)
                print(f"[DEBUG_DETECTION {i}] Class determined by get_detected_class: '{detected_class}'")
                # --- End of detailed logging for this detection ---

                print(f"[STATUS] Processing detection {i+1}/{min(len(detections), max_regions)}: {detected_class} (Raw Conf: {raw_confidence:.3f})")
                
                # Simplified Validation (Area and non-empty)
                if isinstance(mask_tensor, torch.Tensor):
                    mask_np_cpu = mask_tensor.detach().cpu().numpy() # Ensure mask is on CPU and NumPy
                elif isinstance(mask_tensor, np.ndarray):
                    mask_np_cpu = mask_tensor # Already a NumPy array
                else:
                    print(f"[WARN] Mask tensor for detection {i} is of unexpected type: {type(mask_tensor)}, attempting to convert.")
                    try:
                        mask_np_cpu = np.array(mask_tensor) # Fallback conversion
                    except Exception as e_conv:
                        print(f"[ERROR] Could not convert mask_tensor to NumPy array for detection {i}: {e_conv}")
                        continue # Skip this detection if conversion fails
                
                # Binarize if float, ensure uint8
                if np.issubdtype(mask_np_cpu.dtype, np.floating):
                    mask_processed_cv = (mask_np_cpu > 0.5).astype(np.uint8)
                elif mask_np_cpu.dtype == bool:
                    mask_processed_cv = mask_np_cpu.astype(np.uint8)
                else:
                    mask_processed_cv = mask_np_cpu.astype(np.uint8)

                if np.sum(mask_processed_cv) == 0:
                    print(f"[STATUS] Skipping detection {i+1} ({detected_class}) - Empty mask after processing.")
                    continue

                current_area_pixels = np.sum(mask_processed_cv)
                current_area_ratio = current_area_pixels / image_area if image_area > 0 else 0
                
                # Apply min_area_ratio filter (parameter from function call)
                if min_area_ratio > 0 and current_area_ratio < min_area_ratio:
                    print(f"[STATUS] Skipping detection {i+1} ({detected_class}) - Area ratio {current_area_ratio:.4f} < min_area_ratio {min_area_ratio:.4f}")
                    continue
                
                # Store the processed CPU/NumPy mask for PE
                masks_cpu_np.append(mask_processed_cv) 

                y_indices, x_indices = np.where(mask_processed_cv)
                x_min, x_max = int(x_indices.min()), int(x_indices.max())
                y_min, y_max = int(y_indices.min()), int(y_indices.max())
                
                metadata = {
                    "region_id": str(uuid.uuid4()),
                    "image_source": str(image_source) if isinstance(image_source, str) else "uploaded_image",
                    "filename": base_filename,
                    "bbox": [x_min, y_min, x_max, y_max],
                    "area_ratio": float(current_area_ratio),
                    "detected_class": detected_class,
                    "confidence": float(raw_confidence), # Using raw confidence
                    "type": "region",
                    "detection_method": "grounded_sam_simplified",
                    "layer_used": optimal_layer
                }
                label_text = f"{detected_class} ({raw_confidence:.2f})"
                metadata_list.append(metadata)
                labels.append(label_text)
                print(f"[STATUS] Successfully processed detection {i+1}: {label_text}")

            except Exception as e_det_loop:
                print(f"[ERROR] Error processing detection {i} in simplified loop: {e_det_loop}")
                import traceback
                traceback.print_exc()
                continue
        
        if not masks_cpu_np: # Check if any valid masks were collected
            print(f"[WARNING] No valid regions found after simplified processing.")
            cleanup_temp_resources(temp_file_paths_list=[temp_image_path])
            return image_np if image_np is not None else None, [], [], [], [], None
            
        print(f"[STATUS] Processing whole image with PE for spatial feature extraction using {len(masks_cpu_np)} masks...")
        
        embeddings = [] # Re-initialize embeddings here
        with torch.no_grad():
            pe_input = preprocess_to_use(pil_image).unsqueeze(0).to(device_to_use)
            
            try:
                if pe_vit_model_to_use is not None:
                    intermediate_features = pe_vit_model_to_use.forward_features(pe_input, layer_idx=max(1, optimal_layer))
                    print(f"[STATUS] Successfully extracted intermediate features from layer {max(1, optimal_layer)} using VisionTransformer")
                else:
                    intermediate_features = pe_model_to_use.encode_image(pe_input) # This will be a single vector
                    print(f"[STATUS] Using fallback PE model (single vector) - spatial masking will select this vector if mask is non-empty.")
                    
            except Exception as e_feat:
                print(f"[ERROR] Failed to extract PE features: {e_feat}")
                cleanup_temp_resources(temp_file_paths_list=[temp_image_path])
                return None, [], [], [], [], f"Failed to extract PE features: {e_feat}"
            
            # If intermediate_features is a single vector (B, D), reshape to (B, 1, 1, D) for consistency if spatial pooling is expected
            if pe_vit_model_to_use is None and len(intermediate_features.shape) == 2: # (B, D)
                 # Make it look like a 1x1 spatial grid for compatibility with pooling logic
                intermediate_features = intermediate_features.unsqueeze(1).unsqueeze(1) 
            
            # Convert features to spatial grid format if it's not already (B,H,W,D)
            # tokens_to_grid expects (B, N+1, D) or (B, N, D)
            if len(intermediate_features.shape) == 3: # (B, N_tokens, D)
                spatial_features = tokens_to_grid(intermediate_features, pe_model_to_use)
            elif len(intermediate_features.shape) == 4: # Already (B, H, W, D)
                spatial_features = intermediate_features
            else: # Fallback for single vector after unsqueezing
                spatial_features = intermediate_features # Should be (B, 1, 1, D)

            B, H, W, D_feat = spatial_features.shape # Renamed D to D_feat to avoid conflict
            print(f"[STATUS] Feature grid dimensions: {H}x{W}, embedding dimension: {D_feat}")

            spatial_features = spatial_features.to(device_to_use)
            pe_model_to_use = pe_model_to_use.to(device_to_use) # Ensure model is on device

            # Extract region-specific embeddings using spatial masking
            # We iterate over masks_cpu_np which are already NumPy arrays on CPU
            temp_embeddings = [] # Store embeddings before final checks
            temp_metadata = []
            temp_labels = []
            final_masks_for_output = []


            for i, mask_cv_np in enumerate(masks_cpu_np): # mask_cv_np is from masks_cpu_np
                try:
                    # Resize mask to match feature map size
                    # mask_cv_np is already a processed uint8 NumPy array
                    mask_resized_np = cv2.resize(mask_cv_np, (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)

                    if np.sum(mask_resized_np) == 0:
                        print(f"[WARNING] Mask {i} is empty after resizing to feature grid, skipping")
                        continue
                    
                    # masked_features selection using NumPy boolean array
                    masked_features = spatial_features[0, mask_resized_np, :] # Shape: (num_masked_pixels, D_feat)

                    if masked_features.shape[0] == 0:
                        print(f"[WARNING] No features extracted for mask {i} after spatial selection, skipping")
                        continue

                    region_embedding_pooled = apply_spatial_pooling(masked_features, pooling_strategy)
                    print(f"[STATUS] Applied {pooling_strategy} pooling to {masked_features.shape[0]} spatial features for mask {i}")

                    region_embedding_pooled = region_embedding_pooled.to(device_to_use)

                    # Final projection logic revised for clarity and correctness
                    final_embedding = None

                    if pe_vit_model_to_use is not None: # Features came from ViT intermediate layer
                        print(f"[STATUS] Projecting ViT features to CLIP space for mask {i}")
                        if hasattr(pe_model_to_use, 'visual') and hasattr(pe_model_to_use.visual, 'ln_post') and hasattr(pe_model_to_use.visual, 'proj'):
                            # Ensure region_embedding_pooled is float32 if ln_post expects it
                            if pe_model_to_use.visual.ln_post.weight.dtype == torch.float32 and region_embedding_pooled.dtype != torch.float32:
                                region_embedding_pooled = region_embedding_pooled.float()
                            
                            # Apply ln_post first
                            normed_features = pe_model_to_use.visual.ln_post(region_embedding_pooled)
                            # Then matrix multiply with proj
                            projected_embedding = normed_features @ pe_model_to_use.visual.proj
                            final_embedding = projected_embedding[0] 
                        elif hasattr(pe_model_to_use, 'proj') and pe_model_to_use.proj is not None:
                            # Fallback for some CLIP models if visual.proj isn't the path
                            projected_embedding = region_embedding_pooled @ pe_model_to_use.proj
                            final_embedding = projected_embedding[0]
                        else:
                            print(f"[WARN] ViT features used, but no standard projection head found in PE CLIP model. Using pooled ViT features directly for mask {i}.")
                            final_embedding = region_embedding_pooled[0]
                    else: # Features came from pe_model_to_use.encode_image() (already projected)
                        print(f"[STATUS] Using already projected features from encode_image() for mask {i}")
                        final_embedding = region_embedding_pooled[0]

                    # Normalize the final embedding (common practice)
                    if final_embedding is not None and final_embedding.norm().item() > 1e-6 : # Avoid division by zero
                        final_embedding = final_embedding / final_embedding.norm(dim=-1, keepdim=True)
                    
                    if final_embedding is not None:
                        # Basic validation: check for NaN/inf before appending
                        final_embedding_np_check = final_embedding.detach().cpu().numpy()
                        if np.isnan(final_embedding_np_check).any() or np.isinf(final_embedding_np_check).any():
                            print(f"[WARN] Embedding for mask {i} contains NaN/inf, skipping.")
                            continue

                        temp_embeddings.append(final_embedding.detach().cpu()) # Store on CPU
                        temp_metadata.append(metadata_list[i]) # Assumes masks_cpu_np and metadata_list are in sync
                        temp_labels.append(labels[i])
                        final_masks_for_output.append(mask_cv_np) # The original binarized mask
                        print(f"[STATUS] Successfully generated embedding for mask {i}")
                    else:
                        print(f"[WARN] Final embedding for mask {i} was None, skipping.")
                        
                except Exception as e_emb_loop:
                    print(f"[ERROR] Error generating embedding for mask {i}: {e_emb_loop}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            embeddings = temp_embeddings
            metadata_list = temp_metadata
            labels = temp_labels
            # The masks returned should be the `final_masks_for_output`
            # which are the binarized, original-resolution masks corresponding to successful embeddings
            
            if not embeddings:
                print(f"[WARNING] No embeddings generated after PE processing.")
                # Return the image_np and empty lists for other outputs
                cleanup_temp_resources(temp_file_paths_list=[temp_image_path])
                return image_np if image_np is not None else None, [], [], [], [], None

            print(f"[SUCCESS] Successfully extracted {len(embeddings)} region embeddings.")
            # Return final_masks_for_output instead of masks_cpu_np
            cleanup_temp_resources(temp_file_paths_list=[temp_image_path])
            return image_np, final_masks_for_output, embeddings, metadata_list, labels, None

    except Exception as e:
        print(f"[CRITICAL ERROR] in extract_region_embeddings_autodistill: {e}")
        import traceback
        error_message = f"Critical error in simplified pipeline: {e}\\n{traceback.format_exc()}"
        print(error_message)
        # Ensure cleanup is called on any exception
        cleanup_temp_resources(temp_file_paths_list=[temp_image_path])
        return None, [], [], [], [], error_message
    
    finally:
        # General cleanup for any remaining temp files or resources not explicitly handled above
        # The main cleanup logic is now within cleanup_temp_resources, called before returns
        # This finally block can be used for any other last-minute checks if needed.
        # temp_image_path should already be cleaned by calls within try/except
        print(f"[FINALLY] extract_region_embeddings_autodistill finished.")
        # Removed ensure_consistent_dimensions call here

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
    Extract embeddings for whole images using Perception Encoder
    """
    # Access global models if needed
    global pe_model, pe_vit_model, preprocess, device
    
    # Use provided models or fall back to globals
    pe_model_to_use = pe_model_param if pe_model_param is not None else pe_model
    pe_vit_model_to_use = pe_vit_model_param if pe_vit_model_param is not None else pe_vit_model
    preprocess_to_use = preprocess_param if preprocess_param is not None else preprocess
    device_to_use = device_param if device_param is not None else device
    
    try:
        # Load the image
        if isinstance(image_source, str):
            if image_source.startswith(('http://', 'https://')):
                pil_image = download_image(image_source)
            else:
                pil_image = load_local_image(image_source)
        else:
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
            pil_image = pil_image.resize((new_width, new_height), Image.LANCZOS)

        # Convert to numpy array for processing
        image = np.array(pil_image)
        print(f"[STATUS] Image converted to array, shape: {image.shape}")

        # Create metadata for the whole image
        image_id = str(uuid.uuid4())
        
        # We'll update this after processing to include the actual layer used
        metadata = {
            "region_id": image_id,
            "image_source": str(image_source) if isinstance(image_source, str) else "uploaded_image",
            "bbox": [0, 0, image.shape[1], image.shape[0]],  # Full image bbox
            "area": image.shape[0] * image.shape[1],
            "area_ratio": 1.0,  # Full image has area ratio of 1
            "processing_type": "whole_image",  # Indicates this is a whole image, not a region
            "embedding_method": "pe_encoder",
            "layer_used": optimal_layer  # Will be updated after processing
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
                        # Allow user to select any layer they want - let the model handle invalid layers
                        safe_layer = max(1, optimal_layer)  # Only ensure minimum layer of 1
                        features = pe_vit_model_to_use.forward_features(pe_input, layer_idx=safe_layer)
                        embedding_method = f"vit_forward_features_layer{safe_layer}"
                        print(f"[STATUS] Successfully extracted features using VisionTransformer forward_features with layer {safe_layer}")
                    except Exception as e:
                        print(f"[STATUS] Error using VisionTransformer forward_features with layer {optimal_layer}: {e}")
                        
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
                                # Respect user's layer choice, but clamp to available layers
                                safe_layer = max(1, min(optimal_layer, len(hidden_states)-1))
                                if isinstance(hidden_states, list) and len(hidden_states) > safe_layer:
                                    features = hidden_states[safe_layer]
                                    embedding_method = f"visual_transformer_hidden_states_layer{safe_layer}"
                                    if safe_layer != optimal_layer:
                                        print(f"[STATUS] Note: Requested layer {optimal_layer} not available, using layer {safe_layer} (max available: {len(hidden_states)-1})")
                                    else:
                                        print(f"[STATUS] Successfully extracted features using visual transformer hidden states with layer {safe_layer}")
                        
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
            
            # Extract actual layer used from embedding_method for accurate metadata
            actual_layer_used = optimal_layer  # default fallback
            if "layer" in embedding_method:
                try:
                    # Extract layer number from embedding_method string
                    layer_part = embedding_method.split("layer")[-1]
                    actual_layer_used = int(layer_part)
                except (ValueError, IndexError):
                    # If extraction fails, use the optimal_layer as fallback
                    actual_layer_used = optimal_layer
            
            metadata["layer_used"] = actual_layer_used
            
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

def setup_qdrant(collection_name, vector_size, max_retries=5, retry_delay=3.0):
    """Setup Qdrant collection for storing image region embeddings with retry logic"""
    persist_path = "./image_retrieval_project/qdrant_data"
    os.makedirs(persist_path, exist_ok=True)
    
    # Try to connect with retries
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                print(f"Retry attempt {attempt}/{max_retries} connecting to Qdrant...")
                
            client = QdrantClient(path=persist_path)
            print(f"Successfully connected to local storage at {persist_path}")
            
            try:
                # Check collections
                collections = client.get_collections().collections
                collection_names = [collection.name for collection in collections]
                
                if collection_name not in collection_names:
                    # Create the collection with standard vector configuration
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
                    print(f"✅ Created new collection: {collection_name}")
                else:
                    print(f"✅ Using existing collection: {collection_name}")
                    try:
                        # Verify the collection has the correct vector size
                        collection_info = client.get_collection(collection_name=collection_name)
                        existing_vector_size = collection_info.config.params.vectors.size
                        if existing_vector_size != vector_size:
                            print(f"⚠️ Warning: Collection {collection_name} has vector size {existing_vector_size}, but requested size is {vector_size}")
                            print(f"This might cause issues when searching. Consider using a different collection name.")
                    except Exception as inner_e:
                        print(f"⚠️ Warning: Could not verify vector size: {inner_e}")
                
                return client
                
            except Exception as inner_e:
                print(f"Error working with collection: {inner_e}")
                client.close()
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                continue
                
        except RuntimeError as e:
            if "already accessed by another instance" in str(e):
                print(f"⚠️ Database is locked by another process. Waiting {retry_delay} seconds before retry {attempt+1}/{max_retries}...")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    print(f"❌ Failed to access database after {max_retries} attempts - database is locked!")
                    print(f"Please ensure no other instances of the application are running.")
                    return None
            else:
                print(f"❌ Error connecting to Qdrant: {e}")
                import traceback
                traceback.print_exc()
                return None
        except Exception as e:
            print(f"❌ Unexpected error in setup_qdrant: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    return None

def store_embeddings_in_qdrant(client, collection_name, embeddings, metadata_list):
    """Store embeddings in Qdrant collection"""
    # Check if client is valid
    if client is None:
        print(f"❌ Cannot store embeddings: Database client is not connected")
        return []
        
    if not embeddings or not metadata_list:
        print(f"❌ Nothing to store: Empty embeddings or metadata")
        return []
        
    try:
        print(f"[STATUS] Processing {len(embeddings)} embeddings for storage in {collection_name}...")
        
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

        # Prepare points for upsert
        points = []
        for i in range(len(embedding_vectors)):
            points.append(models.PointStruct(
                id=point_ids[i],
                vector=embedding_vectors[i],
                payload=sanitized_metadata[i]
            ))

        if not points:
            print(f"⚠️ Warning: No valid points to store after processing")
            return []

        # Upsert points in batches
        batch_size = 100
        stored_count = 0
        
        for i in range(0, len(points), batch_size):
            batch = points[i:i+batch_size]
            try:
                client.upsert(
                    collection_name=collection_name,
                    points=batch
                )
                stored_count += len(batch)
                print(f"[STATUS] Stored batch {i//batch_size + 1}/{(len(points)-1)//batch_size + 1} ({len(batch)} points)")
            except Exception as batch_error:
                print(f"❌ Error storing batch {i//batch_size + 1}: {batch_error}")
                continue

        print(f"✅ Successfully stored {stored_count}/{len(points)} embeddings in collection: {collection_name}")
        return point_ids
    except Exception as e:
        print(f"❌ Error in store_embeddings_in_qdrant: {e}")
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
    resume_from_checkpoint=True,
    pooling_strategy="top_k"
):
    """Process all images in a folder and build a searchable database"""
    print(f"Starting to process folder: {folder_path}")
    print(f"Looking for objects matching: {text_prompts}")
    print(f"Using embedding layer: {optimal_layer}")
    print(f"Using pooling strategy: {pooling_strategy}")

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
            image, masks, embeddings, metadata, labels, error_message = extract_region_embeddings_autodistill(
                image_path,
                text_prompt=text_prompts,
                pe_model_param=pe_model,
                pe_vit_model_param=pe_vit_model,
                preprocess_param=preprocess,
                device_param=device,
                is_url=False,
                min_area_ratio=min_area_ratio,
                max_regions=max_regions,
                optimal_layer=optimal_layer,
                pooling_strategy=pooling_strategy
            )

            if image is not None and len(embeddings) > 0:
                print(f"✅ Found {len(embeddings)} valid regions in image {idx+1}")

                # Initialize database if not done yet
                if not initialization_complete:
                    # FIX: Properly extract vector dimension from embedding tensor
                    first_embedding = embeddings[0]
                    if len(first_embedding.shape) == 2 and first_embedding.shape[0] == 1:
                        # Shape is [1, D], extract D
                        vector_size = first_embedding.shape[1]
                    elif len(first_embedding.shape) == 1:
                        # Shape is [D], use D
                        vector_size = first_embedding.shape[0]
                    else:
                        # Flatten for safety
                        vector_size = first_embedding.numel()
                    
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
                    # FIX: Properly extract vector dimension from embedding tensor
                    # For whole image, embedding is a single tensor, not a list
                    if len(embedding.shape) == 2 and embedding.shape[0] == 1:
                        # Shape is [1, D], extract D
                        vector_size = embedding.shape[1]
                    elif len(embedding.shape) == 1:
                        # Shape is [D], use D
                        vector_size = embedding.shape[0]
                    else:
                        # Flatten for safety
                        vector_size = embedding.numel()
                    
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
        
        # Split prompts if provided as period-separated string
        prompt_list = [p.strip() for p in prompts.split(".")] if prompts else ["person", "building", "car", "text", "object"]
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
                # Extract region embeddings - use period-separated format
                image, masks, embeddings, metadata, labels, error_message = extract_region_embeddings_autodistill(
                    img_path,
                    " . ".join(prompt_list) + " .",
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
                        # FIX: Properly extract vector dimension from embedding tensor
                        first_embedding = embeddings[0]
                        if len(first_embedding.shape) == 2 and first_embedding.shape[0] == 1:
                            # Shape is [1, D], extract D
                            vector_size = first_embedding.shape[1]
                        elif len(first_embedding.shape) == 1:
                            # Shape is [D], use D
                            vector_size = first_embedding.shape[0]
                        else:
                            # Flatten for safety
                            vector_size = first_embedding.numel()
                        
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
                                  optimal_layer=40, min_area_ratio=0.01, max_regions=5, 
                                  resume_from_checkpoint=True, pooling_strategy="top_k"):
    """Process folder with progress updates for the UI with advanced parameters"""
    try:
        print(f"[STATUS] Starting folder processing: {folder_path}")
        print(f"[STATUS] Using collection: {collection_name}")
        print(f"[STATUS] Using prompts: {prompts}")
        print(f"[STATUS] Advanced parameters: optimal_layer={optimal_layer}, min_area_ratio={min_area_ratio}, " 
              f"max_regions={max_regions}, resume_from_checkpoint={resume_from_checkpoint}, pooling_strategy={pooling_strategy}")
        
        # Input validation
        if not folder_path or not os.path.exists(folder_path):
            error_msg = f"❌ Error: Folder '{folder_path}' does not exist or is invalid"
            print(f"[ERROR] {error_msg}")
            return error_msg, gr.update(visible=False)
            
        if not collection_name or not isinstance(collection_name, str):
            collection_name = f"image_database_{int(time.time())}"
            print(f"[WARNING] Invalid collection name provided, using: {collection_name}")
        
        # Ensure parameters are in valid ranges
        optimal_layer = max(1, int(optimal_layer))  # Only ensure minimum layer of 1
        min_area_ratio = max(0.001, min(float(min_area_ratio), 0.5))  # Between 0.001-0.5
        max_regions = max(1, min(int(max_regions), 20))  # Between 1-20
        
        # Get global model variables
        global pe_model, pe_vit_model, preprocess, device
        
        # Process and validate prompts
        if not prompts or not isinstance(prompts, str) or prompts.strip() == "":
            prompts = "person . building . car . text . object ."
            print(f"[WARNING] No valid prompts provided, using defaults: {prompts}")
            
        # Split prompts if provided as period-separated string
        prompt_list = [p.strip().lower() for p in prompts.split(".") if p.strip()]
        
        # Ensure we have valid prompts
        if not prompt_list:
            prompt_list = ["person", "building", "car", "text", "object"]
        
        print(f"[STATUS] Parsed prompts: {prompt_list}")
        
        # Get file list - create a stable sorting to ensure consistent processing order
        image_files = sorted([f for f in os.listdir(folder_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))])
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
        vector_dimension = None
        
        # Force database cleanup before starting to avoid connection issues
        from threading import Thread
        cleanup_thread = Thread(target=cleanup_qdrant_connections, args=(True,))
        cleanup_thread.start()
        cleanup_thread.join(timeout=10)
        
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
            
            # Skip non-image files silently
            if not os.path.isfile(img_path) or not img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                print(f"[WARNING] Skipping non-image file: {img_file}")
                skipped += 1
                continue
                
            # Skip files that are too small or potentially corrupt
            try:
                img_size = os.path.getsize(img_path)
                if img_size < 1000:  # Files smaller than 1KB are likely invalid
                    print(f"[WARNING] Skipping very small file ({img_size} bytes): {img_file}")
                    skipped += 1
                    continue
            except Exception as e:
                print(f"[WARNING] Error checking file size: {e}")
            
            try:
                # Add memory management to avoid memory leaks
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                
                # Extract region embeddings with custom parameters
                image, masks, embeddings, metadata, labels, error_message = extract_region_embeddings_autodistill(
                    img_path,
                    " . ".join(prompt_list) + " .",
                    pe_model_param=pe_model,
                    pe_vit_model_param=pe_vit_model,
                    preprocess_param=preprocess,
                    device_param=device,
                    min_area_ratio=min_area_ratio,
                    max_regions=max_regions,
                    optimal_layer=optimal_layer,
                    pooling_strategy=pooling_strategy
                )
                
                # Check if there was an error during processing
                if error_message is not None:
                    print(f"[ERROR] Error processing {img_file}: {error_message}")
                    errors += 1
                    # Show error in UI progress updates
                    if (i + 1) % 1 == 0 or i == total - 1:  # Update every image when there are errors
                        error_stats = (
                            f"❌ Error processing {img_file}:\n{error_message}\n\n"
                            f"📊 Progress: {i+1}/{total} images ({(i+1)/total*100:.1f}%)\n"
                            f"✅ Processed: {processed} images\n"
                            f"⏭️ Skipped: {skipped} images (no regions found)\n"
                            f"❌ Errors: {errors} images\n"
                            f"🔍 Total regions found: {total_regions}"
                        )
                        yield error_stats, gr.update(visible=False)
                    continue
                
                if image is not None and len(embeddings) > 0:
                    print(f"[STATUS] Found {len(embeddings)} regions in {img_file}")
                    
                    # Additional validation for database creation
                    valid_embeddings = []
                    valid_metadata = []
                    invalid_count = 0
                    
                    for idx, (emb, meta) in enumerate(zip(embeddings, metadata)):
                        # Skip regions with missing or invalid data
                        if emb is None or meta is None:
                            print(f"[WARNING] Region {idx} has None embedding or metadata, skipping")
                            invalid_count += 1
                            continue
                            
                        # Validate embedding dimensions
                        if vector_dimension is not None:
                            # Use the same logic as dimension detection to extract actual feature dimension
                            if len(emb.shape) == 2 and emb.shape[0] == 1:
                                # Shape is [1, D], extract D
                                actual_dimension = emb.shape[1]
                            elif len(emb.shape) == 1:
                                # Shape is [D], use D
                                actual_dimension = emb.shape[0]
                            else:
                                # Flatten for safety
                                actual_dimension = emb.numel()
                            
                            if actual_dimension != vector_dimension:
                                print(f"[WARNING] Region {idx} has inconsistent embedding dimension: expected {vector_dimension}, got {actual_dimension}")
                                invalid_count += 1
                                continue
                        
                        # Validate the region metadata
                        if "bbox" not in meta or len(meta["bbox"]) != 4:
                            print(f"[WARNING] Region {idx} has invalid bbox, skipping")
                            invalid_count += 1
                            continue
                            
                        # Skip regions with suspiciously high confidence (1.0 exactly)
                        phrase = meta.get("phrase", "")
                        if ": 1.00" in phrase:
                            print(f"[WARNING] Region {idx} has perfect 1.00 confidence, likely false positive: {phrase}")
                            invalid_count += 1
                            continue
                        
                        # Add timestamp to metadata to track when it was processed
                        meta["processed_timestamp"] = time.time()
                        meta["source_filename"] = img_file
                            
                        # Include only valid embeddings
                        valid_embeddings.append(emb)
                        valid_metadata.append(meta)
                    
                    # If we filtered out bad regions, update counts
                    if invalid_count > 0:
                        print(f"[STATUS] Filtered out {invalid_count}/{len(embeddings)} invalid regions")
                        embeddings = valid_embeddings
                        metadata = valid_metadata
                    
                    # Setup Qdrant client if needed
                    if len(embeddings) > 0:
                        # Store the vector dimension for consistency checking
                        if vector_dimension is None:
                            # FIX: Properly extract vector dimension from embedding tensor
                            # The embeddings are shape [1, D] or [D], we want D
                            first_embedding = embeddings[0]
                            
                            # EXTENSIVE DEBUG OUTPUT
                            print(f"[DEBUG] Raw first_embedding: {first_embedding}")
                            print(f"[DEBUG] first_embedding type: {type(first_embedding)}")
                            print(f"[DEBUG] first_embedding shape: {first_embedding.shape}")
                            print(f"[DEBUG] first_embedding.shape has {len(first_embedding.shape)} dimensions")
                            if len(first_embedding.shape) >= 1:
                                print(f"[DEBUG] first_embedding.shape[0] = {first_embedding.shape[0]}")
                            if len(first_embedding.shape) >= 2:
                                print(f"[DEBUG] first_embedding.shape[1] = {first_embedding.shape[1]}")
                            
                            if len(first_embedding.shape) == 2 and first_embedding.shape[0] == 1:
                                # Shape is [1, D], extract D
                                vector_dimension = first_embedding.shape[1]
                                print(f"[DEBUG] Using 2D logic: vector_dimension = {vector_dimension}")
                            elif len(first_embedding.shape) == 1:
                                # Shape is [D], use D
                                vector_dimension = first_embedding.shape[0]
                                print(f"[DEBUG] Using 1D logic: vector_dimension = {vector_dimension}")
                            else:
                                # Flatten for safety
                                vector_dimension = first_embedding.numel()
                                print(f"[DEBUG] Using fallback logic: vector_dimension = {vector_dimension}")
                            
                            print(f"[STATUS] Established vector dimension: {vector_dimension}")
                            print(f"[DEBUG] First embedding shape: {first_embedding.shape}")
                        
                        if client is None:
                            print(f"[STATUS] Setting up database with vector size {vector_dimension}")
                            client = setup_qdrant(collection_with_layer, vector_dimension)
                            
                            # If we still couldn't create a client, try aggressive cleanup
                            if client is None:
                                print(f"[WARNING] Failed to create database client, attempting recovery...")
                                cleanup_qdrant_connections(force=True)
                                time.sleep(2)
                                client = setup_qdrant(collection_with_layer, vector_dimension)
                                
                                if client is None:
                                    error_msg = "❌ Failed to create database connection after cleanup attempts."
                                    print(f"[ERROR] {error_msg}")
                                    return error_msg, gr.update(visible=False)
                        
                        # Add layer information to each region's metadata
                        for md in metadata:
                            md["embedding_layer"] = optimal_layer
                            md["processing_parameters"] = {
                                "min_area_ratio": min_area_ratio,
                                "max_regions": max_regions
                            }
                        
                        # Store embeddings in batches with retry mechanism
                        retry_count = 0
                        max_retries = 3
                        stored_successfully = False
                        actual_stored_count = 0
                        
                        while not stored_successfully and retry_count < max_retries:
                            try:
                                # Store embeddings
                                print(f"[STATUS] Storing {len(embeddings)} embeddings in database")
                                point_ids = store_embeddings_in_qdrant(client, collection_with_layer, embeddings, metadata)
                                actual_stored_count = len(point_ids)
                                stored_successfully = actual_stored_count > 0
                                
                                if stored_successfully:
                                    processed += 1
                                    total_regions += actual_stored_count
                                    print(f"[STATUS] Successfully stored {actual_stored_count} regions in database")
                                else:
                                    print(f"[WARNING] Failed to store embeddings in database (0 stored), attempt {retry_count + 1}/{max_retries}")
                                    
                            except Exception as e:
                                print(f"[ERROR] Error storing embeddings, attempt {retry_count + 1}/{max_retries}: {e}")
                                import traceback
                                traceback.print_exc()
                                
                            if not stored_successfully and retry_count < max_retries - 1:
                                # Sleep before retrying
                                time.sleep(1 + retry_count)
                                retry_count += 1
                                # Try to recreate client
                                try:
                                    client.close()
                                except:
                                    pass
                                client = setup_qdrant(collection_with_layer, vector_dimension)
                            else:
                                break
                                
                        if not stored_successfully:
                            errors += 1
                            print(f"[ERROR] Failed to store regions for {img_file} after {max_retries} attempts - NO REGIONS STORED")
                    else:
                        print(f"[STATUS] No valid regions remain after filtering for {img_file}")
                        skipped += 1
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
            try:
                client.close()
            except Exception as e:
                print(f"[WARNING] Error closing client: {e}")
        
        # Final stats
        if processed == 0 and total > 0:
            if errors > 0:
                final_message = (
                    f"⚠️ Warning: No images were successfully processed!\n\n"
                    f"📁 Total images: {total}\n"
                    f"❌ Errors: {errors} images\n"
                    f"⏭️ Skipped: {skipped} images\n\n"
                    f"Please check the console for error messages."
                )
            else:
                final_message = (
                    f"⚠️ Warning: No regions were found in any images!\n\n"
                    f"📁 Total images: {total}\n"
                    f"⏭️ Skipped: {skipped} images (no regions found)\n\n"
                    f"Try adjusting the parameters:\n"
                    f"- Decrease Min Region Size (currently {min_area_ratio})\n"
                    f"- Add more detection prompts (currently {prompts})\n"
                )
        else:
            final_message = (
                f"✅ Complete! Processing summary:\n\n"
                f"📁 Total images: {total}\n"
                f"✅ Successfully processed: {processed}\n"
                f"⏭️ Skipped: {skipped} images (no regions found)\n"
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
                        # FIX: Properly extract vector dimension from embedding tensor
                        # For whole image, embedding is a single tensor, not a list
                        if len(embedding.shape) == 2 and embedding.shape[0] == 1:
                            # Shape is [1, D], extract D
                            vector_size = embedding.shape[1]
                        elif len(embedding.shape) == 1:
                            # Shape is [D], use D
                            vector_size = embedding.shape[0]
                        else:
                            # Flatten for safety
                            vector_size = embedding.numel()
                        
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
            
            # Store the full collection name exactly as provided
            # This ensures search operations use the correct collection name with layer suffix
            self.active_collection = collection_name
            print(f"[STATUS] Changed active collection from {previous} to {collection_name}")
            
            # Close existing client if database changed
            if self.active_client is not None:
                print(f"[STATUS] Closing previous database connection")
                self.active_client.close()
                self.active_client = None
                
            return f"Set active database to: {collection_name}"
        return "No database selected"

    def process_image_with_prompt(self, image, text_prompt, min_area_ratio=0.01, max_regions=5, optimal_layer=40, pooling_strategy="top_k"):
        """Process an uploaded image with a text prompt to detect regions"""
        if isinstance(image, np.ndarray):
            image_pil = Image.fromarray(image)
        else:
            image_pil = image

        # Extract regions from image
        image_np, masks, embeddings, metadata, labels, error_message = extract_region_embeddings_autodistill(
            image_pil,
            text_prompt=text_prompt,
            pe_model_param=self.pe_model,
            pe_vit_model_param=self.pe_vit_model,
            preprocess_param=self.preprocess,
            device_param=self.device,
            is_url=False,
            min_area_ratio=min_area_ratio,
            max_regions=max_regions,
            optimal_layer=optimal_layer,
            pooling_strategy=pooling_strategy
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
        """Creates a visualization of a single region with enhanced visibility"""
        preview = image.copy()
        
        # Create a mask for highlighting
        mask_3d = np.stack([mask, mask, mask], axis=2)
        
        # Make the region much brighter while dimming the surroundings
        highlighted = np.where(mask_3d, np.minimum(preview * 1.8, 255), preview * 0.3)

        # Find contours for drawing a border
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Draw a thicker green border around the region
        cv2.drawContours(highlighted, contours, -1, (0, 255, 0), 3)

        # Create the figure with a border
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(highlighted.astype(np.uint8))
        
        # Show the label if provided
        if label:
            ax.set_title(label, fontsize=12, pad=10)
            
        # Add a border to the figure
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(3)
            spine.set_color('#01579b')  # Blue border color
            
        ax.axis('on')  # Show border
        ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, 
                      labelbottom=False, labelleft=False)  # Hide ticks but keep border

        # Convert to image
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
            
            # Use exactly the collection name the user selected
            collection_name = self.active_collection
            print(f"[STATUS] Using selected collection: {collection_name}")
            
            # Verify the collection exists
            collections = self.active_client.get_collections().collections
            collection_names = [collection.name for collection in collections]
            print(f"[STATUS] Available collections: {collection_names}")
            
            # Check if collection exists
            if collection_name not in collection_names:
                print(f"[ERROR] Collection {collection_name} not found")
                available_cols = "\n".join(collection_names)
                return None, f"Error: Collection '{collection_name}' not found. Available collections:\n{available_cols}", gr.update(visible=False), gr.update(choices=[], value=None)
            
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
            
            # Perform the search
            try:
                search_results = self.active_client.search(
                    collection_name=collection_name,
                    query_vector=embedding_list,
                    limit=limit,
                    score_threshold=threshold
                )
                
                print(f"[STATUS] Found {len(search_results)} results in {collection_name}")
                
                # Add collection information to results
                for result in search_results:
                    if hasattr(result, "payload"):
                        # Set a default collection type based on name for display
                        if "_layer" in collection_name:
                            result.payload["collection_type"] = "region"
                        elif "_whole_images_layer" in collection_name:
                            result.payload["collection_type"] = "whole_image"
                        else:
                            result.payload["collection_type"] = "unknown"
                        result.payload["collection_name"] = collection_name
                
                if not search_results:
                    print(f"[STATUS] No similar items found above threshold {threshold}")
                    return None, f"No similar items found with similarity threshold {threshold}.", gr.update(visible=False), gr.update(choices=[], value=None)
                
                # Create visualization of results
                return self.create_unified_search_results_visualization(
                    search_results, 
                    self.detected_regions["image"],
                    self.detected_regions["masks"][region_idx],
                    self.detected_regions["labels"][region_idx],
                    "region"
                )
            except Exception as e:
                print(f"[ERROR] Error during search in collection {collection_name}: {e}")
                import traceback
                traceback.print_exc()
                return None, f"Error searching collection {collection_name}: {str(e)}", gr.update(visible=False), gr.update(choices=[], value=None)
        
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
            print(f"[INFO] Search layer ({optimal_layer}) doesn't match the layer used for processing ({actual_layer})")
            print(f"[INFO] This is fine as PE embeddings are compatible across different layers")

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
            
            # Use exactly the collection name the user selected
            collection_name = self.active_collection
            print(f"[STATUS] Using selected collection: {collection_name}")
            
            # Verify the collection exists
            collections = self.active_client.get_collections().collections
            collection_names = [collection.name for collection in collections]
            print(f"[STATUS] Available collections: {collection_names}")
            
            # Check if collection exists
            if collection_name not in collection_names:
                print(f"[ERROR] Collection {collection_name} not found")
                available_cols = "\n".join(collection_names)
                return None, f"Error: Collection '{collection_name}' not found. Available collections:\n{available_cols}", gr.update(visible=False), gr.update(choices=[], value=None)
            
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
            
            # Perform the search
            try:
                search_results = self.active_client.search(
                    collection_name=collection_name,
                    query_vector=embedding_list,
                    limit=limit,
                    score_threshold=threshold
                )
                
                print(f"[STATUS] Found {len(search_results)} results in {collection_name}")
                
                # Add collection information to results
                for result in search_results:
                    if hasattr(result, "payload"):
                        # Set a default collection type based on name for display
                        if "_layer" in collection_name:
                            result.payload["collection_type"] = "region"
                        elif "_whole_images_layer" in collection_name:
                            result.payload["collection_type"] = "whole_image"
                        else:
                            result.payload["collection_type"] = "unknown"
                        result.payload["collection_name"] = collection_name
                
                if not search_results:
                    print(f"[STATUS] No similar items found above threshold {threshold}")
                    return None, f"No similar items found with similarity threshold {threshold}.", gr.update(visible=False), gr.update(choices=[], value=None)
                
                # Create visualization of results
                return self.create_unified_search_results_visualization(
                    search_results, 
                    self.whole_image["image"],
                    None,  # No mask for whole image
                    "Whole Image",  # Label for whole image
                    "whole_image"
                )
            except Exception as e:
                print(f"[ERROR] Error during search in collection {collection_name}: {e}")
                import traceback
                traceback.print_exc()
                return None, f"Error searching collection {collection_name}: {str(e)}", gr.update(visible=False), gr.update(choices=[], value=None)
        
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
                    
                    # Display the region with a highlight border
                    ax.imshow(region)
                    
                    # Add a border around the region to make it stand out
                    border_width = 3  
                    for spine in ax.spines.values():
                        spine.set_visible(True)
                        spine.set_linewidth(border_width)
                        spine.set_color('#01579b')  # Blue border color
                    
                    # Get detected class name from metadata (improved labeling)
                    detected_class = metadata.get("detected_class", metadata.get("phrase", "Object"))
                    confidence = metadata.get("confidence", "")
                    confidence_str = f" ({confidence:.2f})" if confidence else ""
                    
                    # Create a badge showing it's a region with the detected class
                    badge_props = dict(boxstyle="round,pad=0.3", fc="#e1f5fe", ec="#01579b", alpha=0.8)
                    badge_text = f"{detected_class.upper()}"
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
                            "detected_class": detected_class,
                            "confidence": confidence,
                            "score": score,
                            "type": "region",
                            "collection_name": collection_name
                        }
                    })
                    
                    # Create radio choice label with actual detected class
                    choice_label = f"Result {i+1}: {filename} - {detected_class}{confidence_str}"
                    
                    # Add to result info text with better details
                    result_info.append(f"**Result {i+1}:** {filename}\n- Type: Region ({detected_class})\n- Score: {score:.4f}\n")
                    
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
                            "detected_class": "whole_image",
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
                
                # Add filename in a less obtrusive way - smaller text at the bottom
                ax.text(0.5, 0.03, f"{filename}", transform=ax.transAxes, 
                        ha='center', va='bottom', fontsize=8, 
                        bbox=dict(facecolor='white', alpha=0.6, pad=2,
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
        """Displays an enlarged version of the selected search result with region highlighted in full image"""
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
                
            # Safely handle metadata
            if "metadata" not in result_data:
                print(f"[WARNING] No 'metadata' key in result_data: {list(result_data.keys())}")
                # Create default metadata
                metadata = {
                    "filename": f"Result {result_idx+1}",
                    "phrase": "Unknown",
                    "score": 0.0,
                    "image_source": None
                }
            else:
                metadata = result_data["metadata"]
                
            # Get metadata values safely with defaults
            filename = metadata.get("filename", f"Result {result_idx+1}")
            phrase = metadata.get("detected_class", "Unknown") # Changed "phrase" to "detected_class"
            score = metadata.get("score", 0.0)
            image_source = metadata.get("image_source", None)
            result_type = metadata.get("type", "region")
            
            # Create a more visually appealing enlarged display
            fig = plt.figure(figsize=(12, 10), constrained_layout=True)
            gs = gridspec.GridSpec(1, 1, figure=fig)
            ax = fig.add_subplot(gs[0, 0])
            
            # Load the full original image if we have a source
            if image_source and os.path.exists(image_source):
                try:
                    # Load the full original image
                    full_image = np.array(load_local_image(image_source))
                    
                    if result_type == "region":
                        # Get bounding box if available
                        bbox = result_data.get("bbox", None)
                        
                        # Create a copy of the image to draw on
                        display_image = full_image.copy()
                        
                        # If we have a valid bounding box, highlight the region
                        if bbox and len(bbox) == 4:
                            x_min, y_min, x_max, y_max = bbox
                            
                            # Ensure bbox is within image bounds
                            x_min = max(0, min(x_min, display_image.shape[1]-1))
                            y_min = max(0, min(y_min, display_image.shape[0]-1))
                            x_max = max(x_min+1, min(x_max, display_image.shape[1]))
                            y_max = max(y_min+1, min(y_max, display_image.shape[0]))
                            
                            # Draw rectangle around region
                            cv2.rectangle(display_image, 
                                         (int(x_min), int(y_min)), 
                                         (int(x_max), int(y_max)), 
                                         (0, 255, 0), 3)
                                         
                            # Highlight the region slightly
                            alpha = 0.3
                            roi = display_image[y_min:y_max, x_min:x_max]
                            highlighted_roi = np.clip(roi * 1.3, 0, 255).astype(np.uint8)
                            display_image[y_min:y_max, x_min:x_max] = cv2.addWeighted(
                                highlighted_roi, alpha, roi, 1-alpha, 0)
                    else:
                        # For whole image matches, just use the full image
                        display_image = full_image
                        
                    # Show the full image with highlighted region
                    ax.imshow(display_image)
                except Exception as e:
                    print(f"[WARNING] Error loading full image: {e}")
                    # Fallback to the region image
                    region = result_data.get("image", None)
                    if region is not None:
                        ax.imshow(region)
            else:
                # If no source image available, use the region image
                region = result_data.get("image", None)
                if region is not None:
                    ax.imshow(region)
                else:
                    ax.text(0.5, 0.5, "Image not available", 
                           ha='center', va='center', fontsize=14)
            
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
                        placeholder="Examples: 'person . car . building .' OR 'all the chairs in the room' OR 'a person with pink clothes'",
                        label="Detection Prompts",
                        info="💡 Use period-separated for multiple objects (person . car . building .) OR natural language for specific descriptions (all the chairs, a person with red shirt)",
                        value="person . car . building ."
                    )
                
                # Add parameter controls before the process button
                with gr.Row():
                    with gr.Column():
                        embedding_layer = gr.Slider(
                            minimum=1, maximum=50, value=40, step=1,
                            label="Embedding Layer",
                            info="Layer of the Perception Encoder to use for detection"
                        )
                    with gr.Column():
                        min_area_ratio = gr.Slider(
                            minimum=0.001, maximum=0.1, value=0.01, step=0.001,
                            label="Minimum Area Ratio",
                            info="Minimum size of regions to detect (as a fraction of image size)"
                        )
                    with gr.Column():
                        max_regions = gr.Slider(
                            minimum=1, maximum=20, value=5, step=1,
                            label="Maximum Regions",
                            info="Maximum number of regions to detect per image"
                        )
                
                with gr.Row():
                    process_button = gr.Button("Detect Regions", variant="primary")
            
            # Whole-image mode components
            with gr.Group(visible=False) as whole_image_mode_group:
                # Add embedding layer for whole image processing too
                with gr.Row():
                    whole_image_layer = gr.Slider(
                        minimum=1, maximum=50, value=40, step=1,
                        label="Embedding Layer",
                        info="Layer of the Perception Encoder to use for processing"
                    )
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
                        
                        # Remove the embedding layer slider from here since we moved it above
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
                inputs=[input_image, text_prompt, min_area_ratio, max_regions, embedding_layer],
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
                inputs=[input_image, whole_image_layer],
                outputs=[processed_output, whole_image_info]
            )
            
            whole_search_button.click(
                self.search_whole_image,
                inputs=[whole_similarity_slider, whole_max_results_dropdown, whole_image_layer],
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
            "layer_used": metadata.get("layer_used", optimal_layer) if metadata else optimal_layer
        }

        if embedding is None:
            return image_np, "Failed to process image"

        # Create a simple visualization with border
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(image_np)
        
        # Use the actual layer from metadata if available
        actual_layer = metadata.get("layer_used", optimal_layer) if metadata else optimal_layer
        ax.set_title(f"Whole Image Processed (Layer {actual_layer})")
        
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
        
        # Use actual layer in the response message too
        return processed_image, f"Image processed successfully using layer {actual_layer}"

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
    with gr.Blocks(title="Revers-o: GroundedSAM + Perception Encoders Image Similarity Search") as demo:
        gr.Markdown("# 🔍 Revers-o: GroundedSAM + Perception Encoders Image Similarity Search")
        
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
                                placeholder="Examples: 'person . car . building .' OR 'all the chairs in the room' OR 'a person with pink clothes'",
                                label="Detection Prompts",
                                info="💡 Use period-separated for multiple objects (person . car . building .) OR natural language for specific descriptions (all the chairs, a person with red shirt)",
                                value="person . car . building ."
                            )
                            
                            # Add pooling strategy selection
                            pooling_strategy = gr.Dropdown(
                                choices=["top_k", "max", "attention", "average"],
                                value="top_k",
                                label="Feature Pooling Strategy",
                                info="How to combine spatial features from detected regions"
                            )
                            
                            # Add region detection parameters
                            with gr.Row():
                                with gr.Column():
                                    min_area_ratio = gr.Slider(
                                        minimum=0.001, maximum=0.1, value=0.01, step=0.001,
                                        label="Minimum Area Ratio",
                                        info="Minimum size of regions to detect (as a fraction of image size)"
                                    )
                                with gr.Column():
                                    max_regions = gr.Slider(
                                        minimum=1, maximum=20, value=5, step=1,
                                        label="Maximum Regions",
                                        info="Maximum number of regions to detect per image"
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
                        # Display the logo image
                        gr.Image("big_logo.png", show_label=False, container=False)
                        
                        db_progress = gr.Textbox(label="Processing Status")
                        db_done_msg = gr.Markdown(visible=False)
                
                # Add section for database verification and repair
                gr.Markdown("## Database Maintenance", elem_id="db_maintenance")
                gr.Markdown("Verify and repair existing databases if you encounter issues with region detection or search quality.")
                
                with gr.Row():
                    with gr.Column():
                        verify_db_dropdown = gr.Dropdown(
                            choices=app_state.available_databases if app_state.available_databases else ["No databases found"],
                            value=app_state.available_databases[0] if app_state.available_databases else None,
                            label="Select Database to Verify/Repair"
                        )
                    with gr.Column():
                        force_rebuild_checkbox = gr.Checkbox(
                            value=False,
                            label="Force Complete Rebuild",
                            info="Enable this to force a complete rebuild of the database (slower but more thorough)"
                        )
                
                with gr.Row():
                    with gr.Column():
                        verify_db_button = gr.Button("Verify Database", variant="secondary")
                    with gr.Column():
                        repair_db_button = gr.Button("Repair Database", variant="primary")
                
                verify_repair_status = gr.Textbox(label="Verification/Repair Status")
                        
                # Function to toggle region options based on mode
                def toggle_region_options(mode):
                    if mode == "Region Detection (GroundedSAM + PE)":
                        return gr.update(visible=True)
                    else:
                        return gr.update(visible=False)
                
                db_mode.change(toggle_region_options, inputs=[db_mode], outputs=[region_options_group])
                
                # Connect database verification functions
                def verify_database(collection_name):
                    """Verify the selected database"""
                    if collection_name and collection_name != "No databases found":
                        print(f"[STATUS] Verifying database: {collection_name}")
                        success, message = verify_repair_database(collection_name, force_rebuild=False)
                        if success:
                            return f"✅ {message}"
                        else:
                            return f"❌ {message}"
                    else:
                        return "No database selected for verification"
                
                def repair_database(collection_name, force_rebuild):
                    """Repair the selected database"""
                    if collection_name and collection_name != "No databases found":
                        print(f"[STATUS] Repairing database: {collection_name} (force_rebuild={force_rebuild})")
                        success, message = verify_repair_database(collection_name, force_rebuild=force_rebuild)
                        
                        # Refresh databases after repair
                        app_state.update_available_databases(verbose=True)
                        
                        if success:
                            return f"✅ {message}"
                        else:
                            return f"❌ {message}"
                    else:
                        return "No database selected for repair"
                
                # Connect verification buttons
                verify_db_button.click(
                    verify_database,
                    inputs=[verify_db_dropdown],
                    outputs=[verify_repair_status]
                )
                
                repair_db_button.click(
                    repair_database,
                    inputs=[verify_db_dropdown, force_rebuild_checkbox],
                    outputs=[verify_repair_status]
                )
                
                # Refresh button for database verification dropdown
                def refresh_verify_databases():
                    """Refresh the database list for verification"""
                    app_state.update_available_databases(verbose=True)
                    if app_state.available_databases:
                        return gr.Dropdown(choices=app_state.available_databases, value=app_state.available_databases[0])
                    else:
                        return gr.Dropdown(choices=["No databases found"], value=None)
                
                # Connect build database functions
                def process_folder_with_mode(folder_path, collection_name, prompts, layer, mode, min_area_ratio=0.01, max_regions=5, pooling_strategy="top_k"):
                    """Process folder based on selected mode"""
                    print(f"[STATUS] Starting folder processing with mode: {mode}")
                    print(f"[STATUS] Using pooling strategy: {pooling_strategy}")
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
                            min_area_ratio=min_area_ratio,
                            max_regions=max_regions,
                            pooling_strategy=pooling_strategy
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
                        
                    # After completing database creation, refresh verify dropdown
                    verify_db_dropdown.choices = app_state.update_available_databases(verbose=True)
                    if app_state.available_databases:
                        verify_db_dropdown.value = app_state.available_databases[0]
                
                create_db_button.click(
                    process_folder_with_mode,
                    inputs=[db_folder_path, db_collection_name, region_prompts, layer_slider, db_mode, min_area_ratio, max_regions, pooling_strategy],
                    outputs=[db_progress, db_done_msg]
                )
            
            # Tab 2: Search Database
            with gr.TabItem("Search Database"):
                gr.Markdown("## Search Database")
                gr.Markdown("Upload an image, detect regions, and search for similar regions in your database.")
                
                # Database selection
                with gr.Row():
                    with gr.Column(scale=1):
                        # Refresh button for database list
                        refresh_db_button = gr.Button("Refresh Database List")
                    with gr.Column(scale=2):
                        db_dropdown = gr.Dropdown(
                            choices=app_state.available_databases if app_state.available_databases else ["No databases found"],
                            value=app_state.available_databases[0] if app_state.available_databases else None,
                            label="Select Database"
                        )
                    with gr.Column(scale=1):
                        # Delete database button
                        delete_db_button = gr.Button("Delete Selected Database", variant="stop")
                
                # Add row for reset database connection button
                with gr.Row():
                    with gr.Column(scale=3):
                        # Empty space for alignment
                        gr.Markdown("") 
                    with gr.Column(scale=1):
                        # Reset database connection button
                        reset_db_button = gr.Button("Reset Database Connection", variant="secondary")
                
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
                            placeholder="Examples: 'person . car . building .' OR 'all the chairs in the room' OR 'a person with pink clothes'",
                            label="Detection Prompts",
                            info="💡 Use period-separated for multiple objects (person . car . building .) OR natural language for specific descriptions (all the chairs, a person with red shirt)",
                            value="person . car . building ."
                        )
                    
                    # Add parameter controls before the process button
                    with gr.Row():
                        with gr.Column():
                            embedding_layer = gr.Slider(
                                minimum=1, maximum=50, value=40, step=1,
                                label="Embedding Layer",
                                info="Layer of the Perception Encoder to use for detection"
                            )
                        with gr.Column():
                            min_area_ratio = gr.Slider(
                                minimum=0.001, maximum=0.1, value=0.01, step=0.001,
                                label="Minimum Area Ratio",
                                info="Minimum size of regions to detect (as a fraction of image size)"
                            )
                        with gr.Column():
                            max_regions = gr.Slider(
                                minimum=1, maximum=20, value=5, step=1,
                                label="Maximum Regions",
                                info="Maximum number of regions to detect per image"
                            )
                    
                    with gr.Row():
                        process_button = gr.Button("Detect Regions", variant="primary")
                
                # Whole-image mode components
                with gr.Group(visible=False) as whole_image_mode_group:
                    # Add embedding layer for whole image processing too
                    with gr.Row():
                        whole_image_layer = gr.Slider(
                            minimum=1, maximum=50, value=40, step=1,
                            label="Embedding Layer",
                            info="Layer of the Perception Encoder to use for processing"
                        )
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
                            
                            # Remove the embedding layer slider from here since we moved it above
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
                    
                    # Function to delete the selected database
                    def delete_selected_database(collection_name):
                        if collection_name and collection_name != "No databases found":
                            success, message = delete_database_collection(collection_name)
                            if success:
                                # Refresh the database list after successful deletion
                                app_state.update_available_databases(verbose=True)
                                if app_state.available_databases:
                                    return gr.Dropdown(choices=app_state.available_databases, value=app_state.available_databases[0]), f"{message}"
                                else:
                                    return gr.Dropdown(choices=["No databases found"], value=None), f"{message}"
                            return gr.update(), f"{message}"
                        return gr.update(), "No database selected for deletion"
                    
                    # Connect delete button
                    delete_db_button.click(delete_selected_database, inputs=[db_dropdown], outputs=[db_dropdown, detection_info])
                    
                    # Function to reset the database connection
                    def reset_database_connection():
                        """Reset the database connection by forcefully cleaning up locks"""
                        try:
                            print(f"[STATUS] Forcefully resetting database connection...")
                            
                            # Close any existing client if present
                            if interface.active_client is not None:
                                try:
                                    interface.active_client.close()
                                    interface.active_client = None
                                    print(f"[STATUS] Closed existing database client")
                                except Exception as e:
                                    print(f"[WARNING] Error closing client: {e}")
                            
                            # Run aggressive cleanup
                            success = cleanup_qdrant_connections(force=True)
                            
                            # Refresh database list
                            app_state.update_available_databases(verbose=True)
                            
                            if success:
                                message = "✅ Database connection has been reset successfully."
                                if app_state.available_databases:
                                    return gr.Dropdown(choices=app_state.available_databases, value=app_state.available_databases[0]), message
                                else:
                                    return gr.Dropdown(choices=["No databases found"], value=None), message
                            else:
                                # If the cleanup wasn't successful
                                message = "⚠️ Database reset partially completed. You may need to restart the application if issues persist."
                                if app_state.available_databases:
                                    return gr.Dropdown(choices=app_state.available_databases, value=app_state.available_databases[0]), message
                                else:
                                    return gr.Dropdown(choices=["No databases found"], value=None), message
                            
                        except Exception as e:
                            print(f"[ERROR] Error during database reset: {e}")
                            import traceback
                            traceback.print_exc()
                            return gr.update(), f"❌ Error resetting database: {str(e)}"
                    
                    # Connect reset button
                    reset_db_button.click(reset_database_connection, inputs=[], outputs=[db_dropdown, detection_info])
                    
                    # Connect buttons to functions for region-based processing
                    process_button.click(
                        interface.process_image_with_prompt,
                        inputs=[input_image, text_prompt, min_area_ratio, max_regions, embedding_layer],
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
                        inputs=[input_image, whole_image_layer],
                        outputs=[processed_output, whole_image_info]
                    )
                    
                    whole_search_button.click(
                        interface.search_whole_image,
                        inputs=[whole_similarity_slider, whole_max_results_dropdown, whole_image_layer],
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
                    # Revers-o: GroundedSAM + Perception Encoders Image Similarity Search
                    
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

def cleanup_qdrant_connections(force=False):
    """Attempt to forcefully clean up any existing Qdrant connections"""
    try:
        # Path to the lock file and data directories
        qdrant_data_dir = os.path.join("image_retrieval_project", "qdrant_data")
        lock_file = os.path.join(qdrant_data_dir, ".lock")
        collection_dir = os.path.join(qdrant_data_dir, "collection")
        collections_dir = os.path.join(qdrant_data_dir, "collections")
        
        # Create directories if they don't exist
        os.makedirs(qdrant_data_dir, exist_ok=True)
        
        # Force kill any processes using the qdrant files on macOS/Linux
        if sys.platform in ["darwin", "linux"]:
            try:
                import subprocess
                print(f"[STATUS] Checking for processes using Qdrant database...")
                # Find processes using the qdrant_data directory
                result = subprocess.run(
                    f"lsof -t +D {qdrant_data_dir} 2>/dev/null || echo ''", 
                    shell=True, 
                    text=True, 
                    capture_output=True
                )
                
                if result.stdout.strip():
                    pids = result.stdout.strip().split("\n")
                    print(f"[STATUS] Found {len(pids)} process(es) using Qdrant data directory")
                    
                    # Try to kill each process
                    for pid in pids:
                        if pid and pid.strip():
                            try:
                                # Don't kill our own process
                                if int(pid) != os.getpid():
                                    print(f"[STATUS] Terminating process {pid}...")
                                    subprocess.run(f"kill -9 {pid}", shell=True)
                                    print(f"[STATUS] Terminated process {pid}")
                            except Exception as e:
                                print(f"[WARNING] Could not terminate process {pid}: {e}")
                else:
                    print(f"[STATUS] No processes found using Qdrant database")
            except Exception as e:
                print(f"[WARNING] Error checking for processes using Qdrant: {e}")
        
        # Check if the lock file exists
        lock_removed = False
        if os.path.exists(lock_file):
            print(f"[STATUS] Found existing Qdrant lock file. Attempting to clean up...")
            try:
                # Try to remove the lock file
                os.remove(lock_file)
                print(f"[STATUS] Successfully removed lock file")
                lock_removed = True
                # Sleep briefly to let the filesystem settle
                time.sleep(1)
            except Exception as e:
                print(f"[WARNING] Could not remove lock file: {e}")
                
                # If force is True, try more aggressive methods
                if force:
                    print(f"[STATUS] Attempting to force remove lock...")
                    try:
                        # On Unix-like systems, try using system commands
                        if sys.platform in ["darwin", "linux"]:
                            import subprocess
                            subprocess.run(f"rm -f {lock_file}", shell=True)
                            if not os.path.exists(lock_file):
                                print(f"[STATUS] Successfully force-removed lock file")
                                lock_removed = True
                                time.sleep(1)
                    except Exception as force_e:
                        print(f"[WARNING] Force removal failed: {force_e}")
        else:
            print(f"[STATUS] No Qdrant lock file found")
            lock_removed = True  # No lock to remove
            
        # If force is True and we couldn't remove the lock, try rebuilding the database structure
        if force and not lock_removed:
            print(f"[STATUS] Attempting to rebuild Qdrant database structure...")
            try:
                # Backup the collections
                timestamp = int(time.time())
                backup_dir = f"./image_retrieval_project/qdrant_backup_{timestamp}"
                os.makedirs(backup_dir, exist_ok=True)
                
                # Copy collection data if directories exist
                if os.path.exists(collection_dir):
                    import shutil
                    shutil.copytree(collection_dir, os.path.join(backup_dir, "collection"))
                    print(f"[STATUS] Backed up collection directory to {backup_dir}/collection")
                
                if os.path.exists(collections_dir):
                    import shutil
                    shutil.copytree(collections_dir, os.path.join(backup_dir, "collections"))
                    print(f"[STATUS] Backed up collections directory to {backup_dir}/collections")
                
                # Remove the entire qdrant_data directory
                import shutil
                shutil.rmtree(qdrant_data_dir, ignore_errors=True)
                time.sleep(1)
                
                # Recreate the directory structure
                os.makedirs(qdrant_data_dir, exist_ok=True)
                
                # Restore collections
                if os.path.exists(os.path.join(backup_dir, "collection")):
                    shutil.copytree(os.path.join(backup_dir, "collection"), collection_dir)
                    print(f"[STATUS] Restored collection directory")
                
                if os.path.exists(os.path.join(backup_dir, "collections")):
                    shutil.copytree(os.path.join(backup_dir, "collections"), collections_dir)
                    print(f"[STATUS] Restored collections directory")
                
                print(f"[STATUS] Qdrant database structure rebuilt")
                return True
            except Exception as rebuild_e:
                print(f"[ERROR] Failed to rebuild database: {rebuild_e}")
                return False
        
        return lock_removed
    except Exception as e:
        print(f"[ERROR] Error during Qdrant cleanup: {e}")
        return False
        
def main():
    # Parse command line arguments for backward compatibility
    parser = argparse.ArgumentParser(description="Grounded SAM Region Search Application")
    parser.add_argument("--build-database", action="store_true", help="Build database from images folder")
    parser.add_argument("--interface", action="store_true", help="Launch Gradio interface")
    parser.add_argument("--all", action="store_true", help="Build database and launch interface")
    parser.add_argument("--folder", default="./my_images", help="Folder containing images to process")
    parser.add_argument("--prompts", default="person . building . car . text . object .", help="Text prompts for detection (period-separated format)")
    parser.add_argument("--collection", default="grounded_image_regions", help="Qdrant collection name")
    # Removed legacy interface options
    # Add mode option
    parser.add_argument("--mode", choices=["region", "whole_image"], default="region", 
                        help="Processing mode: 'region' for GroundedSAM+PE or 'whole_image' for PE only")
    parser.add_argument("--optimal-layer", type=int, default=40, help="Optimal layer for PE model (default: 40)")
    # Add port option
    parser.add_argument("--port", type=int, default=7860, help="Port to use for the Gradio interface")

    args = parser.parse_args()

    # Clean up any existing database connections with force option
    print(f"[STATUS] Running database cleanup and integrity checks...")
    # Try to clean up with a timeout to prevent blocking forever
    import threading
    cleanup_thread = threading.Thread(target=cleanup_qdrant_connections, args=(True,))
    cleanup_thread.start()
    # Wait for 10 seconds max
    cleanup_thread.join(timeout=10)
    
    # If the thread is still alive, it's taking too long
    if cleanup_thread.is_alive():
        print(f"[WARNING] Database cleanup is taking too long - continuing with application startup")
        # We don't want to block forever, just continue with application startup
    else:
        print(f"[STATUS] Database cleanup completed")

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

    # Check if we should show the interface (either standalone or after database build)
    launch_interface = args.interface or args.all or not args.build_database
    
    # Build interface if needed
    if launch_interface:
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

    # The interface launch is now handled at the beginning of the function
    # This legacy code has been removed

def verify_repair_database(collection_name, force_rebuild=False):
    """
    Verifies and optionally repairs a database collection
    
    Args:
        collection_name: Name of the collection to verify
        force_rebuild: If True, forces a complete rebuild of the collection
                     This is useful if the collection is corrupted
    
    Returns:
        tuple: (success, message)
    """
    try:
        print(f"[STATUS] Verifying database collection: {collection_name}")
        
        # Check if the collection exists
        qdrant_data_dir = os.path.join("image_retrieval_project", "qdrant_data")
        if not os.path.exists(qdrant_data_dir):
            return False, f"Qdrant data directory not found at {qdrant_data_dir}"
        
        # Try to connect to the database
        try:
            print(f"[STATUS] Connecting to Qdrant...")
            client = QdrantClient(path=qdrant_data_dir)
            print(f"[STATUS] Successfully connected to Qdrant database")
        except Exception as e:
            print(f"[ERROR] Failed to connect to Qdrant: {e}")
            
            # Try to fix connection issues
            print(f"[STATUS] Attempting to fix database connection...")
            cleanup_success = cleanup_qdrant_connections(force=True)
            
            if cleanup_success:
                try:
                    client = QdrantClient(path=qdrant_data_dir)
                    print(f"[STATUS] Successfully reconnected to database after cleanup")
                except Exception as e2:
                    return False, f"Failed to connect to database even after cleanup: {e2}"
            else:
                return False, f"Failed to clean up database connections: {e}"
        
        # Check if the collection exists
        try:
            collections = client.get_collections().collections
            collection_names = [collection.name for collection in collections]
            print(f"[STATUS] Found collections: {collection_names}")
            
            if collection_name not in collection_names:
                return False, f"Collection '{collection_name}' not found"
        except Exception as e:
            print(f"[ERROR] Failed to get collections list: {e}")
            return False, f"Failed to list collections: {e}"
        
        # Get collection info
        try:
            collection_info = client.get_collection(collection_name=collection_name)
            vector_size = collection_info.config.params.vectors.size
            print(f"[STATUS] Collection '{collection_name}' has vector size {vector_size}")
            
            # Count points
            count_result = client.count(collection_name=collection_name)
            point_count = count_result.count
            print(f"[STATUS] Collection has {point_count} points")
            
            if point_count == 0:
                return False, f"Collection '{collection_name}' is empty (0 points)"
            
            # If we're not forcing a rebuild, check a sample to ensure quality
            if not force_rebuild and point_count > 0:
                print(f"[STATUS] Checking sample points for quality...")
                
                # Get a small random sample
                sample_size = min(10, point_count)
                try:
                    # Search with random vector to get points
                    random_vector = np.random.rand(vector_size).astype(np.float32)
                    random_vector = random_vector / np.linalg.norm(random_vector)
                    
                    # Get sample points
                    sample = client.search(
                        collection_name=collection_name,
                        query_vector=random_vector.tolist(),
                        limit=sample_size,
                        score_threshold=0.0  # No threshold to get diverse samples
                    )
                    
                    print(f"[STATUS] Retrieved {len(sample)} sample points")
                    
                    # Check if the points have the necessary metadata
                    valid_points = 0
                    required_keys = ["bbox", "region_id", "image_source", "phrase"]
                    for point in sample:
                        if all(key in point.payload for key in required_keys):
                            valid_points += 1
                        else:
                            missing = [key for key in required_keys if key not in point.payload]
                            print(f"[WARNING] Point {point.id} is missing required metadata: {missing}")
                    
                    validity_percentage = (valid_points / len(sample)) * 100 if sample else 0
                    print(f"[STATUS] {validity_percentage:.1f}% of sampled points have valid metadata")
                    
                    if validity_percentage < 50:
                        print(f"[WARNING] Less than 50% of points have valid metadata")
                        if force_rebuild:
                            print(f"[STATUS] Will rebuild collection due to force_rebuild flag")
                        else:
                            return False, f"Database quality check failed: only {validity_percentage:.1f}% of points have valid metadata"
                    
                except Exception as e:
                    print(f"[ERROR] Error sampling points: {e}")
                    return False, f"Failed to sample points: {e}"
            
            # If force_rebuild is True, rebuild the collection
            if force_rebuild:
                print(f"[STATUS] Rebuilding collection '{collection_name}'...")
                
                # Create a backup first
                backup_name = f"{collection_name}_backup_{int(time.time())}"
                try:
                    print(f"[STATUS] Creating backup collection '{backup_name}'...")
                    client.create_collection(
                        collection_name=backup_name,
                        vectors_config=models.VectorParams(
                            size=vector_size,
                            distance=models.Distance.COSINE
                        )
                    )
                    
                    # Copy all points to the backup collection
                    batch_size = 100
                    offset = 0
                    
                    while offset < point_count:
                        # Scroll to get a batch of points
                        scroll_result = client.scroll(
                            collection_name=collection_name,
                            limit=batch_size,
                            offset=offset
                        )
                        
                        points = scroll_result[0]
                        if not points:
                            break
                            
                        # Convert to point structs
                        point_structs = []
                        for point in points:
                            point_structs.append(
                                models.PointStruct(
                                    id=point.id,
                                    vector=point.vector,
                                    payload=point.payload
                                )
                            )
                        
                        # Insert into backup
                        client.upsert(
                            collection_name=backup_name,
                            points=point_structs
                        )
                        
                        offset += len(points)
                        print(f"[STATUS] Backed up {offset}/{point_count} points...")
                    
                    print(f"[STATUS] Successfully created backup in collection '{backup_name}'")
                    
                    # Now rebuild the original collection
                    print(f"[STATUS] Deleting original collection '{collection_name}'...")
                    client.delete_collection(collection_name=collection_name)
                    
                    print(f"[STATUS] Recreating collection '{collection_name}'...")
                    client.create_collection(
                        collection_name=collection_name,
                        vectors_config=models.VectorParams(
                            size=vector_size,
                            distance=models.Distance.COSINE
                        )
                    )
                    
                    # Restore points from backup with better error handling
                    batch_size = 100
                    offset = 0
                    restored_count = 0
                    
                    while True:
                        try:
                            # Scroll to get a batch of points
                            scroll_result = client.scroll(
                                collection_name=backup_name,
                                limit=batch_size,
                                offset=offset
                            )
                            
                            points = scroll_result[0]
                            if not points:
                                break
                                
                            # Convert to point structs
                            point_structs = []
                            for point in points:
                                point_structs.append(
                                    models.PointStruct(
                                        id=point.id,
                                        vector=point.vector,
                                        payload=point.payload
                                    )
                                )
                            
                            # Insert into original
                            client.upsert(
                                collection_name=collection_name,
                                points=point_structs
                            )
                            
                            restored_count += len(points)
                            offset += len(points)
                            print(f"[STATUS] Restored {restored_count}/{point_count} points...")
                            
                        except Exception as e:
                            print(f"[ERROR] Error restoring batch at offset {offset}: {e}")
                            offset += batch_size  # Skip problematic batch
                    
                    print(f"[STATUS] Rebuild complete. Restored {restored_count}/{point_count} points.")
                    
                    if restored_count < point_count:
                        print(f"[WARNING] Some points ({point_count - restored_count}) could not be restored.")
                        return True, f"Collection rebuilt with partial success. Only {restored_count}/{point_count} points restored. Original data is preserved in '{backup_name}'."
                    
                    return True, f"Collection '{collection_name}' successfully rebuilt. Original data is preserved in '{backup_name}'."
                    
                except Exception as e:
                    print(f"[ERROR] Error during rebuild: {e}")
                    return False, f"Failed to rebuild collection: {e}"
            
            # If we got here without force_rebuild, the collection is valid
            return True, f"Collection '{collection_name}' verified successfully. {point_count} points with vector size {vector_size}."
            
        except Exception as e:
            print(f"[ERROR] Error getting collection info: {e}")
            return False, f"Failed to get collection info: {e}"
            
    except Exception as e:
        print(f"[ERROR] Verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False, f"Verification failed: {e}"

if __name__ == "__main__":
    main()