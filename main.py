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
        self.update_available_databases()
    
    def update_available_databases(self):
        """Update the list of available databases"""
        self.available_databases = list_available_databases()
        return self.available_databases

def list_available_databases():
    """Find all existing Qdrant collections in the project directory"""
    # Check both possible directories - older Qdrant uses 'collection', newer might use 'collections'
    collections = []
    
    # Path 1: Check the 'collection' directory (singular)
    collection_dir = os.path.join("image_retrieval_project", "qdrant_data", "collection")
    if os.path.exists(collection_dir):
        collections.extend([d for d in os.listdir(collection_dir) if os.path.isdir(os.path.join(collection_dir, d))])
        print(f"[STATUS] Found collections in 'collection' directory: {collections}")
    else:
        print(f"[STATUS] Collection directory not found at: {collection_dir}")
    
    # Path 2: Check the 'collections' directory (plural)
    collections_dir = os.path.join("image_retrieval_project", "qdrant_data", "collections")
    if os.path.exists(collections_dir):
        plural_collections = [d for d in os.listdir(collections_dir) if os.path.isdir(os.path.join(collections_dir, d))]
        collections.extend(plural_collections)
        print(f"[STATUS] Found collections in 'collections' directory: {plural_collections}")
    
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
                print(f"[STATUS] Found collections via Qdrant API: {collections}")
            except Exception as e:
                print(f"[STATUS] Error querying Qdrant API: {e}")
    
    # Remove duplicates while preserving order
    unique_collections = []
    for c in collections:
        if c not in unique_collections:
            unique_collections.append(c)
    
    if not unique_collections:
        print(f"[STATUS] No collections found in any directory")
    else:
        print(f"[STATUS] Final available collections: {unique_collections}")
    
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
            client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE
                )
            )
            print(f"Created new collection: {collection_name}")
        else:
            print(f"Using existing collection: {collection_name}")

        return client
    except Exception as e:
        print(f"Error in setup_qdrant: {e}")
        return None

def store_embeddings_in_qdrant(client, collection_name, embeddings, metadata_list):
    """Store embeddings in Qdrant collection"""
    try:
        embedding_vectors = [emb.detach().cpu().numpy().tolist() for emb in embeddings]
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

        # Upsert points in batches
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i+batch_size]
            client.upsert(
                collection_name=collection_name,
                points=batch
            )

        client.get_collection(collection_name)

        print(f"Stored {len(points)} embeddings in Qdrant collection: {collection_name}")
        return point_ids
    except Exception as e:
        print(f"Error in store_embeddings_in_qdrant: {e}")
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

                    client = setup_qdrant(collection_name, vector_size)
                    if client is None:
                        print("❌ Failed to initialize Qdrant client.")
                        return None

                    initialization_complete = True

                # Store embeddings
                store_embeddings_in_qdrant(client, collection_name, embeddings, metadata)
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
        
        # Update Gradio with initial status
        yield (f"🔍 Starting to process {total} images in {folder_path}\n"
              f"📊 Parameters:\n"
              f"  - Semantic Layer: {optimal_layer}\n"
              f"  - Min Region Size: {min_area_ratio}\n"
              f"  - Max Regions: {max_regions}\n"
              f"  - Resume from checkpoint: {resume_from_checkpoint}"), gr.update(visible=False)
        
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
                    f"🔍 Total regions found: {total_regions}\n\n"
                    f"🧠 Using semantic layer: {optimal_layer}\n"
                    f"🔍 Min region size: {min_area_ratio}\n"
                    f"📏 Max regions per image: {max_regions}"
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
            previous = self.active_collection
            self.active_collection = collection_name
            print(f"[STATUS] Changed active collection from {previous} to {collection_name}")
            
            # Close existing client if database changed
            if self.active_client is not None:
                print(f"[STATUS] Closing previous database connection")
                self.active_client.close()
                self.active_client = None
                
            return True
        return False

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

    def search_region(self, region_selection, similarity_threshold=0.5, max_results=5):
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
            
            collection_name = self.active_collection
            print(f"[STATUS] Searching in collection: {collection_name}")
            
            # Verify collection exists
            try:
                collections = self.active_client.get_collections().collections
                collection_names = [collection.name for collection in collections]
                if collection_name not in collection_names:
                    print(f"[ERROR] Collection {collection_name} not found. Available collections: {collection_names}")
                    if collection_names:
                        # Auto-switch to an available collection
                        collection_name = collection_names[0]
                        self.active_collection = collection_name
                        print(f"[STATUS] Auto-switching to available collection: {collection_name}")
                    else:
                        return None, f"Error: Collection '{collection_name}' not found and no other collections are available.", gr.update(visible=False), gr.update(choices=[], value=None)
            except Exception as e:
                print(f"[ERROR] Error checking collections: {e}")
            
            # Convert embedding to list for Qdrant
            print(f"[STATUS] Converting embedding to list format...")
            embedding_list = embedding.cpu().numpy().tolist()
            
            # Search for similar regions
            print(f"[STATUS] Executing search with threshold {similarity_threshold}, max results {max_results}...")
            search_results = self.active_client.search(
                collection_name=collection_name,
                query_vector=embedding_list,
                limit=max_results,
                score_threshold=similarity_threshold
            )
            
            print(f"[STATUS] Search complete, found {len(search_results)} results")
            
            if not search_results:
                print(f"[STATUS] No similar regions found above threshold {similarity_threshold}")
                return None, f"No similar regions found with similarity threshold {similarity_threshold}.", gr.update(visible=False), gr.update(choices=[], value=None)

            # Process search results
            print(f"[STATUS] Creating visualization for {len(search_results)} search results...")
            return self.create_search_results_visualization(search_results, region_idx)
        
        except Exception as e:
            error_message = f"Error searching for similar regions: {str(e)}"
            print(f"[ERROR] {error_message}")
            import traceback
            traceback.print_exc()
            return None, error_message, gr.update(visible=False), gr.update(choices=[], value=None)

    def create_search_results_visualization(self, filtered_results, query_region_idx):
        """Create visualization of search results"""
        print(f"[STATUS] Creating search results visualization for {len(filtered_results)} results...")
        self.search_result_images = []

        # Get query region info
        query_label = self.detected_regions["labels"][query_region_idx]
        query_mask = self.detected_regions["masks"][query_region_idx]
        query_image = self.detected_regions["image"]
        
        print(f"[STATUS] Query region: {query_label}")
        
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
        
        # Prepare figure for visualization - use a better layout system
        print(f"[STATUS] Setting up visualization figure with {grid_rows}x{grid_cols} grid...")
        fig = plt.figure(figsize=(fig_width, fig_height))
        
        # Create a layout with a dedicated area for query and proper spacing
        gs = gridspec.GridSpec(grid_rows + 1, grid_cols, height_ratios=[1] + [3] * grid_rows)
        
        # Query section (top row, spans all columns)
        ax_query = fig.add_subplot(gs[0, :])
        query_img = self.create_region_preview(query_image, query_mask)
        ax_query.imshow(query_img)
        ax_query.set_title(f"Query Region: {query_label}", fontsize=14, pad=10)
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
                bbox = metadata.get("bbox", [0, 0, 100, 100])
                phrase = metadata.get("phrase", "Unknown")
                score = result.score
                
                print(f"[STATUS] Result {i+1} - Source: {os.path.basename(image_source) if isinstance(image_source, str) else 'embedded_image'}, Score: {score:.4f}")
                
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
                    img = np.array(load_local_image(image_source))
                    self.image_cache[image_source] = img

                # Extract region using bbox
                x_min, y_min, x_max, y_max = bbox
                print(f"[STATUS] Extracting region from bbox: {bbox}")
                region = img[y_min:y_max, x_min:x_max]
                
                # Display region with improved labeling
                ax.imshow(region)
                
                # Create a clearer title with result number and score
                title = f"Result {i+1}: {score:.2f}"
                ax.set_title(title, fontsize=12, pad=5)
                
                # Add filename with better visibility and positioning
                # Use a semi-transparent background for better text readability
                ax.text(0.5, 0.03, f"{filename}", transform=ax.transAxes, 
                        ha='center', va='bottom', fontsize=10, 
                        bbox=dict(facecolor='white', alpha=0.8, pad=3,
                                  edgecolor='lightgray', boxstyle='round'))
                ax.axis('off')

                # Create radio button choice with comprehensive information
                choice_label = f"Result {i+1}: {filename} - {phrase}"
                radio_choices.append(choice_label)
                
                # Add to result info text with more details
                result_info.append(f"**Result {i+1}:** {filename}\n- Phrase: {phrase}\n- Score: {score:.4f}\n")
                
                # Store region for possible enlargement
                self.search_result_images.append({
                    "image": region,
                    "bbox": bbox,
                    "metadata": {
                        "image_source": image_source,
                        "filename": filename,
                        "phrase": phrase,
                        "score": score
                    }
                })

            except Exception as e:
                print(f"[ERROR] Error displaying result {i+1}: {e}")
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
            region = result_data["image"]
            metadata = result_data["metadata"]
            
            # Create a more visually appealing enlarged display
            fig = plt.figure(figsize=(10, 8), constrained_layout=True)
            gs = gridspec.GridSpec(1, 1, figure=fig)
            ax = fig.add_subplot(gs[0, 0])
            
            # Display the image
            ax.imshow(region)
            ax.axis('off')
            
            # Add a comprehensive, well-formatted title
            title = f"Result {result_idx+1}: {metadata['filename']}"
            ax.set_title(title, fontsize=16, pad=15)
            
            # Add detailed metadata information in a visually appealing box
            info_text = (
                f"Phrase: {metadata['phrase']}\n"
                f"Score: {metadata['score']:.4f}"
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
            return None

    def update_region_preview(self, region_selection):
        """Updates the region preview when a region is selected"""
        if region_selection is None or self.detected_regions["image"] is None:
            return None

        region_idx = int(region_selection.split(":")[0].replace("Region ", "")) - 1
        mask = self.detected_regions["masks"][region_idx]
        label = self.detected_regions["labels"][region_idx]

        return self.create_region_preview(self.detected_regions["image"], mask, label)

    def build_interface(self):
        """Build the search interface"""
        with gr.Blocks(title="Grounded SAM Region Search") as interface:
            # Interface title and description
            gr.Markdown("# Grounded SAM Region Search Interface")
            gr.Markdown("Upload an image, specify a text prompt to detect regions, select a region, and search for similar regions in the database.")
            
            # Add status indicator at the top
            status_indicator = gr.Markdown("**Status:** Ready")
            
            with gr.Row():
                with gr.Column(scale=1):
                    # Input components
                    input_image = gr.Image(label="Upload Image", type="pil")
                    text_prompt = gr.Textbox(
                        label="Text Prompt (e.g., 'person, car, building')",
                        placeholder="Enter objects to detect...",
                        value="person, car, building, chair, table"
                    )
                    process_button = gr.Button("Detect Regions")

                    # Region selection
                    region_dropdown = gr.Dropdown(label="Select a Region", choices=[])
                    region_preview = gr.Image(label="Selected Region Preview")

                    # Search controls
                    similarity_slider = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.5,
                        step=0.05,
                        label="Similarity Threshold",
                        info="Only show results with similarity score ≥ this value"
                    )

                    max_results_dropdown = gr.Dropdown(
                        label="Number of Results to Display",
                        choices=[1, 2, 3, 4, 5, 6, 8, 10],
                        value=5,
                        info="Maximum number of similar regions to display"
                    )

                    search_button = gr.Button("Search Similar Regions")
                    
                    # Collection info
                    collection_info = gr.Textbox(
                        label="Active Collection",
                        value=f"Current database: {self.active_collection}",
                        interactive=False
                    )

                with gr.Column(scale=2):
                    # Output components
                    segmented_output = gr.Image(label="Detected Regions")
                    detection_info = gr.Textbox(label="Detection Info")

                    # Search results
                    search_results_output = gr.Image(label="Search Results")
                    search_info = gr.Markdown(label="Search Results Info")

                    # Result selection
                    with gr.Column(visible=False) as results_section:
                        gr.Markdown("## Select Result to View Details")
                        result_dropdown = gr.Dropdown(
                            label="Select a result to see details",
                            choices=[],
                            interactive=True
                        )
                        
                        # Add enlarged result display
                        enlarged_result = gr.Image(label="Enlarged Result View", visible=True)

            # Function to update status indicator
            def update_status(message):
                return f"**Status:** {message}"
                
            # Modified process_image_with_prompt that updates status
            def process_image_with_status(image, text_prompt):
                status_indicator.value = update_status("Detecting regions... please wait")
                result = self.process_image_with_prompt(image, text_prompt)
                status_indicator.value = update_status("Region detection complete")
                return result
                
            # Modified search_region that updates status
            def search_with_status(region_selection, similarity_threshold, max_results):
                status_indicator.value = update_status("Searching for similar regions... please wait")
                result = self.search_region(region_selection, similarity_threshold, max_results)
                status_indicator.value = update_status("Search complete")
                return result

            # Event handlers with status updates
            process_button.click(
                fn=process_image_with_status,
                inputs=[input_image, text_prompt],
                outputs=[segmented_output, detection_info, region_dropdown, region_preview]
            )

            region_dropdown.change(
                fn=self.update_region_preview,
                inputs=[region_dropdown],
                outputs=[region_preview]
            )

            search_button.click(
                fn=search_with_status,
                inputs=[region_dropdown, similarity_slider, max_results_dropdown],
                outputs=[search_results_output, search_info, results_section, result_dropdown]
            )
            
            # Add handler for result selection
            result_dropdown.change(
                self.display_enlarged_result,
                inputs=[result_dropdown],
                outputs=[enlarged_result]
            )

            # Help section at the bottom
            gr.Markdown("## Instructions:")
            gr.Markdown("""
            1. Upload an image using the panel on the left
            2. Enter a text prompt describing objects you want to detect (e.g., 'person, car, building')
            3. Click 'Detect Regions' to process the image (this may take a few moments)
            4. Select one of the detected regions from the dropdown
            5. Adjust the similarity threshold if needed (higher = more similar)
            6. Click 'Search Similar Regions' to find similar regions (this may take a few moments)
            7. View search results and select individual results for details
            """)

        return interface

# =============================================================================
# MULTI-MODE INTERFACE
# =============================================================================

def show_landing_page(app_state=None):
    """Switch to landing page mode"""
    if app_state:
        app_state.mode = "landing"
    return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)

def show_build_interface(app_state=None):
    """Switch to database building mode"""
    if app_state:
        app_state.mode = "building"
    return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)

def show_search_interface(app_state=None):
    """Switch to search interface mode"""
    if app_state:
        app_state.mode = "searching"
    return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)

def select_database(database_name, app_state=None):
    """Select and activate a database for searching"""
    if database_name and database_name != "No databases found" and app_state:
        app_state.active_database = database_name
        return database_name
    return None

def create_multi_mode_interface(pe_model, pe_vit_model, preprocess, device):
    """Create the multi-mode interface with landing page, database builder, and search interface"""
    app_state = AppState()
    search_interface = GradioInterface(pe_model, pe_vit_model, preprocess, device)
    
    with gr.Blocks(title="Grounded SAM Region Search") as demo:
        # Landing Page Section
        with gr.Column(visible=True) as landing_section:
            gr.Markdown("# Grounded SAM Region Search")
            gr.Markdown("### Choose an operation mode:")
            with gr.Row():
                create_db_btn = gr.Button("Create New Database", size="large")
                search_db_btn = gr.Button("Search Existing Database", size="large")
        
        # Database Builder Section
        with gr.Column(visible=False) as builder_section:
            gr.Markdown("# Database Builder")
            with gr.Row():
                folder_input = gr.Textbox(label="Image Folder Path", value="./my_images")
                prompts_input = gr.Textbox(label="Detection Prompts", value="person, building, car, text, object")
            with gr.Row():
                collection_input = gr.Textbox(label="Database Name", value="grounded_image_regions")
            
            # New advanced parameters section
            with gr.Accordion("Advanced Parameters", open=False):
                with gr.Row():
                    optimal_layer = gr.Slider(
                        minimum=1,
                        maximum=50,
                        value=40,
                        step=1,
                        label="Semantic Layer (1-50)",
                        info="Higher layers capture more abstract concepts, lower layers capture more visual details"
                    )
                with gr.Row():
                    min_area_ratio = gr.Slider(
                        minimum=0.001,
                        maximum=0.1,
                        value=0.005,
                        step=0.001,
                        label="Minimum Region Size",
                        info="Smaller values detect smaller regions (as fraction of image size)"
                    )
                with gr.Row():
                    max_regions = gr.Slider(
                        minimum=1,
                        maximum=20,
                        value=5,
                        step=1,
                        label="Maximum Regions Per Image",
                        info="Maximum number of regions to extract from each image"
                    )
                with gr.Row():
                    resume_checkbox = gr.Checkbox(
                        value=True,
                        label="Resume from checkpoint if available",
                        info="Continue from last processed image if interrupted"
                    )
            
            start_btn = gr.Button("Start Processing")
            progress_output = gr.Textbox(label="Progress", interactive=False)
            back_to_menu_btn1 = gr.Button("Back to Main Menu")
            go_to_search_btn = gr.Button("Go to Search", visible=False)
        
        # Search Interface Section
        with gr.Column(visible=False) as search_section:
            gr.Markdown("# Search Interface")
            
            # Get fresh list of available databases
            available_dbs = list_available_databases()
            
            # Database selection at the top with refresh button
            with gr.Row():
                database_dropdown = gr.Dropdown(
                    label="Select Database", 
                    choices=available_dbs if available_dbs else ["No databases found"],
                    value=available_dbs[0] if available_dbs else None,
                    interactive=True
                )
                refresh_dbs_btn = gr.Button("🔄 Refresh Database List")
            
            # Active database indicator
            active_db_indicator = gr.Markdown(
                f"**Active Database:** {search_interface.active_collection if search_interface.active_collection else 'None'}"
            )
            
            # Include search interface components
            with gr.Row():
                with gr.Column(scale=1):
                    input_image = gr.Image(label="Upload Image", type="pil")
                    text_prompt = gr.Textbox(
                        label="Text Prompt (e.g., 'person, car, building')",
                        placeholder="Enter objects to detect...",
                        value="person, car, building, chair, table, computer, phone, book"
                    )
                    process_button = gr.Button("Detect Regions")

                    region_dropdown = gr.Dropdown(label="Select a Region", choices=[])
                    region_preview = gr.Image(label="Selected Region Preview")

                    similarity_slider = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.5,
                        step=0.05,
                        label="Similarity Threshold"
                    )

                    max_results_dropdown = gr.Dropdown(
                        label="Number of Results to Display",
                        choices=[1, 2, 3, 4, 5, 6, 8, 10],
                        value=5
                    )

                    search_button = gr.Button("Search Similar Regions")

                with gr.Column(scale=2):
                    segmented_output = gr.Image(label="Detected Regions")
                    detection_info = gr.Textbox(label="Detection Info")

                    search_results_output = gr.Image(label="Search Results")
                    search_info = gr.Textbox(label="Search Results Info", max_lines=8)

                    with gr.Column(visible=False) as button_section:
                        gr.Markdown("## Enlarge Results")
                        result_selector = gr.Radio(
                            label="Select a result to enlarge",
                            choices=[],
                            value=None
                        )
                        
                        # Add enlarged result display
                        enlarged_result = gr.Image(label="Enlarged Result View", visible=True)
            
            back_to_menu_btn2 = gr.Button("Back to Main Menu")
            
            # Function to refresh database list
            def refresh_databases():
                fresh_dbs = list_available_databases()
                return gr.update(
                    choices=fresh_dbs if fresh_dbs else ["No databases found"],
                    value=fresh_dbs[0] if fresh_dbs else None
                )
            
            # Function to update active database indicator
            def update_active_db_indicator(db_name):
                success = search_interface.set_active_collection(db_name)
                if success:
                    return f"**Active Database:** {db_name}"
                else:
                    return f"**Active Database:** {search_interface.active_collection}"
            
            # Connect event handlers for search interface
            process_button.click(
                search_interface.process_image_with_prompt,
                inputs=[input_image, text_prompt],
                outputs=[segmented_output, detection_info, region_dropdown, region_preview]
            )

            region_dropdown.change(
                search_interface.update_region_preview,
                inputs=[region_dropdown],
                outputs=[region_preview]
            )

            search_button.click(
                search_interface.search_region,
                inputs=[region_dropdown, similarity_slider, max_results_dropdown],
                outputs=[search_results_output, search_info, button_section, result_selector]
            )
            
            # Add handler for result selection in multi-mode interface
            result_selector.change(
                search_interface.display_enlarged_result,
                inputs=[result_selector],
                outputs=[enlarged_result]
            )
            
            # Database selection and refresh handlers
            refresh_dbs_btn.click(
                fn=refresh_databases,
                inputs=None,
                outputs=[database_dropdown]
            )
            
            database_dropdown.change(
                fn=update_active_db_indicator,
                inputs=[database_dropdown],
                outputs=[active_db_indicator]
            )
        
        # Event handlers for mode switching
        create_db_btn.click(
            fn=show_build_interface,
            inputs=None,
            outputs=[landing_section, builder_section, search_section]
        )
        
        search_db_btn.click(
            fn=show_search_interface,
            inputs=None, 
            outputs=[landing_section, builder_section, search_section]
        )
        
        back_to_menu_btn1.click(
            fn=show_landing_page,
            inputs=None,
            outputs=[landing_section, builder_section, search_section]
        )
        
        back_to_menu_btn2.click(
            fn=show_landing_page,
            inputs=None,
            outputs=[landing_section, builder_section, search_section]
        )
        
        # Modified processing functionality with advanced parameters
        start_btn.click(
            fn=process_folder_with_progress_advanced,
            inputs=[
                folder_input, 
                prompts_input, 
                collection_input,
                optimal_layer,
                min_area_ratio,
                max_regions,
                resume_checkbox
            ],
            outputs=[progress_output, go_to_search_btn]
        )
        
        go_to_search_btn.click(
            fn=show_search_interface,
            inputs=None,
            outputs=[landing_section, builder_section, search_section]
        )

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
        demo.launch(share=False, server_name="127.0.0.1", server_port=7860)
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

        client = process_folder_for_region_database(
            folder_path=args.folder,
            collection_name=args.collection,
            text_prompts=args.prompts,
            pe_model=pe_model,
            pe_vit_model=pe_vit_model,
            preprocess=preprocess,
            device=device,
            optimal_layer=40,
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
        demo.launch(share=False, server_name="127.0.0.1", server_port=7860)

if __name__ == "__main__":
    main()