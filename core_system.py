# Core System: SimpleReverso Class
# This file contains the main logic for the visual investigation system.

import sys
sys.path.append('./perception_models') # For pe and transforms

import os
import time
import uuid
import tempfile # Keep for now, might be used by GroundedSAM or other parts
from pathlib import Path # Keep for now

import numpy as np
import cv2
import torch
# import matplotlib.pyplot as plt # No longer used directly for visualization
from PIL import Image, ImageDraw, ImageFont

# Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http import models

import shutil
# import hashlib # No longer used in this file
# import urllib.parse # No longer used in this file

# Perception models
import core.vision_encoder.pe as pe
import core.vision_encoder.transforms as transforms

# GroundedSAM
from autodistill_grounded_sam import GroundedSAM
from autodistill.detection import CaptionOntology

# Supervision
from supervision.detection.core import Detections

# Gradio - to be checked if needed directly, or if progress objects can be passed
# import gradio as gr # Commented out for now, will add if strictly necessary

# class SimpleReverso: # Placeholder removed
#     pass

class SimpleReverso:
    """Simplified visual investigation system"""

    def __init__(self):
        print("🚀 Initializing Simple Revers-o...")

        # Setup device
        self.device = self.setup_device()

        # Load optimized PE model (L14-336, layer 24)
        self.pe_model, self.preprocess = self.load_pe_model()

        # Initialize GroundedSAM
        self.grounded_sam = None

        # Vector database
        self.vector_db = None
        self.current_database = None

        # State
        self.detected_regions = []
        self.region_embeddings = []

        print("✅ Simple Revers-o ready!")

    def list_databases(self):
        """List all available databases"""
        db_path = "./simple_reverso_db"
        if not os.path.exists(db_path):
            return []

        databases = []
        for name in os.listdir(db_path):
            db_dir = os.path.join(db_path, name)
            if os.path.isdir(db_dir):
                # Check if it's a valid database by looking for meta.json
                # For simplicity, we'll assume any folder is a DB for now
                # A more robust check might involve reading meta.json if it exists
                databases.append(name)
        return databases

    def load_database(self, database_name):
        """Load an existing database"""
        if not database_name:
            return "❌ Please provide a database name"

        db_path = f"./simple_reverso_db/{database_name}"
        if not os.path.exists(db_path):
            return f"❌ Database not found: {database_name}"

        try:
            client = QdrantClient(path=db_path)
            collection_name = f"simple_reverso_{database_name}"

            # Verify collection exists
            collections = client.get_collections().collections
            if not any(c.name == collection_name for c in collections):
                # Attempt to handle legacy database names if any
                if any(c.name == database_name for c in collections): # Legacy check
                    collection_name = database_name
                else:
                    return f"❌ Collection not found in database: {database_name}"

            # Store database info
            self.vector_db = client
            self.current_database = collection_name

            return f"✅ Loaded database: {database_name}"

        except Exception as e:
            return f"❌ Error loading database: {str(e)}"

    def delete_database(self, database_name):
        """Delete a database"""
        if not database_name:
            return "❌ Please provide a database name"

        db_path = f"./simple_reverso_db/{database_name}"
        if not os.path.exists(db_path):
            return f"❌ Database not found: {database_name}"

        try:
            # Remove the database directory
            shutil.rmtree(db_path)
            return f"✅ Deleted database: {database_name}"
        except Exception as e:
            return f"❌ Error deleting database: {str(e)}"

    def unlock_database(self, database_name):
        """Remove the lock file from a database"""
        if not database_name:
            return "❌ Please provide a database name"

        db_path = f"./simple_reverso_db/{database_name}"
        if not os.path.exists(db_path):
            return f"❌ Database not found: {database_name}"

        lock_file = os.path.join(db_path, ".lock")
        if os.path.exists(lock_file):
            try:
                os.remove(lock_file)
                return f"✅ Removed lock file from database: {database_name}"
            except Exception as e:
                return f"❌ Error removing lock file: {str(e)}"
        else:
            return f"ℹ️ No lock file found for database: {database_name}"

    def setup_device(self):
        """Setup optimal compute device"""
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            print(f"🔥 Using MPS: {device}")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"🔥 Using CUDA: {device}")
        else:
            device = torch.device("cpu")
            print(f"💻 Using CPU: {device}")
        return device

    def load_pe_model(self):
        """Load PE-Core-L14-336 - optimal for investigation tasks"""
        print("📚 Loading PE-Core-L14-336 (optimal for investigation)...")

        available_configs = pe.CLIP.available_configs()
        print(f"Available PE configs: {available_configs}")

        # Target PE-Core-L14-336 (1B params, layer 24 optimal)
        target_model = "PE-Core-L14-336"

        if target_model in available_configs:
            try:
                pe_model = pe.CLIP.from_config(target_model, pretrained=True)
                print(f"✅ Loaded {target_model}")
            except Exception as e:
                print(f"❌ Failed to load {target_model}: {e}")
                # Fallback to first available
                pe_model = pe.CLIP.from_config(available_configs[0], pretrained=True)
                print(f"🔄 Using fallback: {available_configs[0]}")
        else:
            # Use first available
            pe_model = pe.CLIP.from_config(available_configs[0], pretrained=True)
            print(f"🔄 Using available: {available_configs[0]}")

        # Move to device and optimize
        pe_model = pe_model.to(self.device)
        if self.device.type == 'cuda':
            pe_model = pe_model.half()  # Mixed precision
            print("⚡ Mixed precision enabled")

        # Preprocessing for 336px (L14 optimal)
        preprocess = transforms.get_image_transform(336)

        print(f"🎯 PE model ready - using layer 24 (research optimal)")
        return pe_model, preprocess

    def init_grounded_sam(self, text_prompt):
        """Initialize GroundedSAM with text prompt"""
        if self.grounded_sam is None:
            # Parse prompts
            prompts = [p.strip() for p in text_prompt.split('.') if p.strip()]
            if not prompts:
                prompts = ["object"]

            # Create ontology
            ontology_dict = {prompt: prompt for prompt in prompts}
            ontology = CaptionOntology(ontology_dict)

            # Initialize GroundedSAM with correct parameters
            self.grounded_sam = GroundedSAM(
                ontology=ontology,
                box_threshold=0.35,
                text_threshold=0.25
            )
            print(f"🎯 GroundedSAM ready with prompts: {prompts}")
            print(f"[DEBUG] GroundedSAM version: {self.grounded_sam.__version__ if hasattr(self.grounded_sam, '__version__') else 'unknown'}")
            # print(f"[DEBUG] GroundedSAM config: {self.grounded_sam.__dict__}") # Can be very verbose

            # Print available methods and attributes
            # print(f"[DEBUG] Available methods: {[m for m in dir(self.grounded_sam) if not m.startswith('_')]}")

            # Print predict method signature
            if hasattr(self.grounded_sam, 'predict'):
                import inspect
                # print(f"[DEBUG] Predict method signature: {inspect.signature(self.grounded_sam.predict)}")

    def detect_regions(self, image, text_prompt="person . car . building"):
        """Detect regions using GroundedSAM"""
        print(f"🔍 Detecting regions with prompt: '{text_prompt}'")

        # Initialize GroundedSAM if needed
        self.init_grounded_sam(text_prompt)

        # Save image temporarily for GroundedSAM
        # Ensure temp_path is unique enough if multiple instances run
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, f"temp_image_{uuid.uuid4().hex[:8]}.jpg")

        if isinstance(image, np.ndarray):
            image_pil = Image.fromarray(image)
        elif isinstance(image, str): # If filepath is passed
             image_pil = Image.open(image)
        else: # Assuming PIL image
            image_pil = image

        image_pil.convert("RGB").save(temp_path) # Ensure RGB for GroundedSAM

        try:
            # Detect with GroundedSAM
            detections = self.grounded_sam.predict(temp_path)
            print(f"✅ Found {len(detections)} regions")
            # print(f"[DEBUG] Detection type: {type(detections)}")
            # print(f"[DEBUG] Detection attributes: {dir(detections)}")

            # Extract boxes and masks
            boxes = detections.xyxy
            class_ids = detections.class_id
            confidences = detections.confidence

            masks = None
            if hasattr(detections, 'mask') and detections.mask is not None:
                masks = detections.mask
            elif hasattr(detections, 'masks') and detections.masks is not None: # Common variation
                masks = detections.masks
            elif hasattr(detections, 'data') and 'masks' in detections.data:
                masks = detections.data['masks']

            # print(f"[DEBUG] Boxes shape: {boxes.shape if hasattr(boxes, 'shape') else 'no shape'}")
            # print(f"[DEBUG] Masks type: {type(masks)}")
            # if masks is not None: print(f"[DEBUG] Masks shape: {masks.shape if hasattr(masks, 'shape') else 'no shape'}")
            # print(f"[DEBUG] Class IDs: {class_ids}")
            # print(f"[DEBUG] Confidences: {confidences}")

            if isinstance(masks, torch.Tensor):
                masks = masks.detach().cpu().numpy()
                # print(f"[DEBUG] Converted masks to numpy array")

            self.detected_regions = Detections(
                xyxy=boxes,
                mask=masks, # This expects numpy array or None
                confidence=confidences,
                class_id=class_ids
            )

            # print(f"[DEBUG] Created detections type: {type(self.detected_regions)}")
            # if hasattr(self.detected_regions, 'mask') and self.detected_regions.mask is not None:
            #     print(f"[DEBUG] Created detections mask shape: {self.detected_regions.mask.shape if hasattr(self.detected_regions.mask, 'shape') else 'no shape'}")

            return len(detections)

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def extract_embeddings(self, image):
        """Extract PE embeddings from detected regions"""
        if not self.detected_regions or len(self.detected_regions) == 0:
            print("❌ No regions detected")
            return [], []

        print(f"🧠 Extracting PE embeddings from {len(self.detected_regions)} regions...")

        if isinstance(image, np.ndarray):
            image_pil = Image.fromarray(image)
        elif isinstance(image, str): # If filepath is passed
             image_pil = Image.open(image)
        else: # Assuming PIL image
            image_pil = image

        image_tensor = self.preprocess(image_pil.convert("RGB")).unsqueeze(0).to(self.device) # Ensure RGB

        embeddings = []
        metadata = []

        with torch.no_grad():
            features = self.pe_model.encode_image(image_tensor)
            # print(f"✅ Using standard PE encoding (still very effective)")
            # print(f"[DEBUG] Raw features shape: {features.shape}")

            if len(features.shape) == 3:
                global_embedding = features.mean(dim=1)
                # print(f"[DEBUG] Using global embedding approach due to token format")
            elif len(features.shape) == 2:
                global_embedding = features
                # print(f"[DEBUG] Global embedding shape: {global_embedding.shape}")
            else:
                print(f"[ERROR] Unexpected feature shape: {features.shape}")
                return [], []

            try:
                ontology_class_names = self.grounded_sam.ontology.classes()
            except Exception:
                try:
                    ontology_class_names = list(self.grounded_sam.ontology.prompts_to_classes.keys())
                except:
                    ontology_class_names = ["object"]

            for i in range(min(len(self.detected_regions), 50)): # Limit processing for safety
                try:
                    if hasattr(self.detected_regions, 'mask') and self.detected_regions.mask is not None and i < len(self.detected_regions.mask):
                        mask_np = self.detected_regions.mask[i]
                    else:
                        print(f"⚠️ No mask available for detection {i}, using global embedding.")
                        # Fallback: use global image embedding if mask is missing
                        region_embedding = global_embedding[0]
                        bbox = [0, 0, image_pil.width, image_pil.height] # Full image bbox
                        area_ratio = 1.0
                        # Continue to append this global embedding for the detection
                        # This ensures we have an embedding for every detection, even if mask processing fails
                        # but metadata might be less precise
                        # Get confidence & class if available
                        raw_confidence = float(self.detected_regions.confidence[i]) if hasattr(self.detected_regions, 'confidence') and i < len(self.detected_regions.confidence) else 0.0
                        class_id = int(self.detected_regions.class_id[i]) if hasattr(self.detected_regions, 'class_id') and i < len(self.detected_regions.class_id) else -1
                        detected_class = ontology_class_names[class_id] if 0 <= class_id < len(ontology_class_names) else "unknown"

                        embeddings.append((region_embedding / region_embedding.norm()).cpu())
                        meta = {
                            "region_id": str(uuid.uuid4()), "bbox": bbox, "area_ratio": area_ratio,
                            "detection_index": i, "confidence": raw_confidence, "detected_class": detected_class,
                            "mask_status": "missing_or_unavailable"
                        }
                        metadata.append(meta)
                        print(f"✅ Extracted global embedding for region {i} due to missing mask.")
                        continue


                    raw_confidence = float(self.detected_regions.confidence[i]) if hasattr(self.detected_regions, 'confidence') and i < len(self.detected_regions.confidence) else 0.0
                    class_id = int(self.detected_regions.class_id[i]) if hasattr(self.detected_regions, 'class_id') and i < len(self.detected_regions.class_id) else -1
                    detected_class = ontology_class_names[class_id] if 0 <= class_id < len(ontology_class_names) else "object"

                    # print(f"[STATUS] Processing detection {i+1}: {detected_class} (Confidence: {raw_confidence:.3f})")

                    if mask_np.dtype == bool: mask_processed = mask_np.astype(np.uint8)
                    elif np.issubdtype(mask_np.dtype, np.floating): mask_processed = (mask_np > 0.5).astype(np.uint8)
                    else: mask_processed = mask_np.astype(np.uint8)

                    if np.sum(mask_processed) == 0:
                        print(f"⚠️ Empty mask for region {i}, skipping")
                        continue

                    region_embedding = global_embedding[0] # Use global for now
                    region_embedding = region_embedding / region_embedding.norm()
                    embeddings.append(region_embedding.cpu())

                    y_indices, x_indices = np.where(mask_processed)
                    bbox = [int(x_indices.min()), int(y_indices.min()), int(x_indices.max()), int(y_indices.max())]

                    meta = {
                        "region_id": str(uuid.uuid4()), "bbox": bbox,
                        "area_ratio": float(np.sum(mask_processed) / mask_processed.size),
                        "detection_index": i, "confidence": raw_confidence, "detected_class": detected_class,
                        "mask_status": "processed"
                    }
                    metadata.append(meta)
                    # print(f"✅ Extracted embedding for region {i}")

                except Exception as e:
                    print(f"❌ Error processing region {i}: {e}")
                    import traceback; traceback.print_exc()
                    continue

        self.region_embeddings = embeddings
        print(f"🎯 Extracted {len(embeddings)} region embeddings")
        return embeddings, metadata

    def process_image_direct_pe(self, image):
        """Process image directly with PE, without GroundedSAM"""
        print("🧠 Processing image directly with PE...")

        if isinstance(image, np.ndarray): image_pil = Image.fromarray(image)
        elif isinstance(image, str): image_pil = Image.open(image)
        else: image_pil = image # Assuming PIL Image

        image_tensor = self.preprocess(image_pil.convert("RGB")).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.pe_model.encode_image(image_tensor)
            if len(features.shape) == 3: embedding = features.mean(dim=1)[0]
            elif len(features.shape) == 2: embedding = features[0]
            else: raise ValueError(f"Unexpected feature shape: {features.shape}")

            embedding = embedding / embedding.norm()
            self.region_embeddings = [embedding.cpu()]

            meta = {
                "region_id": str(uuid.uuid4()), "bbox": [0, 0, image_pil.width, image_pil.height],
                "area_ratio": 1.0, "detection_index": 0, "confidence": 1.0, "detected_class": "full_image"
            }
            print("✅ Extracted global image embedding")
            return [embedding.cpu()], [meta]

    def create_database(self, folder_path, database_name, text_prompt="person . car . building", use_direct_pe=False, progress_callback=None):
        """Create searchable database from image folder"""
        status_messages = []
        def log_status(message):
            status_messages.append(message)
            if progress_callback: progress_callback(message) # Call external progress update
            # print(message) # Optional: also print to console
            return "\n".join(status_messages) # This return is for Gradio Textbox usually

        log_status(f"📁 Creating database '{database_name}' from {folder_path}")
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if any(f.lower().endswith(ext) for ext in image_extensions)]

        if not image_files: return log_status(f"❌ No images found in {folder_path}")

        log_status(f"📊 Found {len(image_files)} images to process")
        log_status(f"🔧 Processing mode: {'Direct PE' if use_direct_pe else 'GroundedSAM + PE'}")

        db_base_path = "./simple_reverso_db"
        os.makedirs(db_base_path, exist_ok=True)
        db_path = os.path.join(db_base_path, database_name)
        # os.makedirs(db_path, exist_ok=True) # QdrantClient creates if not exists for path
        log_status(f"📂 Database will be stored at: {db_path}")

        client = QdrantClient(path=db_path)
        all_embeddings, all_metadata = [], []
        processed, failed = 0, 0

        for i, image_path in enumerate(image_files):
            filename = os.path.basename(image_path)
            log_status(f"🔄 Processing {i+1}/{len(image_files)}: {filename}")
            try:
                image = Image.open(image_path).convert("RGB")
                if use_direct_pe:
                    embeddings, metadata_list = self.process_image_direct_pe(image)
                    log_status(f"✅ Extracted global embedding for {filename}")
                else:
                    num_regions = self.detect_regions(image, text_prompt)
                    if num_regions > 0:
                        embeddings, metadata_list = self.extract_embeddings(image)
                        log_status(f"✅ Found {num_regions} regions, extracted {len(embeddings)} embeddings in {filename}")
                    else:
                        log_status(f"⚠️ No regions found in {filename}, skipping")
                        failed += 1; continue

                for k, meta_item in enumerate(metadata_list):
                    meta_item["image_source"] = image_path # Store full path
                    meta_item["filename"] = filename       # Store filename
                    # Ensure region_id is unique if multiple embeddings per image
                    if len(metadata_list) > 1:
                         meta_item["region_id"] = f"{meta_item.get('region_id', uuid.uuid4())}_{k}"

                all_embeddings.extend(embeddings)
                all_metadata.extend(metadata_list)
                processed += 1
            except Exception as e:
                log_status(f"❌ Error processing {filename}: {str(e)}")
                import traceback; traceback.print_exc()
                failed += 1; continue

        if not all_embeddings: return log_status(f"❌ No embeddings extracted from any images")

        vector_dim = all_embeddings[0].shape[0]
        collection_name = f"simple_reverso_{database_name}"

        try:
            client.recreate_collection( # Use recreate_collection to start fresh
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=vector_dim, distance=models.Distance.COSINE)
            )
            log_status(f"📦 Recreated collection: {collection_name}")
        except Exception as e:
             log_status(f"ℹ️ Note: Collection {collection_name} might already exist or error: {e}")

        points = [models.PointStruct(id=meta["region_id"], vector=emb.cpu().numpy().tolist(), payload=meta) for emb, meta in zip(all_embeddings, all_metadata)]

        # Batch upsert for efficiency
        batch_size = 100
        for j in range(0, len(points), batch_size):
            batch_points = points[j:j+batch_size]
            client.upsert(collection_name=collection_name, points=batch_points)
            log_status(f"💾 Stored batch {j//batch_size + 1}/{(len(points) + batch_size -1)//batch_size} ({len(batch_points)} points)")

        self.vector_db = client
        self.current_database = collection_name

        log_status("\n📊 Final Summary:")
        log_status(f"✅ Successfully processed: {processed} images")
        if failed > 0: log_status(f"⚠️ Failed to process: {failed} images")
        log_status(f"🔍 Total embeddings stored: {len(all_embeddings)}")
        log_status(f"🎯 Database '{database_name}' ready for searching!")
        return "\n".join(status_messages) # Return final combined status

    def search_similar(self, similarity_threshold=0.7, max_results=5):
        """Search for similar regions in database"""
        if not self.region_embeddings: return "❌ No query embeddings available. Please detect/process an image first.", []
        if not self.vector_db or not self.current_database: return "❌ No database loaded. Please create or load a database first.", []

        print(f"🔍 Searching for similar regions (threshold={similarity_threshold}, max_results={max_results})")

        query_embedding = self.region_embeddings[0] # Use first detected/processed region for search

        search_results = self.vector_db.search(
            collection_name=self.current_database,
            query_vector=query_embedding.cpu().numpy().tolist(),
            limit=max_results,
            score_threshold=similarity_threshold
        )

        if not search_results: return f"❌ No similar regions found above threshold {similarity_threshold}", []

        results_text = f"🎯 Found {len(search_results)} similar regions:\n\n"
        similar_items_data = [] # Store (image_object, score, filename, bbox)

        for i, result in enumerate(search_results):
            payload = result.payload
            filename = payload.get("filename", "Unknown")
            score = result.score
            bbox_str = str(payload.get("bbox", "[0,0,0,0]")) # Ensure string for display
            image_path = payload.get("image_source", "")

            results_text += f"{i+1}. {filename} (Similarity: {score:.3f})\n"
            results_text += f"   Source: {image_path}\n"
            results_text += f"   📍 Bounding box: {bbox_str}\n\n"

            if os.path.exists(image_path):
                try:
                    img = Image.open(image_path).convert("RGB")
                    # Potentially draw bbox or crop, then add text
                    # For now, just load the image and add score.
                    # If bbox is available and not full image, consider cropping:
                    # bbox = payload.get("bbox") # original list
                    # if bbox and bbox != [0, 0, img.width, img.height]:
                    #     img = img.crop(tuple(bbox))

                    img_with_text = img.copy()
                    draw = ImageDraw.Draw(img_with_text)
                    font_size = max(15, int(min(img.height, img.width) * 0.05))
                    try: font = ImageFont.truetype("Arial.ttf", font_size) # Try to load Arial
                    except IOError: font = ImageFont.load_default() # Fallback

                    text = f"Score: {score:.3f}"
                    text_bbox = draw.textbbox((5, 5), text, font=font) # Use textbbox for background
                    draw.rectangle([text_bbox[0]-2, text_bbox[1]-2, text_bbox[2]+2, text_bbox[3]+2], fill="black")
                    draw.text((5, 5), text, fill="white", font=font)

                    # Resize for display if too large
                    img_with_text.thumbnail((400, 400), Image.Resampling.LANCZOS)
                    similar_items_data.append({
                        "image": img_with_text, "score": score,
                        "filename": filename, "bbox": payload.get("bbox")
                    })
                except Exception as e:
                    print(f"❌ Error loading/processing image {image_path}: {e}")
                    similar_items_data.append({"image": None, "score": score, "filename": filename, "bbox": payload.get("bbox")}) # Keep placeholder
            else:
                print(f"❌ Image not found: {image_path}")
                similar_items_data.append({"image": None, "score": score, "filename": filename, "bbox": payload.get("bbox")})

        print(f"📊 Processed {len(similar_items_data)} search results for display.")
        return results_text, similar_items_data # Return text summary and list of dicts

    def visualize_detections(self, image, selected_region_index=None):
        """Create visualization of detected regions, highlighting the selected region if provided"""
        if not self.detected_regions or not hasattr(self.detected_regions, 'mask') or self.detected_regions.mask is None:
            print("ℹ️ No detections or masks to visualize.")
            if isinstance(image, np.ndarray): return Image.fromarray(image.astype(np.uint8)) # Return original if numpy
            return image # Return original PIL image if no masks

        if isinstance(image, np.ndarray): img_np = image.copy()
        elif isinstance(image, str): img_np = np.array(Image.open(image).convert('RGB'))
        else: img_np = np.array(image.convert('RGB')) # PIL to Numpy

        if len(img_np.shape) == 2: img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        elif img_np.shape[2] == 4: img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)

        # Create a copy for drawing
        overlay_img = img_np.copy()

        for i, mask_np in enumerate(self.detected_regions.mask):
            if mask_np is None: continue

            color = (0, 255, 0) if i == selected_region_index else (255, 0, 0) # Green if selected, Red otherwise
            linewidth = 3 if i == selected_region_index else 1

            # Ensure mask is binary and correct type for findContours
            binary_mask = (mask_np > 0.5).astype(np.uint8)
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay_img, contours, -1, color, linewidth)

            # Add label text (region number)
            if contours:
                # Get bounding box of the largest contour to place text
                # Or just use the mean of the mask if simple
                y_indices, x_indices = np.where(binary_mask)
                if y_indices.size > 0 and x_indices.size > 0:
                    center_x, center_y = int(x_indices.mean()), int(y_indices.mean())
                    cv2.putText(overlay_img, f"{i+1}", (center_x, center_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color if i != selected_region_index else (0,0,0), 2)

        return Image.fromarray(overlay_img) # Convert back to PIL for Gradio
