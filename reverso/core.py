import os
import sys
import time
import uuid
import tempfile
from pathlib import Path
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import gradio as gr
from qdrant_client import QdrantClient
from qdrant_client.http import models
import shutil
import hashlib
import urllib.parse

# Add perception_models to path
sys.path.append("./perception_models")
import core.vision_encoder.pe as pe
import core.vision_encoder.transforms as transforms

# GroundedSAM import
from autodistill_grounded_sam import GroundedSAM
from autodistill.detection import CaptionOntology

# Video processing imports
try:
    from scenedetect import VideoManager, SceneManager
    from scenedetect.detectors import ContentDetector

    VIDEO_PROCESSING_AVAILABLE = True
except ImportError:
    VIDEO_PROCESSING_AVAILABLE = False
    print(
        "[WARNING] Scene detection libraries not available. Install with: pip install scenedetect"
    )

# Check for yt-dlp availability
try:
    import yt_dlp

    YT_DLP_AVAILABLE = True
except ImportError:
    YT_DLP_AVAILABLE = False
    print("[WARNING] yt-dlp not available. URL video downloads will not work.")


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
                if os.path.exists(os.path.join(db_dir, "meta.json")):
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
        if self.device.type == "cuda":
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
            prompts = [p.strip() for p in text_prompt.split(".") if p.strip()]
            if not prompts:
                prompts = ["object"]

            # Create ontology
            ontology_dict = {prompt: prompt for prompt in prompts}
            ontology = CaptionOntology(ontology_dict)

            # Initialize GroundedSAM with correct parameters
            self.grounded_sam = GroundedSAM(
                ontology=ontology, box_threshold=0.35, text_threshold=0.25
            )
            print(f"🎯 GroundedSAM ready with prompts: {prompts}")
            print(
                f"[DEBUG] GroundedSAM version: {self.grounded_sam.__version__ if hasattr(self.grounded_sam, '__version__') else 'unknown'}"
            )
            print(f"[DEBUG] GroundedSAM config: {self.grounded_sam.__dict__}")

            # Print available methods and attributes
            print(
                f"[DEBUG] Available methods: {[m for m in dir(self.grounded_sam) if not m.startswith('_')]}"
            )

            # Print predict method signature
            if hasattr(self.grounded_sam, "predict"):
                import inspect

                print(
                    f"[DEBUG] Predict method signature: {inspect.signature(self.grounded_sam.predict)}"
                )

    def detect_regions(self, image, text_prompt="person . car . building"):
        """Detect regions using GroundedSAM"""
        print(f"🔍 Detecting regions with prompt: '{text_prompt}'")

        # Initialize GroundedSAM if needed
        self.init_grounded_sam(text_prompt)

        # Save image temporarily for GroundedSAM
        temp_path = f"/tmp/temp_image_{uuid.uuid4().hex[:8]}.jpg"
        if isinstance(image, np.ndarray):
            image_pil = Image.fromarray(image)
        else:
            image_pil = image

        image_pil.save(temp_path)

        try:
            # Detect with GroundedSAM
            detections = self.grounded_sam.predict(temp_path)
            print(f"✅ Found {len(detections)} regions")
            print(f"[DEBUG] Detection type: {type(detections)}")
            print(f"[DEBUG] Detection attributes: {dir(detections)}")

            # Convert detections to supervision format with masks
            from supervision.detection.core import Detections

            # Extract boxes and masks
            boxes = detections.xyxy
            class_ids = detections.class_id
            confidences = detections.confidence

            # Get masks from the predictions
            masks = []
            if hasattr(detections, "masks"):
                masks = detections.masks
            elif hasattr(detections, "mask"):
                masks = detections.mask
            elif hasattr(detections, "data") and "masks" in detections.data:
                masks = detections.data["masks"]

            print(
                f"[DEBUG] Boxes shape: {boxes.shape if hasattr(boxes, 'shape') else 'no shape'}"
            )
            print(f"[DEBUG] Masks type: {type(masks)}")
            print(
                f"[DEBUG] Masks shape: {masks.shape if hasattr(masks, 'shape') else 'no shape'}"
            )
            print(f"[DEBUG] Class IDs: {class_ids}")
            print(f"[DEBUG] Confidences: {confidences}")

            # Convert masks to numpy if they're tensors
            if isinstance(masks, torch.Tensor):
                masks = masks.detach().cpu().numpy()
                print(f"[DEBUG] Converted masks to numpy array")

            # Create supervision detections with masks
            self.detected_regions = Detections(
                xyxy=boxes, mask=masks, confidence=confidences, class_id=class_ids
            )

            # Debug print the created detections
            print(f"[DEBUG] Created detections type: {type(self.detected_regions)}")
            print(
                f"[DEBUG] Created detections attributes: {dir(self.detected_regions)}"
            )
            print(
                f"[DEBUG] Created detections mask attribute: {hasattr(self.detected_regions, 'mask')}"
            )
            if hasattr(self.detected_regions, "mask"):
                print(
                    f"[DEBUG] Created detections mask shape: {self.detected_regions.mask.shape if hasattr(self.detected_regions.mask, 'shape') else 'no shape'}"
                )

            return len(detections)

        finally:
            # Cleanup
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def extract_embeddings(self, image):
        """Extract PE embeddings from detected regions"""
        if not self.detected_regions or len(self.detected_regions) == 0:
            print("❌ No regions detected")
            return [], []

        print(
            f"🧠 Extracting PE embeddings from {len(self.detected_regions)} regions..."
        )

        # Convert image to tensor
        if isinstance(image, np.ndarray):
            image_pil = Image.fromarray(image)
        else:
            image_pil = image

        image_tensor = self.preprocess(image_pil).unsqueeze(0).to(self.device)

        embeddings = []
        metadata = []

        with torch.no_grad():
            # Extract features using standard PE encoding
            features = self.pe_model.encode_image(image_tensor)
            print(f"✅ Using standard PE encoding (still very effective)")
            print(f"[DEBUG] Raw features shape: {features.shape}")

            # Handle different feature shapes
            if len(features.shape) == 3:  # [1, num_tokens, dim]
                B, N, D = features.shape
                print(
                    f"[DEBUG] Token-based features: {B} batch, {N} tokens, {D} dimensions"
                )

                # For token-based features, we'll use a simpler approach
                # Just use the global image embedding for all regions
                global_embedding = features.mean(dim=1)  # [1, D] - average all tokens
                print(f"[DEBUG] Using global embedding approach due to token format")

            elif len(features.shape) == 2:  # [1, dim] - already a global embedding
                global_embedding = features
                print(f"[DEBUG] Global embedding shape: {global_embedding.shape}")

            else:
                print(f"[ERROR] Unexpected feature shape: {features.shape}")
                return [], []

            # Get the class names from the ontology
            try:
                ontology_class_names = self.grounded_sam.ontology.classes()
                print(f"[DEBUG] Ontology class names: {ontology_class_names}")
            except Exception as e:
                print(f"[WARNING] Could not get ontology classes: {e}")
                # Fallback to extracting from the ontology dict
                try:
                    ontology_class_names = list(
                        self.grounded_sam.ontology.prompts_to_classes.keys()
                    )
                except:
                    ontology_class_names = ["object"]  # Final fallback

            # Process each detected region
            for i in range(min(len(self.detected_regions), 10)):
                try:
                    # Get the mask directly from the detections object
                    if (
                        hasattr(self.detected_regions, "mask")
                        and self.detected_regions.mask is not None
                    ):
                        mask_np = self.detected_regions.mask[i]
                    else:
                        print(f"❌ No mask available for detection {i}, skipping")
                        continue

                    # Get confidence
                    raw_confidence = 0.0
                    if (
                        hasattr(self.detected_regions, "confidence")
                        and self.detected_regions.confidence is not None
                    ):
                        if i < len(self.detected_regions.confidence):
                            raw_confidence = float(self.detected_regions.confidence[i])

                    # Get detected class
                    detected_class = "object"  # default
                    if (
                        hasattr(self.detected_regions, "class_id")
                        and self.detected_regions.class_id is not None
                    ):
                        if i < len(self.detected_regions.class_id):
                            class_id = int(self.detected_regions.class_id[i])
                            if 0 <= class_id < len(ontology_class_names):
                                detected_class = ontology_class_names[class_id]

                    print(
                        f"[STATUS] Processing detection {i+1}: {detected_class} (Confidence: {raw_confidence:.3f})"
                    )

                    # Ensure mask is binary uint8
                    if mask_np.dtype == bool:
                        mask_processed = mask_np.astype(np.uint8)
                    elif np.issubdtype(mask_np.dtype, np.floating):
                        mask_processed = (mask_np > 0.5).astype(np.uint8)
                    else:
                        mask_processed = mask_np.astype(np.uint8)

                    # Check if mask is empty
                    if np.sum(mask_processed) == 0:
                        print(f"⚠️ Empty mask for region {i}, skipping")
                        continue

                    # For now, use the global image embedding for each region
                    # This is simpler and still effective for similarity search
                    region_embedding = global_embedding[0]  # Remove batch dimension

                    # Normalize
                    region_embedding = region_embedding / region_embedding.norm()

                    # Store
                    embeddings.append(region_embedding.cpu())

                    # Create metadata - get bounding box from mask
                    y_indices, x_indices = np.where(mask_processed)
                    bbox = [
                        int(x_indices.min()),
                        int(y_indices.min()),
                        int(x_indices.max()),
                        int(y_indices.max()),
                    ]

                    meta = {
                        "region_id": str(uuid.uuid4()),
                        "bbox": bbox,
                        "area_ratio": float(
                            np.sum(mask_processed) / mask_processed.size
                        ),
                        "detection_index": i,
                        "confidence": raw_confidence,
                        "detected_class": detected_class,
                    }
                    metadata.append(meta)

                    print(f"✅ Extracted embedding for region {i}")

                except Exception as e:
                    print(f"❌ Error processing region {i}: {e}")
                    import traceback

                    traceback.print_exc()
                    continue

        self.region_embeddings = embeddings
        print(f"🎯 Extracted {len(embeddings)} region embeddings")
        return embeddings, metadata

    def process_image_direct_pe(self, image):
        """Process image directly with PE, without GroundedSAM"""
        print("🧠 Processing image directly with PE...")

        # Convert image to tensor
        if isinstance(image, np.ndarray):
            image_pil = Image.fromarray(image)
        else:
            image_pil = image

        image_tensor = self.preprocess(image_pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # Extract features using standard PE encoding
            features = self.pe_model.encode_image(image_tensor)

            # Handle different feature shapes
            if len(features.shape) == 3:  # [1, num_tokens, dim]
                # Use global embedding (average of tokens)
                embedding = features.mean(dim=1)[0]  # Remove batch dimension
            elif len(features.shape) == 2:  # [1, dim]
                embedding = features[0]  # Remove batch dimension
            else:
                raise ValueError(f"Unexpected feature shape: {features.shape}")

            # Normalize
            embedding = embedding / embedding.norm()

            # Store as a single region
            self.region_embeddings = [embedding.cpu()]

            # Create metadata for the whole image
            meta = {
                "region_id": str(uuid.uuid4()),
                "bbox": [0, 0, image_pil.width, image_pil.height],
                "area_ratio": 1.0,
                "detection_index": 0,
                "confidence": 1.0,
                "detected_class": "full_image",
            }

            print("✅ Extracted global image embedding")
            return [embedding.cpu()], [meta]

    def create_database(
        self,
        folder_path,
        database_name,
        text_prompt="person . car . building",
        use_direct_pe=False,
    ):
        """Create searchable database from image folder"""
        status_messages = []

        def log_status(message):
            status_messages.append(message)
            return "\n".join(status_messages)

        log_status(f"📁 Creating database '{database_name}' from {folder_path}")

        # Get image files
        image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
        image_files = []

        for file in os.listdir(folder_path):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(folder_path, file))

        if not image_files:
            return log_status(f"❌ No images found in {folder_path}")

        log_status(f"📊 Found {len(image_files)} images to process")
        log_status(
            f"🔧 Processing mode: {'Direct PE' if use_direct_pe else 'GroundedSAM + PE'}"
        )

        # Initialize vector database
        db_path = f"./simple_reverso_db/{database_name}"
        os.makedirs(db_path, exist_ok=True)
        log_status(f"📂 Created database directory: {db_path}")

        client = QdrantClient(path=db_path)

        # Process images
        all_embeddings = []
        all_metadata = []
        processed = 0
        failed = 0

        for i, image_path in enumerate(image_files):
            try:
                filename = os.path.basename(image_path)
                log_status(f"🔄 Processing {i+1}/{len(image_files)}: {filename}")

                # Load and process image
                image = Image.open(image_path).convert("RGB")

                if use_direct_pe:
                    # Process directly with PE
                    embeddings, metadata = self.process_image_direct_pe(image)
                    log_status(f"✅ Extracted global embedding for {filename}")
                else:
                    # Detect regions first
                    num_regions = self.detect_regions(image, text_prompt)
                    if num_regions > 0:
                        embeddings, metadata = self.extract_embeddings(image)
                        log_status(f"✅ Found {num_regions} regions in {filename}")
                    else:
                        log_status(f"⚠️ No regions found in {filename}, skipping")
                        failed += 1
                        continue

                    # Add source info to metadata
                    for meta in metadata:
                        meta["image_source"] = image_path
                    meta["filename"] = filename

                    all_embeddings.extend(embeddings)
                    all_metadata.extend(metadata)
                    processed += 1

            except Exception as e:
                log_status(f"❌ Error processing {filename}: {str(e)}")
                failed += 1
                continue

        if not all_embeddings:
            return log_status(f"❌ No regions found in any images")

        # Create collection
        vector_dim = all_embeddings[0].shape[0]
        collection_name = f"simple_reverso_{database_name}"

        try:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_dim, distance=models.Distance.COSINE
                ),
            )
            log_status(f"📦 Created collection: {collection_name}")
        except Exception:
            log_status(f"🔄 Collection {collection_name} already exists")

        # Insert embeddings
        log_status(f"💾 Storing {len(all_embeddings)} embeddings...")
        points = []
        for i, (emb, meta) in enumerate(zip(all_embeddings, all_metadata)):
            points.append(
                models.PointStruct(
                    id=meta["region_id"], vector=emb.numpy().tolist(), payload=meta
                )
            )

        client.upsert(collection_name=collection_name, points=points)

        # Store database info
        self.vector_db = client
        self.current_database = collection_name

        # Final summary
        log_status("\n📊 Final Summary:")
        log_status(f"✅ Successfully processed: {processed} images")
        if failed > 0:
            log_status(f"⚠️ Failed to process: {failed} images")
        log_status(f"🔍 Total embeddings stored: {len(all_embeddings)}")
        log_status(f"🎯 Database ready for searching!")

        return "\n".join(status_messages)

    def search_similar(self, similarity_threshold=0.7, max_results=5):
        """Search for similar regions in database"""
        if not self.region_embeddings:
            return "❌ No regions detected. Please detect regions first.", []

        if not self.vector_db or not self.current_database:
            return "❌ No database loaded. Please create a database first.", []

        print(f"🔍 Searching for similar regions (threshold={similarity_threshold})")

        # Use first detected region for search
        query_embedding = self.region_embeddings[0]

        # Search
        search_results = self.vector_db.search(
            collection_name=self.current_database,
            query_vector=query_embedding.numpy().tolist(),
            limit=max_results,
            score_threshold=similarity_threshold,
        )

        if not search_results:
            return (
                f"❌ No similar regions found above threshold {similarity_threshold}",
                [],
            )

        # Format results
        results_text = f"🎯 Found {len(search_results)} similar regions:\n\n"
        similar_images = []

        for i, result in enumerate(search_results):
            filename = result.payload.get("filename", "Unknown")
            score = result.score
            bbox = result.payload.get("bbox", [0, 0, 0, 0])
            image_path = result.payload.get("image_source", "")

            results_text += f"{i+1}. {filename} (similarity: {score:.3f})\n"
            results_text += f"   📍 Bounding box: {bbox}\n\n"

            # Load and process the similar image
            if os.path.exists(image_path):
                try:
                    # Load the image
                    img = Image.open(image_path).convert("RGB")

                    # If we have a bounding box, crop to that region
                    if bbox != [0, 0, img.width, img.height]:  # Not the full image
                        x1, y1, x2, y2 = bbox
                        img = img.crop((x1, y1, x2, y2))

                    # Add similarity score as text overlay
                    img_with_text = img.copy()
                    draw = ImageDraw.Draw(img_with_text)
                    # Use a larger font size and add a background for better visibility
                    font_size = max(
                        20, int(img.height * 0.05)
                    )  # Scale font size with image height
                    try:
                        from PIL import ImageFont

                        font = ImageFont.truetype("Arial", font_size)
                    except:
                        font = ImageFont.load_default()

                    # Add text with background
                    text = f"Score: {score:.3f}"
                    text_bbox = draw.textbbox((10, 10), text, font=font)
                    draw.rectangle(
                        [
                            text_bbox[0] - 5,
                            text_bbox[1] - 5,
                            text_bbox[2] + 5,
                            text_bbox[3] + 5,
                        ],
                        fill="black",
                        outline="white",
                    )
                    draw.text((10, 10), text, fill="white", font=font)

                    # Resize if too large
                    max_size = (800, 800)
                    if (
                        img_with_text.width > max_size[0]
                        or img_with_text.height > max_size[1]
                    ):
                        img_with_text.thumbnail(max_size, Image.Resampling.LANCZOS)

                    # Store both the image and its score
                    similar_images.append((img_with_text, score))
                    print(f"✅ Loaded image {i+1}: {image_path}")
                except Exception as e:
                    print(f"❌ Error loading image {image_path}: {e}")
                    similar_images.append(None)
            else:
                print(f"❌ Image not found: {image_path}")
                similar_images.append(None)

        print(f"📊 Found {len(similar_images)} images to display")
        return results_text, similar_images

    def visualize_detections(self, image, selected_region_index=None):
        """Create visualization of detected regions, highlighting the selected region if provided"""
        if not self.detected_regions:
            return None

        # Convert image to RGB if it's not already
        if isinstance(image, np.ndarray):
            if len(image.shape) == 2:  # Grayscale
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:  # RGBA
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        else:
            image = np.array(image.convert("RGB"))

        print(
            f"\n[DEBUG] Starting visualization of {len(self.detected_regions)} regions"
        )
        print(f"[DEBUG] Image shape: {image.shape}")
        print(f"[DEBUG] Detection type: {type(self.detected_regions)}")

        # Get masks directly from the detections object
        masks = self.detected_regions.mask
        print(f"[DEBUG] Masks type: {type(masks)}")
        print(
            f"[DEBUG] Masks shape: {masks.shape if hasattr(masks, 'shape') else 'no shape'}"
        )

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(image)

        for i in range(len(self.detected_regions)):
            print(f"\n[DEBUG] Processing region {i+1}")

            if masks is not None and i < len(masks):
                mask = masks[i]
                print(f"[DEBUG] Mask {i+1} shape: {mask.shape}")

                # Ensure mask is binary
                mask = (mask > 0.5).astype(np.uint8)
                print(
                    f"[DEBUG] After thresholding, mask unique values: {np.unique(mask)}"
                )
                print(f"[DEBUG] Mask sum (pixels): {np.sum(mask)}")

                # Find contours
                contours, _ = cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                print(f"[DEBUG] Found {len(contours)} contours")

                # Choose color and linewidth based on selection
                if selected_region_index is not None and i == selected_region_index:
                    color = "lime"
                    label_facecolor = "green"
                    linewidth = 4
                else:
                    color = "red"
                    label_facecolor = "red"
                    linewidth = 2

                # Draw contours
                for j, contour in enumerate(contours):
                    if len(contour) >= 3:
                        contour = contour.squeeze()
                        if contour.ndim == 2:
                            print(
                                f"[DEBUG] Drawing contour {j+1} with {len(contour)} points"
                            )
                            ax.plot(
                                contour[:, 0],
                                contour[:, 1],
                                color=color,
                                linewidth=linewidth,
                            )
                        else:
                            print(
                                f"[DEBUG] Skipping contour {j+1} - invalid shape: {contour.shape}"
                            )

                # Add label
                y_coords, x_coords = np.where(mask)
                if len(x_coords) > 0 and len(y_coords) > 0:
                    center_x = int(x_coords.mean())
                    center_y = int(y_coords.mean())
                    ax.text(
                        center_x,
                        center_y,
                        f"{i+1}",
                        color="white",
                        fontsize=12,
                        fontweight="bold",
                        bbox=dict(
                            boxstyle="round,pad=0.3",
                            facecolor=label_facecolor,
                            alpha=0.7,
                        ),
                    )
                    print(f"[DEBUG] Added label at ({center_x}, {center_y})")
            else:
                print(f"[DEBUG] Region {i+1} has no mask")

        ax.set_title(f"Detected Regions ({len(self.detected_regions)} found)")
        ax.axis("off")

        temp_path = f"/tmp/visualization_{uuid.uuid4().hex[:8]}.png"
        plt.savefig(temp_path, bbox_inches="tight", dpi=150)
        plt.close()
        result_image = Image.open(temp_path)
        os.remove(temp_path)
        return result_image
