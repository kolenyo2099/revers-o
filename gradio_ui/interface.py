import gradio as gr
from PIL import Image
import numpy as np
import cv2 # For create_region_preview and search result visualization (if needed directly)
import matplotlib.pyplot as plt # For create_region_preview and search result visualization
from matplotlib import gridspec # For search result visualization
import io
import os
# import time # Not directly used in GradioInterface methods after refactor
# import uuid # Not directly used in GradioInterface methods after refactor
# import tempfile # Not directly used in GradioInterface methods after refactor
# import shutil # Not directly used in GradioInterface methods after refactor
# import urllib.parse # Not directly used in GradioInterface methods after refactor
# import hashlib # Not directly used in GradioInterface methods after refactor

# Qdrant client might be needed if search operations are not fully abstracted
from qdrant_client import QdrantClient

# Core module imports
from core.database import list_available_databases, delete_database_collection, cleanup_qdrant_connections
# AppState will be imported in tabs.py where create_multi_mode_interface is

# Image processing imports
from image_processing.extraction import extract_region_embeddings_autodistill, extract_whole_image_embeddings
from image_processing.visualization import visualize_detected_regions
from image_processing.utils import load_local_image # Used in search result visualization

# Video processing imports - only constants needed for UI conditional rendering
# from video_processing.extraction import VIDEO_PROCESSING_AVAILABLE # Will be used in tabs.py
# from video_processing.download import YT_DLP_AVAILABLE # Will be used in tabs.py


class GradioInterface:
    """Gradio interface for the Grounded SAM Region Search application"""

    def __init__(self, pe_model, pe_vit_model, preprocess, device):
        """Initialize the interface with required models"""
        self.pe_model = pe_model
        self.pe_vit_model = pe_vit_model
        self.preprocess = preprocess
        self.device = device

        self.detected_regions = {
            "image": None, "masks": [], "embeddings": [],
            "metadata": [], "labels": []
        }
        self.whole_image = {
            "image": None, "embedding": None, "metadata": None,
            "label": None, "layer_used": None
        }

        self.image_cache = {}
        self.search_result_images = []
        self.active_client = None

        self.available_collections = list_available_databases()
        if self.available_collections:
            self.active_collection = self.available_collections[0]
        else:
            self.active_collection = "grounded_image_regions" # Default
        print(f"[STATUS] GradioInterface initialized. Active collection: {self.active_collection}")

    def set_active_collection(self, collection_name):
        """Set the active collection for search operations"""
        if collection_name and collection_name != "No databases found":
            if collection_name == self.active_collection:
                print(f"[DEBUG] Collection already set to {collection_name}")
                return f"Using database: {collection_name}"
            previous = self.active_collection
            self.active_collection = collection_name
            print(f"[STATUS] Changed active collection from {previous} to {collection_name}")
            if self.active_client is not None:
                try: self.active_client.close()
                except Exception as e: print(f"Error closing previous Qdrant client: {e}")
                self.active_client = None
            return f"Set active database to: {collection_name}"
        return "No database selected or invalid selection."

    def process_image_with_prompt(self, image, text_prompt, min_area_ratio=0.01, max_regions=5,
                                extraction_mode="single", optimal_layer=40, preset_name="object_focused",
                                custom_layer_1=30, custom_layer_2=40, custom_layer_3=47,
                                custom_weight_1=0.3, custom_weight_2=0.4, custom_weight_3=0.3,
                                pooling_strategy="top_k", temperature=0.07):
        if image is None:
            return None, "No image provided.", gr.Dropdown(choices=[], value=None), None
        if isinstance(image, np.ndarray): image_pil = Image.fromarray(image)
        else: image_pil = image

        custom_layers = [custom_layer_1, custom_layer_2, custom_layer_3]
        custom_weights = [custom_weight_1, custom_weight_2, custom_weight_3]

        image_np, masks, embeddings, metadata, labels, error_message = extract_region_embeddings_autodistill(
            image_pil, text_prompt,
            pe_model_param=self.pe_model, pe_vit_model_param=self.pe_vit_model,
            preprocess_param=self.preprocess, device_param=self.device,
            min_area_ratio=min_area_ratio, max_regions=max_regions,
            optimal_layer=optimal_layer, extraction_mode=extraction_mode, preset_name=preset_name,
            custom_layers=custom_layers, custom_weights=custom_weights, temperature=temperature,
            pooling_strategy=pooling_strategy
        )

        if error_message:
            return None, error_message, gr.Dropdown(choices=[], value=None), None

        self.detected_regions = {"image": image_np, "masks": masks, "embeddings": embeddings, "metadata": metadata, "labels": labels}

        if not masks:
            return image_np, f"No regions found with prompt: '{text_prompt}'", gr.Dropdown(choices=[], value=None), None

        fig = visualize_detected_regions(image_np.copy(), masks, labels) # Pass a copy for safety
        choices = [f"Region {i+1}: {label}" for i, label in enumerate(labels)]
        buf = io.BytesIO(); fig.savefig(buf, format='png'); buf.seek(0); plt.close(fig)
        segmented_image = Image.open(buf)
        region_preview = self.create_region_preview(image_np, masks[0], labels[0]) if masks else None
        return segmented_image, f"Found {len(masks)} regions", gr.Dropdown(choices=choices, value=choices[0] if choices else None), region_preview

    def create_region_preview(self, image, mask, label=None):
        preview = image.copy()
        mask_3d = np.stack([mask, mask, mask], axis=2)
        highlighted = np.where(mask_3d, np.minimum(preview * 1.8, 255), preview * 0.3)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(highlighted, contours, -1, (0, 255, 0), 3)
        fig, ax = plt.subplots(figsize=(5,5)); ax.imshow(highlighted.astype(np.uint8))
        if label: ax.set_title(label, fontsize=12, pad=10)
        for spine in ax.spines.values(): spine.set_visible(True); spine.set_linewidth(3); spine.set_color('#01579b')
        ax.axis('on'); ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
        buf = io.BytesIO(); fig.savefig(buf, format='png'); buf.seek(0); plt.close(fig)
        return Image.open(buf)

    def _ensure_qdrant_client(self):
        if self.active_client is None:
            try:
                self.active_client = QdrantClient(path="./image_retrieval_project/qdrant_data")
                print(f"[STATUS] Qdrant client initialized.")
            except Exception as e:
                print(f"[ERROR] Failed to initialize Qdrant client: {e}")
                raise
        return self.active_client

    def search_region(self, region_selection, similarity_threshold=0.5, max_results=5):
        if region_selection is None or self.detected_regions["image"] is None or not self.detected_regions["embeddings"]:
            return None, "Please select a region or ensure regions are detected.", gr.update(visible=False), gr.update(choices=[], value=None)

        try:
            region_idx = int(region_selection.split(":")[0].replace("Region ", "")) - 1
            if not (0 <= region_idx < len(self.detected_regions["embeddings"])):
                return None, "Invalid region selection.", gr.update(visible=False), gr.update(choices=[], value=None)
            embedding = self.detected_regions["embeddings"][region_idx]

            client = self._ensure_qdrant_client()
            collections = client.get_collections().collections
            collection_names = [c.name for c in collections]
            if self.active_collection not in collection_names:
                return None, f"Error: Collection '{self.active_collection}' not found.", gr.update(visible=False), gr.update(choices=[], value=None)

            embedding_list = embedding[0].cpu().numpy().tolist() if len(embedding.shape) == 2 and embedding.shape[0] == 1 else embedding.cpu().numpy().flatten().tolist()

            search_results = client.search(
                collection_name=self.active_collection, query_vector=embedding_list,
                limit=int(max_results), score_threshold=float(similarity_threshold)
            )
            if not search_results:
                return None, f"No similar items found.", gr.update(visible=False), gr.update(choices=[], value=None)

            return self.create_unified_search_results_visualization(
                search_results, self.detected_regions["image"], self.detected_regions["masks"][region_idx],
                self.detected_regions["labels"][region_idx], "region"
            )
        except Exception as e:
            print(f"[ERROR] Search region failed: {e}")
            return None, f"Search error: {e}", gr.update(visible=False), gr.update(choices=[], value=None)

    def process_whole_image(self, image, extraction_mode="single", optimal_layer=40, preset_name="object_focused",
                            custom_layer_1=30, custom_layer_2=40, custom_layer_3=47,
                            custom_weight_1=0.3, custom_weight_2=0.4, custom_weight_3=0.3,
                            pooling_strategy="top_k", temperature=0.07):
        if image is None:
            return None, "No image provided."
        if isinstance(image, np.ndarray): image_pil = Image.fromarray(image)
        else: image_pil = image

        custom_layers = [custom_layer_1, custom_layer_2, custom_layer_3]
        custom_weights = [custom_weight_1, custom_weight_2, custom_weight_3]

        image_np, embedding, metadata, label = extract_whole_image_embeddings(
            image_pil, pe_model_param=self.pe_model, pe_vit_model_param=self.pe_vit_model,
            preprocess_param=self.preprocess, device_param=self.device,
            extraction_mode=extraction_mode, optimal_layer=optimal_layer, preset_name=preset_name,
            custom_layers=custom_layers, custom_weights=custom_weights, pooling_strategy=pooling_strategy, temperature=temperature
        )

        if embedding is None: return image_np, "Failed to process image."

        actual_layer = metadata.get("layer_used", optimal_layer) if metadata else optimal_layer
        self.whole_image = {"image": image_np, "embedding": embedding, "metadata": metadata, "label": label, "layer_used": actual_layer}

        fig, ax = plt.subplots(figsize=(8,8)); ax.imshow(image_np)
        ax.set_title(f"Whole Image Processed (Layer {actual_layer})")
        for spine_pos in ['top', 'bottom', 'left', 'right']: plt.gca().spines[spine_pos].set_linewidth(5); plt.gca().spines[spine_pos].set_color('green')
        ax.axis('on'); ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
        buf = io.BytesIO(); fig.savefig(buf, format='png'); buf.seek(0); plt.close(fig)
        return Image.open(buf), f"Image processed successfully using layer {actual_layer}."

    def search_whole_image(self, similarity_threshold=0.5, max_results=5):
        if self.whole_image["image"] is None or self.whole_image["embedding"] is None:
            return None, "Please process an image first.", gr.update(visible=False), gr.update(choices=[], value=None)

        embedding = self.whole_image["embedding"]
        try:
            client = self._ensure_qdrant_client()
            collections = client.get_collections().collections
            collection_names = [c.name for c in collections]
            if self.active_collection not in collection_names:
                return None, f"Error: Collection '{self.active_collection}' not found.", gr.update(visible=False), gr.update(choices=[], value=None)

            embedding_list = embedding[0].cpu().numpy().tolist() if len(embedding.shape) == 2 and embedding.shape[0] == 1 else embedding.cpu().numpy().flatten().tolist()

            search_results = client.search(
                collection_name=self.active_collection, query_vector=embedding_list,
                limit=int(max_results), score_threshold=float(similarity_threshold)
            )
            if not search_results:
                return None, f"No similar items found.", gr.update(visible=False), gr.update(choices=[], value=None)

            return self.create_unified_search_results_visualization(
                search_results, self.whole_image["image"], None, "Whole Image", "whole_image"
            )
        except Exception as e:
            print(f"[ERROR] Search whole image failed: {e}")
            return None, f"Search error: {e}", gr.update(visible=False), gr.update(choices=[], value=None)

    def create_unified_search_results_visualization(self, filtered_results, query_image, query_mask=None, query_label="Query", query_type="region"):
        self.search_result_images = []
        num_results = len(filtered_results)
        grid_rows, grid_cols = (1, num_results) if num_results <= 3 else (2, 3) if num_results <=6 else (3,4)
        fig_width, fig_height = min(18, 6 + 3 * grid_cols), 4 + 3 * grid_rows
        fig = plt.figure(figsize=(fig_width, fig_height))
        gs = gridspec.GridSpec(grid_rows + 1, grid_cols, height_ratios=[1] + [3] * grid_rows)
        ax_query = fig.add_subplot(gs[0, :])

        if query_type == "region" and query_mask is not None:
            highlighted = query_image.copy()
            mask_3d = np.stack([query_mask]*3, axis=2)
            highlighted = np.where(mask_3d, np.minimum(highlighted * 1.5, 255), highlighted * 0.4)
            contours, _ = cv2.findContours(query_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(highlighted, contours, -1, (0,255,0), 2)
            ax_query.imshow(highlighted.astype(np.uint8))
            ax_query.set_title(f"Query Region: {query_label}", fontsize=14, pad=10)
        else:
            ax_query.imshow(query_image); ax_query.set_title(f"Query: {query_label}", fontsize=14, pad=10)
        ax_query.axis('off')

        radio_choices, result_info_md = [], ["# Search Results\n\n"]
        for i, result in enumerate(filtered_results):
            if i >= grid_rows * grid_cols: break
            row, col = (i // grid_cols) + 1, i % grid_cols
            ax = fig.add_subplot(gs[row, col])
            try:
                metadata = result.payload
                image_source, score = metadata.get("image_source", "Unknown"), result.score
                collection_type = metadata.get("collection_type", "unknown")
                filename = os.path.basename(image_source) if isinstance(image_source, str) else "embedded_image"

                img = self.image_cache.get(image_source)
                if img is None and isinstance(image_source, str) and os.path.exists(image_source):
                    img = np.array(load_local_image(image_source)); self.image_cache[image_source] = img
                elif img is None: raise FileNotFoundError(f"Image source {image_source} not found or not in cache.")

                display_img, choice_label_suffix, result_type_info = None, "", ""
                badge_text, badge_fc, badge_ec = "", "", ""

                if collection_type == "region":
                    bbox = metadata.get("bbox", [0,0,100,100])
                    x_min, y_min, x_max, y_max = max(0, bbox[0]), max(0, bbox[1]), min(img.shape[1], bbox[2]), min(img.shape[0], bbox[3])
                    display_img = img[y_min:y_max, x_min:x_max]
                    detected_class = metadata.get("detected_class", "Object")
                    confidence = metadata.get("confidence", "")
                    confidence_str = f" ({confidence:.2f})" if isinstance(confidence, float) else ""
                    choice_label_suffix = f"- {detected_class}{confidence_str}"
                    result_type_info = f"Region ({detected_class})"
                    badge_text, badge_fc, badge_ec = detected_class.upper(), "#e1f5fe", "#01579b"
                    self.search_result_images.append({"image": display_img, "bbox": bbox, "metadata": {**metadata, "type": "region"}})
                else: # whole_image or unknown
                    display_img = img
                    choice_label_suffix = "(Whole Image)"
                    result_type_info = "Whole Image"
                    badge_text, badge_fc, badge_ec = "WHOLE IMAGE", "#e8f5e9", "#2e7d32"
                    self.search_result_images.append({"image": display_img, "metadata": {**metadata, "type": "whole_image"}})

                ax.imshow(display_img)
                for spine in ax.spines.values(): spine.set_visible(True); spine.set_linewidth(3); spine.set_color(badge_ec)
                ax.text(0.04, 0.04, badge_text, transform=ax.transAxes, fontsize=9, weight='bold', color=badge_ec,
                        verticalalignment='top', horizontalalignment='left', bbox=dict(boxstyle="round,pad=0.3", fc=badge_fc, ec=badge_ec, alpha=0.8))

                radio_choices.append(f"Result {i+1}: {filename} {choice_label_suffix}")
                result_info_md.append(f"**Result {i+1}:** {filename}\n- Type: {result_type_info}\n- Score: {score:.4f}\n")
                ax.set_title(f"Result {i+1}: {score:.2f}", fontsize=12, pad=5)
                ax.text(0.5, 0.03, filename, transform=ax.transAxes, ha='center', va='bottom', fontsize=8,
                        bbox=dict(facecolor='white', alpha=0.6, pad=2, edgecolor='lightgray', boxstyle='round'))
                ax.axis('off')
            except Exception as e:
                ax.text(0.5,0.5, f"Error: {e}", ha='center',va='center'); ax.axis('off')
                self.search_result_images.append(None); continue

        plt.tight_layout(pad=2.0); fig.subplots_adjust(wspace=0.3, hspace=0.3)
        buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=150); buf.seek(0); plt.close(fig)
        return Image.open(buf), "\n".join(result_info_md), gr.update(visible=True), gr.update(choices=radio_choices, value=None)

    def display_enlarged_result(self, result_selection):
        if result_selection is None or not self.search_result_images: return None
        try:
            result_idx = int(result_selection.split(":")[0].replace("Result ", "")) - 1
            if not (0 <= result_idx < len(self.search_result_images)) or self.search_result_images[result_idx] is None: return None

            result_data = self.search_result_images[result_idx]
            metadata = result_data.get("metadata", {})
            filename = metadata.get("filename", f"Result {result_idx+1}")
            detected_class = metadata.get("detected_class", "Unknown")
            score = metadata.get("score", 0.0)
            image_source = metadata.get("image_source")
            result_type = metadata.get("type", "region")

            fig = plt.figure(figsize=(12,10), constrained_layout=True); gs = gridspec.GridSpec(1,1,figure=fig); ax = fig.add_subplot(gs[0,0])
            display_image_np = None

            if image_source and os.path.exists(image_source):
                full_image_np = np.array(load_local_image(image_source)) # from image_processing.utils
                if result_type == "region" and "bbox" in result_data:
                    display_image_np = full_image_np.copy()
                    bbox = result_data["bbox"]
                    x_min, y_min, x_max, y_max = max(0,bbox[0]), max(0,bbox[1]), min(display_image_np.shape[1],bbox[2]), min(display_image_np.shape[0],bbox[3])
                    cv2.rectangle(display_image_np, (int(x_min),int(y_min)), (int(x_max),int(y_max)), (0,255,0),3)
                    # Optional: Slight highlight to ROI, be careful with color changes
                    # roi = display_image_np[y_min:y_max, x_min:x_max]; highlighted_roi = np.clip(roi * 1.2, 0, 255).astype(np.uint8)
                    # display_image_np[y_min:y_max, x_min:x_max] = cv2.addWeighted(highlighted_roi, 0.5, roi, 0.5, 0)
                else: display_image_np = full_image_np
            elif "image" in result_data and result_data["image"] is not None: # Fallback to stored region/image
                 display_image_np = np.array(result_data["image"])

            if display_image_np is not None: ax.imshow(display_image_np)
            else: ax.text(0.5,0.5,"Image not available",ha='center',va='center')

            ax.axis('off'); title = f"Result {result_idx+1}: {filename}"
            ax.set_title(title, fontsize=16, pad=15)
            info_text = f"Class: {detected_class}\nScore: {score:.4f}" if isinstance(score, (float,int)) else f"Class: {detected_class}\nScore: {score}"
            props = dict(boxstyle='round,pad=1', facecolor='white', alpha=0.8, edgecolor='lightgray')
            ax.text(0.5, 0.03, info_text, transform=ax.transAxes, ha='center', va='bottom', fontsize=12, bbox=props, linespacing=1.5)

            buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=200); buf.seek(0); plt.close(fig)
            return Image.open(buf)
        except Exception as e:
            print(f"[ERROR] Error displaying enlarged result: {e}"); import traceback; traceback.print_exc()
            return None

    def update_region_preview(self, region_selection):
        if region_selection is None or self.detected_regions["image"] is None or not self.detected_regions["masks"]:
            return None
        region_idx = int(region_selection.split(":")[0].replace("Region ", "")) - 1
        if not (0 <= region_idx < len(self.detected_regions["masks"])): return None
        mask = self.detected_regions["masks"][region_idx]
        label = self.detected_regions["labels"][region_idx]
        return self.create_region_preview(self.detected_regions["image"], mask, label)
