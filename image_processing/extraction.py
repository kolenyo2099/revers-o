import os
import gc
import uuid
import tempfile
import numpy as np
import torch
import cv2
from PIL import Image

# Imports from current project structure
from core.models import pe_model as core_pe_model, pe_vit_model as core_pe_vit_model, preprocess as core_preprocess, device as core_device
from image_processing.utils import download_image, load_local_image, tokens_to_grid, get_detection_property
# Import PE utilities from their new location
from core.pe_utils import (
    get_preset_configuration,
    extract_multi_layer_features,
    combine_layer_features,
    apply_spatial_pooling_enhanced
    # Note: Other PE utils like specific pooling strategies (pe_attention_pooling, etc.)
    # are called by apply_spatial_pooling_enhanced, so direct import here is not essential
    # unless they were to be used directly in these extraction functions.
)
# We need to ensure GroundedSAM and CaptionOntology are available.
# These are typically installed packages, but their import was in main.py's function scope.
# For now, let's assume they are importable globally if autodistill_grounded_sam is in PYTHONPATH.
# If not, this might need adjustment based on how these are packaged/installed.
from autodistill_grounded_sam import GroundedSAM
from autodistill.detection import CaptionOntology


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
    # ENHANCED PARAMETERS (with backward-compatible defaults)
    extraction_mode="single",     # "single" or "multi_layer" or "preset"
    optimal_layer=30,            # Updated: Meta PE research optimal balance (was 40)
    preset_name="object_focused", # For preset mode
    custom_layers=[30,40,47],    # For multi_layer mode
    custom_weights=[0.3,0.4,0.3], # For multi_layer mode
    pooling_strategy="top_k",    # Existing parameter, now enhanced
    temperature=0.07             # New temperature parameter
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

    # Kept Helper Functions (or they should be in utils)
    # def apply_spatial_pooling(masked_features, strategy="top_k", top_k_ratio=0.1): # This local helper is no longer needed as apply_spatial_pooling_enhanced is directly used.
    #    """
    #    Apply different pooling strategies to masked features.
    #    Enhanced with PE-optimized strategies while maintaining backward compatibility.
    #    """
    #    return apply_spatial_pooling_enhanced(masked_features, strategy, top_k_ratio, temperature)

    def get_detected_class(detections_obj, det_index, class_names_from_ontology): # Local helper
        if hasattr(detections_obj, 'class_id') and \
           detections_obj.class_id is not None and \
           len(detections_obj.class_id) > det_index:

            class_id_val = detections_obj.class_id[det_index]

            if isinstance(class_id_val, torch.Tensor):
                class_id_val = class_id_val.item()

            if isinstance(class_id_val, (int, np.integer)) and 0 <= class_id_val < len(class_names_from_ontology):
                detected_class_name = class_names_from_ontology[class_id_val]
                print(f"[DEBUG get_class] Using class_id {class_id_val} (type {type(class_id_val)}) to get name '{detected_class_name}' from ontology classes list.")
                return detected_class_name
            else:
                print(f"[DEBUG get_class] class_id_val {class_id_val} (type {type(class_id_val)}) is not a valid index for ontology classes list (len {len(class_names_from_ontology)}).")
        else:
            print(f"[DEBUG get_class] detections_obj.class_id not found or invalid for index {det_index}.")

        default_class = class_names_from_ontology[0] if class_names_from_ontology else "unknown"
        print(f"[DEBUG get_class] Fallback: returning '{default_class}'")
        return default_class

    def cleanup_temp_resources(temp_file_paths_list=None): # Local helper
        if temp_file_paths_list is None:
            temp_file_paths_list = []
        for temp_file in temp_file_paths_list:
            if temp_file and os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                    print(f"[Cleanup] Removed temp file: {temp_file}")
                except Exception as e_file_remove:
                    print(f"[WARN Cleanup] Failed to remove temp file {temp_file}: {e_file_remove}")

        gc.collect()
        print(f"[Cleanup] Garbage collection called.")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"[Cleanup] Cleared CUDA cache.")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            try:
                torch.mps.empty_cache()
                print(f"[Cleanup] Cleared MPS cache.")
            except Exception as e_mps_cache:
                print(f"[WARN Cleanup] Error clearing MPS cache: {e_mps_cache}")

    pe_model_to_use = pe_model_param if pe_model_param is not None else core_pe_model
    pe_vit_model_to_use = pe_vit_model_param if pe_vit_model_param is not None else core_pe_vit_model
    preprocess_to_use = preprocess_param if preprocess_param is not None else core_preprocess
    device_to_use = device_param if device_param is not None else core_device

    detections_obj = None
    embeddings, metadata_list, labels = [], [], []
    temp_image_path = None

    try:
        print(f"[STATUS] Starting simplified region extraction process...")

        if is_url:
            pil_image = download_image(image_source) # from image_processing.utils
        else:
            if isinstance(image_source, str):
                print(f"[STATUS] Loading image from path: {os.path.basename(image_source)}")
                pil_image = load_local_image(image_source) # from image_processing.utils
            else:
                pil_image = image_source

        original_size = pil_image.size
        if max(pil_image.size) > max_image_size:
            pil_image.thumbnail((max_image_size, max_image_size), Image.Resampling.LANCZOS)
            print(f"[STATUS] Resized image from {original_size} to {pil_image.size}")

        image_np = np.array(pil_image)
        image_area = image_np.shape[0] * image_np.shape[1]

        temp_dir = tempfile.gettempdir()
        unique_id = str(uuid.uuid4())[:8]

        if isinstance(image_source, str) and os.path.isfile(image_source):
            base_filename = os.path.basename(image_source)
        else:
            base_filename = f"uploaded_image_{unique_id}.jpg"

        temp_image_path = os.path.join(temp_dir, f"gsam_temp_{unique_id}_{base_filename}")
        pil_image.save(temp_image_path, "JPEG", quality=95)
        print(f"[STATUS] Saved temporary image for GroundedSAM: {temp_image_path}")

        prompts = [p.strip() for p in text_prompt.split('.') if p.strip()]
        if not prompts:
            prompts = ["object", "region"]
            print(f"[WARN] No text prompt provided. Using default prompts: {prompts}")

        formatted_prompt = " . ".join(prompts) + " ."
        if len(prompts) == 1 and len(prompts[0].split()) > 2:
            formatted_prompt = prompts[0]

        ontology_dict = {prompt: prompt for prompt in prompts}
        ontology = CaptionOntology(ontology_dict)

        grounded_sam = GroundedSAM(
            ontology=ontology,
            box_threshold=0.35,
            text_threshold=0.25
        )
        detections = grounded_sam.predict(temp_image_path)
        ontology_class_names = ontology.classes()

        if len(detections) == 0:
            cleanup_temp_resources(temp_file_paths_list=[temp_image_path])
            return image_np if image_np is not None else None, [], [], [], [], None

        masks_cpu_np = []

        for i in range(min(len(detections), max_regions)):
            try:
                mask_tensor = get_detection_property(detections, i, 'mask') # Using helper from utils
                if mask_tensor is None: continue

                raw_confidence = get_detection_property(detections, i, 'confidence', 0.0)
                if isinstance(raw_confidence, torch.Tensor): raw_confidence = raw_confidence.item()
                elif isinstance(raw_confidence, np.ndarray): raw_confidence = float(raw_confidence)
                else: raw_confidence = float(raw_confidence)

                detected_class = get_detected_class(detections, i, ontology_class_names)
                print(f"[STATUS] Processing detection {i+1}/{min(len(detections), max_regions)}: {detected_class} (Raw Conf: {raw_confidence:.3f})")

                if isinstance(mask_tensor, torch.Tensor):
                    mask_np_cpu = mask_tensor.detach().cpu().numpy()
                elif isinstance(mask_tensor, np.ndarray):
                    mask_np_cpu = mask_tensor
                else:
                    mask_np_cpu = np.array(mask_tensor)

                if np.issubdtype(mask_np_cpu.dtype, np.floating):
                    mask_processed_cv = (mask_np_cpu > 0.5).astype(np.uint8)
                elif mask_np_cpu.dtype == bool:
                    mask_processed_cv = mask_np_cpu.astype(np.uint8)
                else:
                    mask_processed_cv = mask_np_cpu.astype(np.uint8)

                if np.sum(mask_processed_cv) == 0: continue

                current_area_pixels = np.sum(mask_processed_cv)
                current_area_ratio = current_area_pixels / image_area if image_area > 0 else 0

                if min_area_ratio > 0 and current_area_ratio < min_area_ratio: continue

                masks_cpu_np.append(mask_processed_cv)

                y_indices, x_indices = np.where(mask_processed_cv)
                x_min, x_max = int(x_indices.min()), int(x_indices.max())
                y_min, y_max = int(y_indices.min()), int(y_indices.max())

                metadata = {
                    "region_id": str(uuid.uuid4()),
                    "image_source": str(image_source) if isinstance(image_source, str) else "uploaded_image",
                    "filename": base_filename, "bbox": [x_min, y_min, x_max, y_max],
                    "area_ratio": float(current_area_ratio), "detected_class": detected_class,
                    "confidence": float(raw_confidence), "type": "region",
                    "detection_method": "grounded_sam_simplified", "layer_used": optimal_layer
                }
                label_text = f"{detected_class} ({raw_confidence:.2f})"
                metadata_list.append(metadata)
                labels.append(label_text)
            except Exception as e_det_loop:
                print(f"[ERROR] Error processing detection {i} in simplified loop: {e_det_loop}")
                continue

        if not masks_cpu_np:
            cleanup_temp_resources(temp_file_paths_list=[temp_image_path])
            return image_np if image_np is not None else None, [], [], [], [], None

        with torch.no_grad():
            pe_input = preprocess_to_use(pil_image).unsqueeze(0).to(device_to_use)
            intermediate_features = None
            try:
                if extraction_mode == "single":
                    if pe_vit_model_to_use is not None:
                        intermediate_features = pe_vit_model_to_use.forward_features(pe_input, layer_idx=max(1, optimal_layer))
                    else:
                        intermediate_features = pe_model_to_use.encode_image(pe_input)
                elif extraction_mode == "preset":
                    config = get_preset_configuration(preset_name)
                    if pe_vit_model_to_use is not None:
                        layer_features = extract_multi_layer_features(pe_input, pe_vit_model_to_use, config["layers"], device_to_use)
                        intermediate_features = combine_layer_features(layer_features, config["weights"], temperature)
                        pooling_strategy = config["pooling"]
                    else: intermediate_features = pe_model_to_use.encode_image(pe_input)
                elif extraction_mode == "multi_layer":
                    if pe_vit_model_to_use is not None:
                        layer_features = extract_multi_layer_features(pe_input, pe_vit_model_to_use, custom_layers, device_to_use)
                        intermediate_features = combine_layer_features(layer_features, custom_weights, temperature)
                    else: intermediate_features = pe_model_to_use.encode_image(pe_input)
                else: # Fallback
                    if pe_vit_model_to_use is not None: intermediate_features = pe_vit_model_to_use.forward_features(pe_input, layer_idx=max(1, optimal_layer))
                    else: intermediate_features = pe_model_to_use.encode_image(pe_input)
            except Exception as e_feat:
                cleanup_temp_resources(temp_file_paths_list=[temp_image_path])
                return None, [], [], [], [], f"Failed to extract PE features: {e_feat}"

            if pe_vit_model_to_use is None and len(intermediate_features.shape) == 2:
                intermediate_features = intermediate_features.unsqueeze(1).unsqueeze(1)

            if len(intermediate_features.shape) == 3:
                spatial_features = tokens_to_grid(intermediate_features, pe_model_to_use) # from image_processing.utils
            elif len(intermediate_features.shape) == 4:
                spatial_features = intermediate_features
            else: spatial_features = intermediate_features

            B, H, W, D_feat = spatial_features.shape
            spatial_features = spatial_features.to(device_to_use)
            pe_model_to_use = pe_model_to_use.to(device_to_use)

            temp_embeddings, temp_metadata, temp_labels, final_masks_for_output = [], [], [], []

            for i, mask_cv_np in enumerate(masks_cpu_np):
                try:
                    mask_resized_np = cv2.resize(mask_cv_np, (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
                    if np.sum(mask_resized_np) == 0: continue
                    masked_features = spatial_features[0, mask_resized_np, :]
                    if masked_features.shape[0] == 0: continue

                    region_embedding_pooled = apply_spatial_pooling_enhanced(masked_features, pooling_strategy, temperature=temperature)
                    region_embedding_pooled = region_embedding_pooled.to(device_to_use)
                    final_embedding = None

                    if pe_vit_model_to_use is not None:
                        if hasattr(pe_model_to_use, 'visual') and hasattr(pe_model_to_use.visual, 'ln_post') and hasattr(pe_model_to_use.visual, 'proj'):
                            if pe_model_to_use.visual.ln_post.weight.dtype == torch.float32 and region_embedding_pooled.dtype != torch.float32:
                                region_embedding_pooled = region_embedding_pooled.float()
                            normed_features = pe_model_to_use.visual.ln_post(region_embedding_pooled)
                            projected_embedding = normed_features @ pe_model_to_use.visual.proj
                            final_embedding = projected_embedding[0]
                        elif hasattr(pe_model_to_use, 'proj') and pe_model_to_use.proj is not None:
                            projected_embedding = region_embedding_pooled @ pe_model_to_use.proj
                            final_embedding = projected_embedding[0]
                        else: final_embedding = region_embedding_pooled[0]
                    else: final_embedding = region_embedding_pooled[0]

                    if final_embedding is not None and final_embedding.norm().item() > 1e-6 :
                        final_embedding = final_embedding / final_embedding.norm(dim=-1, keepdim=True)

                    if final_embedding is not None:
                        final_embedding_np_check = final_embedding.detach().cpu().numpy()
                        if np.isnan(final_embedding_np_check).any() or np.isinf(final_embedding_np_check).any(): continue
                        temp_embeddings.append(final_embedding.detach().cpu())
                        temp_metadata.append(metadata_list[i])
                        temp_labels.append(labels[i])
                        final_masks_for_output.append(mask_cv_np)
                except Exception as e_emb_loop:
                    print(f"[ERROR] Error generating embedding for mask {i}: {e_emb_loop}")
                    continue

            embeddings, metadata_list, labels = temp_embeddings, temp_metadata, temp_labels

            if not embeddings:
                cleanup_temp_resources(temp_file_paths_list=[temp_image_path])
                return image_np if image_np is not None else None, [], [], [], [], None

            cleanup_temp_resources(temp_file_paths_list=[temp_image_path])
            return image_np, final_masks_for_output, embeddings, metadata_list, labels, None
    except Exception as e:
        error_message = f"Critical error in simplified pipeline: {e}\\n{traceback.format_exc()}"
        cleanup_temp_resources(temp_file_paths_list=[temp_image_path])
        return None, [], [], [], [], error_message
    finally:
        print(f"[FINALLY] extract_region_embeddings_autodistill finished.")


def extract_whole_image_embeddings(
    image_source,
    pe_model_param=None,
    pe_vit_model_param=None,
    preprocess_param=None,
    device_param=None,
    max_image_size=800,
    extraction_mode="single",
    optimal_layer=30,
    preset_name="object_focused",
    custom_layers=[30,40,47],
    custom_weights=[0.3,0.4,0.3],
    pooling_strategy="top_k",
    temperature=0.07
):
    from contextlib import nullcontext # Import here if not globally available in this file
    from core.models import pe_model as core_pe_model, pe_vit_model as core_pe_vit_model, preprocess as core_preprocess, device as core_device

    pe_model_to_use = pe_model_param if pe_model_param is not None else core_pe_model
    pe_vit_model_to_use = pe_vit_model_param if pe_vit_model_param is not None else core_pe_vit_model
    preprocess_to_use = preprocess_param if preprocess_param is not None else core_preprocess
    device_to_use = device_param if device_param is not None else core_device

    try:
        if isinstance(image_source, str):
            if image_source.startswith(('http://', 'https://')):
                pil_image = download_image(image_source) # from image_processing.utils
            else:
                pil_image = load_local_image(image_source) # from image_processing.utils
        else:
            pil_image = image_source

        width, height = pil_image.size
        if width > max_image_size or height > max_image_size:
            new_width = max_image_size if width > height else int(width * (max_image_size / height))
            new_height = max_image_size if height >= width else int(height * (max_image_size / width))
            pil_image = pil_image.resize((new_width, new_height), Image.LANCZOS)

        image_np = np.array(pil_image)
        image_id = str(uuid.uuid4())
        metadata = {
            "region_id": image_id, "image_source": str(image_source) if isinstance(image_source, str) else "uploaded_image",
            "bbox": [0, 0, image_np.shape[1], image_np.shape[0]], "area": image_np.shape[0] * image_np.shape[1],
            "area_ratio": 1.0, "processing_type": "whole_image", "embedding_method": "pe_encoder",
            "layer_used": optimal_layer
        }

        with torch.no_grad():
            pe_input = preprocess_to_use(pil_image).unsqueeze(0).to(device_to_use)
            features, embedding_method = None, "unknown"
            amp_context = torch.amp.autocast(device_type='mps', dtype=torch.float16) if torch.backends.mps.is_available() and hasattr(torch.amp, 'autocast') else nullcontext()

            with amp_context:
                if extraction_mode == "single":
                    if pe_vit_model_to_use is not None:
                        try:
                            safe_layer = max(1, optimal_layer)
                            features = pe_vit_model_to_use.forward_features(pe_input, layer_idx=safe_layer)
                            embedding_method = f"vit_forward_features_layer{safe_layer}"
                        except Exception: features = None
                    if features is None : # Fallback or if pe_vit_model_to_use is None
                        features = pe_model_to_use.encode_image(pe_input)
                        embedding_method = "encode_image_fallback"
                elif extraction_mode == "preset":
                    config = get_preset_configuration(preset_name)
                    if pe_vit_model_to_use is not None:
                        try:
                            layer_features = extract_multi_layer_features(pe_input, pe_vit_model_to_use, config["layers"], device_to_use)
                            features = combine_layer_features(layer_features, config["weights"], temperature)
                            embedding_method = f"preset_{preset_name}_layers{config['layers']}"
                        except Exception: features = pe_vit_model_to_use.forward_features(pe_input, layer_idx=40); embedding_method = "preset_fallback_layer40" # Fallback
                    else: features = pe_model_to_use.encode_image(pe_input); embedding_method = "preset_fallback_encode_image"
                elif extraction_mode == "multi_layer":
                    if pe_vit_model_to_use is not None:
                        try:
                            layer_features = extract_multi_layer_features(pe_input, pe_vit_model_to_use, custom_layers, device_to_use)
                            features = combine_layer_features(layer_features, custom_weights, temperature)
                            embedding_method = f"multi_layer_layers{custom_layers}_weights{custom_weights}"
                        except Exception: features = pe_vit_model_to_use.forward_features(pe_input, layer_idx=40); embedding_method = "multi_layer_fallback_layer40" # Fallback
                    else: features = pe_model_to_use.encode_image(pe_input); embedding_method = "multi_layer_fallback_encode_image"
                else: # Fallback to original single layer logic if mode is unknown
                    if pe_vit_model_to_use is not None:
                        try: safe_layer = max(1, optimal_layer); features = pe_vit_model_to_use.forward_features(pe_input, layer_idx=safe_layer); embedding_method = f"fallback_vit_layer{safe_layer}"
                        except Exception: features = None
                    if features is None: features = pe_model_to_use.encode_image(pe_input); embedding_method = "encode_image_fallback"

            if len(features.shape) == 3:
                sequence_features = features.squeeze(0) if features.shape[0] == 1 else features[0]
                embedding = apply_spatial_pooling_enhanced(sequence_features, strategy=pooling_strategy, temperature=temperature)
            else: embedding = features

            embedding = torch.nn.functional.normalize(embedding, dim=-1)

        metadata["embedding_method"] = embedding_method
        actual_layer_used = optimal_layer
        if "layer" in embedding_method:
            try: actual_layer_used = int(embedding_method.split("layer")[-1])
            except (ValueError, IndexError): pass
        metadata["layer_used"] = actual_layer_used

        embedding_cpu = embedding.cpu()
        if torch.backends.mps.is_available(): torch.mps.synchronize()

        return image_np, embedding_cpu, metadata, "Whole Image"
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, None, None, None
