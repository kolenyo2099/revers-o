import os
import time
import torch # For memory management if needed (torch.cuda.empty_cache())
import gradio as gr # For gr.update in progress functions, if they are still generators

# Core module imports
from core.database import setup_qdrant, store_embeddings_in_qdrant
from core.models import pe_model as core_pe_model, pe_vit_model as core_pe_vit_model, preprocess as core_preprocess, device as core_device
from core.checkpoint_utils import save_checkpoint, load_checkpoint, get_processed_files_from_database, should_skip_file
# PE Utils might be needed if any PE parameters are directly used or resolved here, though mostly in extraction.
# from core.pe_utils import ...

# Image processing imports
from image_processing.extraction import extract_region_embeddings_autodistill, extract_whole_image_embeddings

# Video processing (if any batch function directly calls video parts, though unlikely for these)
# from video_processing.extraction import ...

def collect_images_from_multiple_folders(folder_paths_string):
    """
    Collect all image paths from multiple comma-separated folder paths.
    Returns a tuple of (all_image_paths, folder_stats, error_messages)
    """
    folder_paths = [path.strip() for path in folder_paths_string.split(',') if path.strip()]
    if not folder_paths: return [], {}, ["No folder paths provided"]

    all_image_paths, folder_stats, error_messages = [], {}, []
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

    for folder_path in folder_paths:
        if not os.path.exists(folder_path):
            error_messages.append(f"❌ Folder not found: {folder_path}")
            continue
        if not os.path.isdir(folder_path):
            error_messages.append(f"❌ Path is not a directory: {folder_path}")
            continue

        try:
            folder_images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if any(f.lower().endswith(ext) for ext in image_extensions)]
            folder_stats[folder_path] = len(folder_images)
            all_image_paths.extend(folder_images)
        except Exception as e:
            error_messages.append(f"❌ Error reading folder {folder_path}: {str(e)}")

    all_image_paths.sort()
    return all_image_paths, folder_stats, error_messages

# --- Process Folder for Region Database (Advanced, with PE) ---
def process_folder_with_progress_advanced(folder_path, prompts, collection_name,
                                          optimal_layer=30, min_area_ratio=0.01, max_regions=5,
                                          resume_from_checkpoint=True,
                                          # PE Params
                                          extraction_mode="single", preset_name="object_focused",
                                          custom_layers=[30,40,47], custom_weights=[0.3,0.4,0.3],
                                          pooling_strategy="top_k", temperature=0.07,
                                          checkpoint_interval=10): # Added checkpoint_interval
    """Process folder with progress updates for the UI with advanced parameters including PE."""
    # This function is a generator, yielding messages for Gradio UI
    # print statements are for console logging

    # Access models from core.models - ensure they are loaded outside if this is run standalone
    # For Gradio, these are passed to GradioInterface and then to here if needed, or accessed globally
    pe_model, pe_vit_model, preprocess, device = core_pe_model, core_pe_vit_model, core_preprocess, core_device

    if not all([pe_model, preprocess, device]): # pe_vit_model can be None
        yield "❌ Critical Error: Models not loaded. Please restart the application.", gr.update(visible=False)
        return

    image_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))])
    total_files = len(image_files)
    if total_files == 0:
        yield "No images found in folder.", gr.update(visible=False)
        return

    processed_files_checkpoint = set()
    processed_files_database = set()
    if resume_from_checkpoint:
        processed_files_checkpoint = load_checkpoint(collection_name)
        # Client for DB check will be created on first storage attempt if not existing
        # temp_client = setup_qdrant(collection_name, 100, max_retries=1) # Vector size dummy for now
        # if temp_client:
        #    processed_files_database = get_processed_files_from_database(temp_client, collection_name)
        #    temp_client.close()
        # else:
        #    print("[WARNING] Could not connect to DB to get initially processed files for checkpointing.")
        # For simplicity now, relying more on file-based checkpoint for resume, DB check is secondary.


    client = None
    vector_dimension = None
    processed_count, skipped_count, error_count, total_regions_stored = 0,0,0,0

    yield f"🔍 Starting processing: {total_files} images. Collection: {collection_name}", gr.update(visible=False)

    for i, img_file in enumerate(image_files):
        img_path = os.path.join(folder_path, img_file)

        if should_skip_file(img_file, processed_files_checkpoint, processed_files_database):
            skipped_count +=1
            continue

        yield f"🔄 Processing: {i+1}/{total_files} - {img_file}", gr.update(visible=False)
        try:
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(): torch.mps.empty_cache()

            image_np, masks, embeddings, metadata, labels, error_msg = extract_region_embeddings_autodistill(
                img_path, prompts, pe_model, pe_vit_model, preprocess, device,
                min_area_ratio, max_regions, is_url=False, max_image_size=800, # Assuming max_image_size
                extraction_mode=extraction_mode, optimal_layer=optimal_layer, preset_name=preset_name,
                custom_layers=custom_layers, custom_weights=custom_weights,
                pooling_strategy=pooling_strategy, temperature=temperature
            )

            if error_msg:
                print(f"[ERROR] Processing {img_file}: {error_msg}")
                error_count += 1
                continue

            if image_np is not None and embeddings:
                if not client: # First successful embedding, setup Qdrant
                    first_embedding = embeddings[0]
                    if len(first_embedding.shape) == 1: vector_dimension = first_embedding.shape[0]
                    else: vector_dimension = first_embedding.shape[1] # Assuming (1,D) or (D)

                    client = setup_qdrant(collection_name, vector_dimension)
                    if not client:
                        yield "❌ Failed to initialize Qdrant client. Aborting.", gr.update(visible=False)
                        return
                    # After client setup, re-check DB for processed files if not done earlier due to no client
                    if resume_from_checkpoint and not processed_files_database: # only if not already populated
                         processed_files_database = get_processed_files_from_database(client, collection_name)
                         if should_skip_file(img_file, processed_files_checkpoint, processed_files_database): # Re-check after DB list acquired
                             skipped_count +=1
                             if client: client.close(); client=None # Close if opened just for this
                             continue


                # Add additional metadata
                for meta_item in metadata:
                    meta_item["source_folder"] = os.path.basename(folder_path)
                    meta_item["full_path"] = img_path
                    meta_item["embedding_layer"] = optimal_layer # Store which layer was used
                    meta_item["processing_timestamp"] = time.time()

                point_ids = store_embeddings_in_qdrant(client, collection_name, embeddings, metadata)
                if point_ids:
                    total_regions_stored += len(point_ids)
                    processed_count += 1
                    processed_files_checkpoint.add(img_file) # Add to checkpoint set
                else:
                    error_count +=1 # Failed to store
            else:
                skipped_count += 1

            if (i + 1) % checkpoint_interval == 0:
                if resume_from_checkpoint: save_checkpoint(collection_name, processed_files_checkpoint)

        except Exception as e:
            print(f"[ERROR] Unhandled exception processing {img_file}: {e}")
            import traceback; traceback.print_exc()
            error_count += 1

        if (i + 1) % 5 == 0 or (i + 1) == total_files : # Update UI periodically
             yield f"📊 Progress: {i+1}/{total_files}. Stored: {total_regions_stored} regions from {processed_count} images. Skipped: {skipped_count}. Errors: {error_count}.", gr.update(visible=False)

    if resume_from_checkpoint: save_checkpoint(collection_name, processed_files_checkpoint) # Final save
    if client: client.close()
    yield f"✅ Complete! Stored {total_regions_stored} regions from {processed_count} images. Skipped: {skipped_count}. Errors: {error_count}.", gr.update(visible=True)


# --- Process Multiple Folders for Region Database (Advanced, with PE) ---
def process_multiple_folders_with_progress_advanced(folder_paths_string, prompts, collection_name,
                                                    optimal_layer=30, min_area_ratio=0.01, max_regions=5,
                                                    resume_from_checkpoint=True,
                                                    extraction_mode="single", preset_name="object_focused",
                                                    custom_layers=[30,40,47], custom_weights=[0.3,0.4,0.3],
                                                    pooling_strategy="top_k", temperature=0.07,
                                                    checkpoint_interval=10):
    """Processes multiple folders with progress updates for the UI, using advanced parameters."""
    all_image_paths, folder_stats, error_messages = collect_images_from_multiple_folders(folder_paths_string)

    if error_messages: yield f"⚠️ Folder validation issues:\n" + "\n".join(error_messages), gr.update(visible=False)
    if not all_image_paths: yield "❌ No valid images found in any specified folders.", gr.update(visible=False); return

    folder_summary = "📁 Folder Summary:\n" + "\n".join([f"  - {fp}: {count} images" for fp, count in folder_stats.items()])
    yield folder_summary + f"\n\n📊 Total images to process: {len(all_image_paths)}", gr.update(visible=False)

    # Similar logic to process_folder_with_progress_advanced but iterates all_image_paths
    pe_model, pe_vit_model, preprocess, device = core_pe_model, core_pe_vit_model, core_preprocess, core_device
    if not all([pe_model, preprocess, device]):
        yield "❌ Critical Error: Models not loaded.", gr.update(visible=False); return

    processed_files_checkpoint = set()
    processed_files_database = set()
    if resume_from_checkpoint:
        processed_files_checkpoint = load_checkpoint(collection_name)
        # Initial DB check for processed files could be done here if desired, similar to single folder

    client = None
    vector_dimension = None
    processed_count, skipped_count, error_count, total_regions_stored = 0,0,0,0

    for i, img_path in enumerate(all_image_paths):
        img_file = os.path.basename(img_path)
        if should_skip_file(img_path, processed_files_checkpoint, processed_files_database): # Use full path for checkpointing across folders
            skipped_count +=1
            continue

        parent_folder_name = os.path.basename(os.path.dirname(img_path))
        yield f"🔄 Processing: {i+1}/{len(all_image_paths)} - {img_file} (from {parent_folder_name})", gr.update(visible=False)

        try:
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(): torch.mps.empty_cache()

            image_np, masks, embeddings, metadata, labels, error_msg = extract_region_embeddings_autodistill(
                img_path, prompts, pe_model, pe_vit_model, preprocess, device,
                min_area_ratio, max_regions, is_url=False, max_image_size=800,
                extraction_mode=extraction_mode, optimal_layer=optimal_layer, preset_name=preset_name,
                custom_layers=custom_layers, custom_weights=custom_weights,
                pooling_strategy=pooling_strategy, temperature=temperature
            )
            if error_msg: error_count += 1; print(f"Error on {img_file}: {error_msg}"); continue
            if image_np is not None and embeddings:
                if not client:
                    first_embedding = embeddings[0]
                    vector_dimension = first_embedding.shape[0] if len(first_embedding.shape) == 1 else first_embedding.shape[1]
                    client = setup_qdrant(collection_name, vector_dimension)
                    if not client: yield "❌ Qdrant client failed. Aborting.", gr.update(visible=False); return
                    if resume_from_checkpoint and not processed_files_database: # Fill after client is up
                        processed_files_database = get_processed_files_from_database(client, collection_name)
                        if should_skip_file(img_path, processed_files_checkpoint, processed_files_database): # Re-check
                             skipped_count +=1; continue

                for meta_item in metadata: # Add more detailed source info
                    meta_item["source_folder"] = parent_folder_name
                    meta_item["full_path"] = img_path
                    meta_item["embedding_layer"] = optimal_layer
                    meta_item["processing_timestamp"] = time.time()

                point_ids = store_embeddings_in_qdrant(client, collection_name, embeddings, metadata)
                if point_ids: total_regions_stored += len(point_ids); processed_count += 1; processed_files_checkpoint.add(img_path)
                else: error_count +=1
            else: skipped_count += 1
            if (i + 1) % checkpoint_interval == 0 and resume_from_checkpoint: save_checkpoint(collection_name, processed_files_checkpoint)
        except Exception as e: error_count += 1; print(f"Unhandled exception on {img_file}: {e}"); import traceback; traceback.print_exc()
        if (i + 1) % 5 == 0 or (i + 1) == len(all_image_paths): yield f"📊 Progress: {i+1}/{len(all_image_paths)}. Stored: {total_regions_stored} regions. Skipped: {skipped_count}. Errors: {error_count}.", gr.update(visible=False)

    if resume_from_checkpoint: save_checkpoint(collection_name, processed_files_checkpoint)
    if client: client.close()
    yield f"✅ All folders complete! Stored {total_regions_stored} regions from {processed_count} images. Skipped: {skipped_count}. Errors: {error_count}.", gr.update(visible=True)


# --- Process Folder for Whole Image Database (with PE) ---
def process_folder_with_progress_whole_images(folder_path, collection_name,
                                              optimal_layer=30, resume_from_checkpoint=True,
                                              extraction_mode="single", preset_name="object_focused",
                                              custom_layers=[30,40,47], custom_weights=[0.3,0.4,0.3],
                                              pooling_strategy="top_k", temperature=0.07,
                                              checkpoint_interval=10):
    """Process folder with whole image embeddings for the UI, with PE enhancements."""
    pe_model, pe_vit_model, preprocess, device = core_pe_model, core_pe_vit_model, core_preprocess, core_device
    if not all([pe_model, preprocess, device]):
        yield "❌ Critical Error: Models not loaded.", gr.update(visible=False); return

    image_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))])
    total_files = len(image_files)
    if total_files == 0: yield "No images found.", gr.update(visible=False); return

    processed_files_checkpoint = set()
    if resume_from_checkpoint: processed_files_checkpoint = load_checkpoint(collection_name)
    # DB check for processed files can be added here too if needed

    client, vector_dimension = None, None
    processed_count, error_count, skipped_count = 0,0,0
    yield f"🔍 Starting whole image processing: {total_files} images. Collection: {collection_name}", gr.update(visible=False)

    for i, img_file in enumerate(image_files):
        img_path = os.path.join(folder_path, img_file)
        if should_skip_file(img_file, processed_files_checkpoint, set()): # Not checking DB for whole images for now for simplicity
            skipped_count +=1; continue

        yield f"🔄 Processing: {i+1}/{total_files} - {img_file}", gr.update(visible=False)
        try:
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(): torch.mps.empty_cache()

            image_np, embedding, metadata, label = extract_whole_image_embeddings(
                img_path, pe_model, pe_vit_model, preprocess, device, max_image_size=800,
                extraction_mode=extraction_mode, optimal_layer=optimal_layer, preset_name=preset_name,
                custom_layers=custom_layers, custom_weights=custom_weights,
                pooling_strategy=pooling_strategy, temperature=temperature
            )
            if image_np is not None and embedding is not None:
                if not client:
                    vector_dimension = embedding.shape[0] if len(embedding.shape) == 1 else embedding.shape[1]
                    client = setup_qdrant(collection_name, vector_dimension)
                    if not client: yield "❌ Qdrant client failed. Aborting.", gr.update(visible=False); return

                metadata["source_folder"] = os.path.basename(folder_path)
                metadata["full_path"] = img_path
                metadata["processing_timestamp"] = time.time()

                store_embeddings_in_qdrant(client, collection_name, [embedding], [metadata])
                processed_count += 1; processed_files_checkpoint.add(img_file)
            else: error_count += 1; print(f"Failed to extract embedding for {img_file}")
            if (i + 1) % checkpoint_interval == 0 and resume_from_checkpoint: save_checkpoint(collection_name, processed_files_checkpoint)
        except Exception as e: error_count += 1; print(f"Unhandled exception on {img_file}: {e}"); import traceback; traceback.print_exc()
        if (i + 1) % 5 == 0 or (i + 1) == total_files: yield f"📊 Progress: {i+1}/{total_files}. Processed: {processed_count}. Skipped: {skipped_count}. Errors: {error_count}.", gr.update(visible=False)

    if resume_from_checkpoint: save_checkpoint(collection_name, processed_files_checkpoint)
    if client: client.close()
    yield f"✅ Whole image processing complete! Processed {processed_count} images. Skipped: {skipped_count}. Errors: {error_count}.", gr.update(visible=True)

# --- Process Multiple Folders for Whole Image Database (with PE) ---
def process_multiple_folders_with_progress_whole_images(folder_paths_string, collection_name,
                                                        optimal_layer=30, resume_from_checkpoint=True,
                                                        extraction_mode="single", preset_name="object_focused",
                                                        custom_layers=[30,40,47], custom_weights=[0.3,0.4,0.3],
                                                        pooling_strategy="top_k", temperature=0.07,
                                                        checkpoint_interval=10):
    """Processes multiple folders for whole image embeddings with progress updates and PE enhancements."""
    all_image_paths, folder_stats, error_messages = collect_images_from_multiple_folders(folder_paths_string)
    if error_messages: yield f"⚠️ Folder validation issues:\n" + "\n".join(error_messages), gr.update(visible=False)
    if not all_image_paths: yield "❌ No valid images found.", gr.update(visible=False); return

    folder_summary = "📁 Folder Summary:\n" + "\n".join([f"  - {fp}: {count} images" for fp, count in folder_stats.items()])
    yield folder_summary + f"\n\n📊 Total images for whole-image processing: {len(all_image_paths)}", gr.update(visible=False)

    pe_model, pe_vit_model, preprocess, device = core_pe_model, core_pe_vit_model, core_preprocess, core_device
    if not all([pe_model, preprocess, device]):
        yield "❌ Critical Error: Models not loaded.", gr.update(visible=False); return

    processed_files_checkpoint = set()
    if resume_from_checkpoint: processed_files_checkpoint = load_checkpoint(collection_name)

    client, vector_dimension = None, None
    processed_count, error_count, skipped_count = 0,0,0

    for i, img_path in enumerate(all_image_paths):
        img_file = os.path.basename(img_path) # For checkpointing with basename
        if should_skip_file(img_file, processed_files_checkpoint, set()):
             skipped_count +=1; continue

        parent_folder_name = os.path.basename(os.path.dirname(img_path))
        yield f"🔄 Processing whole image: {i+1}/{len(all_image_paths)} - {img_file} (from {parent_folder_name})", gr.update(visible=False)
        try:
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(): torch.mps.empty_cache()

            image_np, embedding, metadata, label = extract_whole_image_embeddings(
                img_path, pe_model, pe_vit_model, preprocess, device, max_image_size=800,
                extraction_mode=extraction_mode, optimal_layer=optimal_layer, preset_name=preset_name,
                custom_layers=custom_layers, custom_weights=custom_weights,
                pooling_strategy=pooling_strategy, temperature=temperature
            )
            if image_np is not None and embedding is not None:
                if not client:
                    vector_dimension = embedding.shape[0] if len(embedding.shape) == 1 else embedding.shape[1]
                    client = setup_qdrant(collection_name, vector_dimension)
                    if not client: yield "❌ Qdrant client failed. Aborting.", gr.update(visible=False); return

                metadata["source_folder"] = parent_folder_name
                metadata["full_path"] = img_path
                metadata["processing_timestamp"] = time.time()

                store_embeddings_in_qdrant(client, collection_name, [embedding], [metadata])
                processed_count += 1; processed_files_checkpoint.add(img_file)
            else: error_count += 1; print(f"Failed to extract whole image embedding for {img_file}")
            if (i + 1) % checkpoint_interval == 0 and resume_from_checkpoint: save_checkpoint(collection_name, processed_files_checkpoint)
        except Exception as e: error_count += 1; print(f"Unhandled exception on {img_file}: {e}"); import traceback; traceback.print_exc()
        if (i + 1) % 5 == 0 or (i + 1) == len(all_image_paths): yield f"📊 Whole Img Progress: {i+1}/{len(all_image_paths)}. Processed: {processed_count}. Skipped: {skipped_count}. Errors: {error_count}.", gr.update(visible=False)

    if resume_from_checkpoint: save_checkpoint(collection_name, processed_files_checkpoint)
    if client: client.close()
    yield f"✅ All folders (whole image) complete! Processed {processed_count} images. Skipped: {skipped_count}. Errors: {error_count}.", gr.update(visible=True)


# Placeholder for original process_folder_with_progress (simpler version)
# This might be removed or refactored to use the advanced one with default PE params
def process_folder_with_progress(*args, **kwargs):
    # This function is now effectively replaced by process_folder_with_progress_advanced
    # If it's still called, it should delegate or be updated.
    # For now, make it call the advanced version with default PE parameters.
    print("[INFO] process_folder_with_progress (simple) called, redirecting to advanced with defaults.")
    # Extract relevant args for advanced function, provide defaults for PE params
    folder_path, prompts, collection_name = args[0], args[1], args[2]
    # Call advanced function with default PE parameters
    yield from process_folder_with_progress_advanced(
        folder_path, prompts, collection_name,
        optimal_layer=30, min_area_ratio=0.01, max_regions=10, # Default vision params
        extraction_mode="single", preset_name="object_focused", # Default PE
        custom_layers=[30,40,47], custom_weights=[0.3,0.4,0.3], # Default PE
        pooling_strategy="top_k", temperature=0.07, # Default PE
        resume_from_checkpoint=True, checkpoint_interval=10
    )
