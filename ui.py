# UI Layer for Simple Revers-o
# This file contains the Gradio interface definition and its callback functions.

import gradio as gr
import os
# import sys # Appears unused
import traceback # For error handling in callbacks
from pathlib import Path # Used for loading README.md

# Import core system and instantiate it globally for UI functions to use
from core_system import SimpleReverso
reverso = SimpleReverso()

# Import video processing functions and flags needed by the UI
from video_processing import (
    extract_frames_with_progress,
    process_local_videos_with_progress,
    YT_DLP_AVAILABLE,
    VIDEO_PROCESSING_AVAILABLE
)

# Placeholder for UI helper functions and create_simple_interface
# These will be moved from main.py
# Example:
# def build_database_ui_callback(folder_path, db_name, db_prompt, use_direct_pe):
#     # ... uses global reverso instance ...
#     pass

# def create_simple_interface():
#     # ... defines the gr.Blocks() interface ...
#     pass

# Helper functions (callbacks for Gradio interface)
# These functions will use the global 'reverso' instance.

def detect_and_extract_ui(image, text_prompt, use_direct_pe=False):
    """Detect regions and extract embeddings - UI focused wrapper."""
    if image is None:
        return None, "❌ Please upload an image", gr.update(choices=[], value=None, visible=False), None

    try:
        if use_direct_pe:
            embeddings, metadata = reverso.process_image_direct_pe(image)
            result_text = (f"✅ Processed image directly with PE\n"
                           f"🧠 Extracted global image embedding\n"
                           f"🎯 Ready to search!")
            viz_image = image # Show original image
            return viz_image, result_text, gr.update(choices=[], value=None, visible=False), None # No regions to select
        else:
            num_regions = reverso.detect_regions(image, text_prompt)
            if num_regions == 0:
                return None, f"❌ No regions found with prompt: '{text_prompt}'", gr.update(choices=[], value=None, visible=False), None

            embeddings, metadata = reverso.extract_embeddings(image)
            viz_image = reverso.visualize_detections(image) # Visualize all detections

            region_options = []
            for i, meta_item in enumerate(metadata):
                confidence = meta_item.get('confidence', 0.0)
                detected_class = meta_item.get('detected_class', 'object')
                region_options.append(f"Region {i+1}: {detected_class} (Conf: {confidence:.2f})")

            result_text = (f"✅ Found {num_regions} regions\n"
                           f"🧠 Extracted {len(embeddings)} embeddings\n"
                           f"🎯 Select a region to search with (or uses first by default)")

            return viz_image, result_text, gr.update(choices=region_options, value=region_options[0] if region_options else None, visible=True), metadata

    except Exception as e:
        traceback.print_exc()
        return None, f"❌ Error in detection/extraction: {str(e)}", gr.update(choices=[], value=None, visible=False), None

def detect_and_extract_with_state_ui(image, text_prompt, use_direct_pe=False):
    """Wrapper for detect_and_extract_ui to manage image state for visualization updates."""
    viz, status, region_options_update, metadata = detect_and_extract_ui(image, text_prompt, use_direct_pe)
    # The 'image' itself is stored in detected_image_state by Gradio, this function just passes through results
    return viz, status, region_options_update, image # Pass image to update detected_image_state

def build_database_ui(folder_path, db_name, db_prompt, use_direct_pe, progress=gr.Progress(track_tqdm=True)):
    """Build searchable database - UI focused wrapper."""
    if not folder_path or not db_name:
        return "❌ Please provide folder path and database name"
    if not os.path.exists(folder_path):
        return f"❌ Folder not found: {folder_path}"

    status_updates = []
    def progress_callback_for_gradio(message):
        # This function will be called by SimpleReverso.create_database
        # It needs to update the Gradio progress bar and collect messages.
        # For now, let's just collect messages. Progress bar update in Gradio is tricky with external callbacks.
        # A simpler way is to let create_database return messages and update progress based on image count.
        status_updates.append(message)
        # We can't directly call progress.update here if it's not the main thread.
        # Instead, create_database in core_system.py was modified to have its own loop and log_status
        # and we can use that log_status to update the UI if we pass a Gradio progress object.
        # For now, the progress_callback in core_system.py is a placeholder.
        # The main progress will be handled by tqdm within create_database if possible,
        # or we update progress based on the number of images processed.

    progress(0, desc="Starting database creation...")
    try:
        # The progress_callback in SimpleReverso.create_database is basic for now.
        # We'll rely on the final output string primarily.
        # A more advanced setup would involve a queue or direct gr.Progress updates from the core.
        # For simplicity, the core function `create_database` logs to console.
        # We can simulate progress here or enhance core_system later.

        # Simulate progress for UI:
        # This is a placeholder. Actual progress should be driven by create_database.
        # For now, create_database returns a string of logs.
        def dummy_progress_for_gradio(status_str):
            # This is a simple way to update the textbox, not a real progress bar update from core
            # This will be the Textbox output from the create_database function itself.
            pass

        result = reverso.create_database(folder_path, db_name, db_prompt, use_direct_pe, progress_callback=dummy_progress_for_gradio)
        progress(1, desc="Database creation complete.")
        return result # This will be a string of logs.
    except Exception as e:
        traceback.print_exc()
        progress(1, desc="Error during database creation.")
        return f"❌ Error creating database: {str(e)}"

def search_database_ui(similarity_threshold, max_results, current_search_results_state, selected_region_info=None):
    """Search for similar regions - UI focused wrapper."""
    try:
        if selected_region_info and reverso.region_embeddings:
            # selected_region_info is like "Region 1: person (Conf: 0.85)"
            try:
                region_index = int(selected_region_info.split()[1].rstrip(':')) - 1
                if 0 <= region_index < len(reverso.region_embeddings):
                    # Set the query embedding in reverso object to the selected one
                    reverso.query_embedding_for_search = [reverso.region_embeddings[region_index]]
                    # Note: SimpleReverso needs a way to set which embedding to use for search,
                    # or search_similar should take an embedding as input.
                    # For now, let's assume search_similar uses the first one in region_embeddings,
                    # so we modify region_embeddings. This is a bit of a hack.
                    # A cleaner way: reverso.search_similar(embedding_to_search=reverso.region_embeddings[region_index], ...)
                    # For this refactor, we keep current SimpleReverso logic: it uses self.region_embeddings[0]
                    # So, we temporarily modify self.region_embeddings to only contain the selected one.
                    original_embeddings = reverso.region_embeddings # Store original
                    reverso.region_embeddings = [reverso.region_embeddings[region_index]]
                    result_text, similar_items_data = reverso.search_similar(similarity_threshold, max_results)
                    reverso.region_embeddings = original_embeddings # Restore original
                else:
                    return "⚠️ Invalid region selected. Please re-detect.", gr.update(choices=[], value=None), None, [], []
            except (ValueError, IndexError) as e:
                # Fallback to first region if parsing fails or index is bad
                result_text, similar_items_data = reverso.search_similar(similarity_threshold, max_results)
        else:
            # Default search using the first (or only) detected embedding
            result_text, similar_items_data = reverso.search_similar(similarity_threshold, max_results)

        image_options = [f"Image {i+1} (Score: {item['score']:.3f})" for i, item in enumerate(similar_items_data) if item["image"] is not None]

        thumbnail_gallery = []
        for item_data in similar_items_data:
            if item_data and item_data["image"] is not None:
                thumbnail = item_data["image"].copy()
                thumbnail.thumbnail((200, 200))
                thumbnail_gallery.append(thumbnail)

        main_display_image = similar_items_data[0]["image"] if similar_items_data and similar_items_data[0]["image"] else None

        return (
            result_text,
            gr.update(choices=image_options, value=image_options[0] if image_options else None),
            main_display_image,
            similar_items_data, # This state will hold the list of dicts
            thumbnail_gallery
        )
    except Exception as e:
        traceback.print_exc()
        return f"❌ Search error: {str(e)}", gr.update(choices=[], value=None), None, [], []

def update_similar_image_ui(selected_image_option, search_results_data_state):
    """Update the displayed similar image based on dropdown selection."""
    if not selected_image_option or not search_results_data_state:
        return None
    try:
        index = int(selected_image_option.split()[1]) - 1 # "Image 1" -> 0
        if 0 <= index < len(search_results_data_state):
            item_data = search_results_data_state[index]
            if item_data and item_data["image"] is not None:
                return item_data["image"]
        return None
    except Exception as e:
        traceback.print_exc()
        return f"❌ Error updating similar image display: {str(e)}"

def list_available_databases_ui():
    databases = reverso.list_databases()
    if not databases:
        return "No databases found"
    return "\n".join(f"- {db}" for db in databases)

def load_selected_database_ui(database_name):
    return reverso.load_database(database_name)

def delete_selected_database_ui(database_name):
    return reverso.delete_database(database_name)

def unlock_selected_database_ui(database_name):
    return reverso.unlock_database(database_name)

def update_region_visualization_ui(selected_region_info, current_detected_image):
    """Update the visualization to highlight the selected region."""
    if current_detected_image is None or selected_region_info is None:
        return current_detected_image # Return original image if no selection or image
    try:
        region_index = int(selected_region_info.split()[1].rstrip(':')) - 1
        return reverso.visualize_detections(current_detected_image, selected_region_index=region_index)
    except Exception as e:
        traceback.print_exc()
        return current_detected_image # Fallback to original image on error

def reload_database_list_ui():
    """Reload the list of available databases for the dropdown."""
    databases = reverso.list_databases()
    if not databases:
        return gr.update(choices=[], value=None), "No databases found. Refresh if you created one."
    return gr.update(choices=databases, value=databases[0] if databases else None), f"✅ Reloaded. Found {len(databases)} databases."


def create_simple_interface():
    """Create simplified Gradio interface using the global 'reverso' instance."""

    with gr.Blocks(title="Simple Revers-o: Visual Investigation Tool") as demo:
        similar_items_state = gr.State([]) # Stores list of dicts from search_similar
        detected_image_state = gr.State(None) # Stores the PIL image that has detections

        gr.Markdown("# 🔍 Simple Revers-o: Visual Investigation Tool")
        gr.Markdown("**GroundedSAM + PE-Core-L14-336 for visual similarity search**")

        with gr.Tabs():
            with gr.TabItem("🎬 Extract Video Frames"):
                gr.Markdown("## Extract Frames from Videos")
                gr.Markdown("Extract frames for database using scene detection or uniform sampling.")
                with gr.Tabs():
                    with gr.TabItem("🔗 From URLs"):
                        if YT_DLP_AVAILABLE:
                            video_urls_input = gr.TextArea(label="Video URLs", placeholder="Enter URLs (comma or newline separated)")
                            url_output_folder_input = gr.Textbox(label="Output Folder for Frames", placeholder="/path/to/save/frames_from_urls")
                            with gr.Row():
                                url_frames_per_scene_slider = gr.Slider(1, 10, value=2, step=1, label="Frames per Scene")
                                url_scene_threshold_slider = gr.Slider(10, 60, value=30, step=5, label="Scene Detection Threshold")
                            url_max_quality_dropdown = gr.Dropdown(["360p", "480p", "720p", "1080p", "best"], value="720p", label="Max Video Quality")
                            url_extract_button = gr.Button("🎬 Extract Frames from URLs", variant="primary")
                            url_extraction_status_markdown = gr.Markdown("URL processing status...")

                            url_extract_button.click(
                                lambda urls, folder, fps, thresh, qual: extract_frames_with_progress(urls, folder, fps, thresh, qual, progress=gr.Progress(track_tqdm=True)),
                                inputs=[video_urls_input, url_output_folder_input, url_frames_per_scene_slider, url_scene_threshold_slider, url_max_quality_dropdown],
                                outputs=[url_extraction_status_markdown]
                            )
                        else:
                            gr.Markdown("⚠️ **URL video downloading (yt-dlp) not available.** Install with: `pip install yt-dlp`")

                    with gr.TabItem("📁 From Local Files"):
                        local_input_folder_input = gr.Textbox(label="Video Folder Path", placeholder="/path/to/local/videos")
                        local_output_folder_input = gr.Textbox(label="Output Folder for Frames", placeholder="/path/to/save/frames_from_local")
                        with gr.Row():
                            local_frames_per_scene_slider = gr.Slider(1, 10, value=2, step=1, label="Frames per Scene")
                            local_scene_threshold_slider = gr.Slider(10, 60, value=30, step=5, label="Scene Detection Threshold")
                        local_extract_button = gr.Button("🎬 Extract Frames from Local Videos", variant="primary")
                        local_extraction_status_markdown = gr.Markdown("Local video processing status...")

                        local_extract_button.click(
                            lambda in_fold, out_fold, fps, thresh: process_local_videos_with_progress(in_fold, out_fold, fps, thresh, progress=gr.Progress(track_tqdm=True)),
                            inputs=[local_input_folder_input, local_output_folder_input, local_frames_per_scene_slider, local_scene_threshold_slider],
                            outputs=[local_extraction_status_markdown]
                        )

            with gr.TabItem("🗃️ Create Database"):
                gr.Markdown("## Build a searchable database from your images")
                db_folder_input = gr.Textbox(label="📁 Image Folder Path", placeholder="/path/to/images")
                db_name_input = gr.Textbox(label="🏷️ Database Name", placeholder="my_visual_db")
                db_prompt_input = gr.Textbox(label="🎯 Detection Prompts (optional, period-separated)", value="person . car . building")
                db_use_direct_pe_checkbox = gr.Checkbox(label="🔍 Use Direct PE (no object detection, faster)", value=False)
                build_db_button = gr.Button("🚀 Build Database", variant="primary")
                db_status_output = gr.Textbox(label="📊 Database Creation Status", lines=10, interactive=False)

                build_db_button.click(
                    build_database_ui, # Uses global reverso
                    inputs=[db_folder_input, db_name_input, db_prompt_input, db_use_direct_pe_checkbox],
                    outputs=[db_status_output]
                )

            with gr.TabItem("🔎 Search Similar"):
                gr.Markdown("## Search for similar regions in your database")
                search_input_image = gr.Image(label="Upload Image for Query", type="pil")
                search_text_prompt_input = gr.Textbox(label="Detection Prompt for Query Image", value="person . car . building")
                search_use_direct_pe_checkbox = gr.Checkbox(label="Use Direct PE for Query Image", value=False)
                detect_query_button = gr.Button("🔎 Detect Regions / Process Query")

                detected_query_viz_image = gr.Image(label="Detected Regions in Query", type="pil", interactive=False)
                query_region_selector_dropdown = gr.Dropdown(label="Select Query Region (uses first if none selected)", choices=[], visible=True, interactive=True)
                detect_query_status_markdown = gr.Markdown("Detection status...")

                detect_query_button.click(
                    detect_and_extract_with_state_ui, # uses global reverso
                    inputs=[search_input_image, search_text_prompt_input, search_use_direct_pe_checkbox],
                    outputs=[detected_query_viz_image, detect_query_status_markdown, query_region_selector_dropdown, detected_image_state]
                )

                query_region_selector_dropdown.change(
                    update_region_visualization_ui, # uses global reverso
                    inputs=[query_region_selector_dropdown, detected_image_state],
                    outputs=detected_query_viz_image
                )

                gr.Markdown("---")
                similarity_threshold_slider = gr.Slider(0.1, 1.0, value=0.7, step=0.05, label="🎚️ Similarity Threshold")
                max_results_dropdown = gr.Dropdown([3, 5, 10, 20, 50], value=5, label="📊 Max Results")
                search_db_button = gr.Button("🎯 Search Database", variant="primary")

                search_results_summary_textbox = gr.Textbox(label="🔍 Search Results Summary", lines=10, interactive=False)
                similar_image_selector_dropdown = gr.Dropdown(label="Select Result to View", choices=[], interactive=True)
                similar_image_display = gr.Image(label="Selected Similar Image Result", type="pil", interactive=False)
                results_thumbnail_gallery = gr.Gallery(label="Search Results Thumbnails", columns=5, height=400, interactive=False)

                search_db_button.click(
                    search_database_ui, # uses global reverso
                    inputs=[similarity_threshold_slider, max_results_dropdown, similar_items_state, query_region_selector_dropdown],
                    outputs=[search_results_summary_textbox, similar_image_selector_dropdown, similar_image_display, similar_items_state, results_thumbnail_gallery]
                )

                similar_image_selector_dropdown.change(
                    update_similar_image_ui,
                    inputs=[similar_image_selector_dropdown, similar_items_state], # Pass the state containing all result data
                    outputs=[similar_image_display]
                )

            with gr.TabItem("⚙️ Database Management"):
                gr.Markdown("## Manage your databases")
                initial_db_list = reverso.list_databases()
                db_selector_dropdown = gr.Dropdown(
                    label="Select Database",
                    choices=initial_db_list,
                    value=initial_db_list[0] if initial_db_list else None,
                    interactive=True
                )

                with gr.Row():
                    load_db_button = gr.Button("📂 Load Database")
                    delete_db_button = gr.Button("🗑️ Delete Database", variant="stop")
                    unlock_db_button = gr.Button("🔓 Unlock Database (if stuck)")
                    reload_db_list_button = gr.Button("🔄 Reload List")

                db_management_status_textbox = gr.Textbox(label="Status", lines=3, interactive=False)

                load_db_button.click(load_selected_database_ui, inputs=[db_selector_dropdown], outputs=[db_management_status_textbox])
                delete_db_button.click(delete_selected_database_ui, inputs=[db_selector_dropdown], outputs=[db_management_status_textbox])
                unlock_db_button.click(unlock_selected_database_ui, inputs=[db_selector_dropdown], outputs=[db_management_status_textbox])
                reload_db_list_button.click(reload_database_list_ui, outputs=[db_selector_dropdown, db_management_status_textbox])

            with gr.TabItem("ℹ️ About"):
                gr.Markdown(Path("README.md").read_text() if os.path.exists("README.md") else "Simple Revers-o: Visual Investigation Tool. See README.md for more details.")


    return demo
