# UI Layer for Simple Revers-o
# This file contains the Gradio interface definition and its callback functions.

import gradio as gr
from gradio import Progress
import os
# import sys # Appears unused
import traceback # For error handling in callbacks
from pathlib import Path # Used for loading README.md
from PIL import ImageDraw, ImageFont
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
from datetime import datetime
import uuid

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
def detect_and_extract_ui(image, text_prompt, use_direct_pe=False):
    """Detect regions and extract embeddings - UI focused wrapper."""
    if image is None:
        return None, "❌ Please upload an image", gr.update(choices=[], value=None, visible=False), None

    try:
        # Clear previous embeddings before new detection
        reverso.region_embeddings = []
        
        if use_direct_pe:
            embeddings, metadata = reverso.process_image_direct_pe(image)
            reverso.region_embeddings = embeddings  # Store embeddings for search
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
            reverso.region_embeddings = embeddings  # Store embeddings for search
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

def build_database_ui(folder_path, db_name, db_prompt, use_direct_pe, resume_from_checkpoint, include_subfolders, progress=gr.Progress()):
    """Create searchable database from images - UI focused wrapper."""
    try:
        def progress_callback(message, progress_value=None):
            if progress_value is not None:
                progress(progress_value, desc=message)
            else:
                progress(0.0, desc=message)

        result = reverso.create_database(
            folder_path=folder_path,
            database_name=db_name,
            text_prompt=db_prompt,
            use_direct_pe=use_direct_pe,
            resume_from_checkpoint=resume_from_checkpoint,
            include_subfolders=include_subfolders,
            progress_callback=progress_callback
        )
        
        # Add finalization message if not already present
        if "Database" in result and "ready for searching" in result:
            if "✨ Finalization complete" not in result:
                result += "\n\n✨ Finalization complete - Database is ready to use!"
        
        return result
    except Exception as e:
        traceback.print_exc()
        return f"❌ Error: {str(e)}"

def stop_database_creation():
    """Stop the current database creation process."""
    try:
        reverso.request_stop()
        return "⏸️ Stop requested. Progress will be saved."
    except Exception as e:
        return f"❌ Error requesting stop: {str(e)}"

def search_database_ui(similarity_threshold, max_results, current_search_results_state, selected_region_info=None):
    """Search database for similar images using the current query image."""
    try:
        if not reverso.vector_db:
            return "⚠️ No database loaded. Please create or load a database first.", None, []

        if not reverso.region_embeddings:
            return "⚠️ No embeddings available. Please detect regions first.", [], []

        if selected_region_info and selected_region_info != "Region 1:":
            try:
                region_index = int(selected_region_info.split()[1].rstrip(':')) - 1
                if 0 <= region_index < len(reverso.region_embeddings):
                    # Create a temporary copy of the embeddings list
                    temp_embeddings = reverso.region_embeddings.copy()
                    # Use only the selected region's embedding
                    reverso.region_embeddings = [temp_embeddings[region_index]]
                    result_text, similar_items_data = reverso.search_similar(similarity_threshold, max_results)
                    # Restore original embeddings
                    reverso.region_embeddings = temp_embeddings
                else:
                    result_text = "⚠️ Invalid region index. Using first region."
                    result_text, similar_items_data = reverso.search_similar(similarity_threshold, max_results)
            except (ValueError, IndexError) as e:
                # Fallback to first region if parsing fails or index is bad
                result_text, similar_items_data = reverso.search_similar(similarity_threshold, max_results)
        else:
            # Default search using the first (or only) detected embedding
            result_text, similar_items_data = reverso.search_similar(similarity_threshold, max_results)

        # Format images for gallery display
        gallery_images = []
        for item in similar_items_data:
            if item["image"] is not None:
                # Add caption with score and filename
                caption = f"Score: {item['score']:.3f} | {item.get('filename', 'Unknown')}"
                gallery_images.append((item["image"], caption))

        return (
            result_text,
            gallery_images,
            similar_items_data
        )
    except Exception as e:
        traceback.print_exc()
        return f"❌ Search error: {str(e)}", [], []

def update_similar_image_ui(selected_image_option, search_results_data_state):
    """Update the displayed similar image based on dropdown selection."""
    if not selected_image_option or not search_results_data_state:
        return None
    try:
        index = int(selected_image_option.split()[1]) - 1 # "Image 1" -> 0
        if 0 <= index < len(search_results_data_state):
            item_data = search_results_data_state[index]
            if item_data and item_data["image"] is not None:
                # Create a copy of the image to add text
                img = item_data["image"].copy()
                # Add filename and score as text overlay
                draw = ImageDraw.Draw(img)
                try:
                    font = ImageFont.truetype("Arial", 12)  # Smaller font size
                except:
                    font = ImageFont.load_default()
                
                # Get filename and score
                filename = item_data.get("filename", "Unknown")
                score = item_data.get("score", 0)
                text = f"File: {filename}\nScore: {score:.3f}"
                
                # Add semi-transparent background for text
                text_bbox = draw.textbbox((0, 0), text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                draw.rectangle(
                    [(0, 0), (text_width + 10, text_height + 10)],  # Smaller padding
                    fill=(0, 0, 0, 100)  # More transparent background
                )
                
                # Add text
                draw.text((5, 5), text, fill=(255, 255, 255), font=font)  # Smaller padding
                return img
        return None
    except Exception as e:
        traceback.print_exc()
        return None

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

    with gr.Blocks(title="Revers-o: Visual Investigation Tool") as demo:
        similar_items_state = gr.State([]) # Stores list of dicts from search_similar
        detected_image_state = gr.State(None) # Stores the PIL image that has detections

        gr.Markdown("# 🔍 Revers-o: Visual Investigation Tool")
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
                                lambda urls, folder, fps, thresh, qual: extract_frames_with_progress(urls, folder, fps, thresh, qual, progress=None),
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
                db_prompt_input = gr.Textbox(
                    label="🎯 Detection Prompts (period-separated, e.g. 'car . building')",
                    placeholder="Enter prompts separated by periods",
                    value=""
                )
                with gr.Row():
                    db_use_direct_pe_checkbox = gr.Checkbox(label="🔍 Use Direct PE (no object detection, faster)", value=False)
                    db_resume_checkpoint_checkbox = gr.Checkbox(label="🔄 Resume from checkpoint", value=False)
                    db_include_subfolders_checkbox = gr.Checkbox(label="📂 Include subfolders", value=False)
                with gr.Row():
                    build_db_button = gr.Button("🚀 Build Database", variant="primary")
                    stop_db_button = gr.Button("⏸️ Stop Processing", variant="stop")
                db_status_output = gr.Markdown("Database creation status will appear here...")

                build_db_button.click(
                    build_database_ui, # Uses global reverso
                    inputs=[db_folder_input, db_name_input, db_prompt_input, db_use_direct_pe_checkbox, db_resume_checkpoint_checkbox, db_include_subfolders_checkbox],
                    outputs=[db_status_output]
                )

                stop_db_button.click(
                    stop_database_creation,
                    outputs=[db_status_output]
                )

            with gr.TabItem("🔎 Search Similar"):
                gr.Markdown("## Search for similar regions in your database")
                search_input_image = gr.Image(label="Upload Image for Query", type="pil")
                search_text_prompt_input = gr.Textbox(
                    label="Detection Prompt for Query Image",
                    placeholder="Enter prompt for detection (e.g. 'car . building')",
                    value=""
                )
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
                
                # Gallery for displaying similar images
                similar_images_gallery = gr.Gallery(
                    label="Similar Image Results",
                    show_label=True,
                    columns=[2, 3, 4, 5],
                    rows=[1, 2, 2, 3],
                    height="auto",
                    object_fit="contain",
                    allow_preview=True,
                    show_download_button=True
                )

                search_db_button.click(
                    search_database_ui, # uses global reverso
                    inputs=[similarity_threshold_slider, max_results_dropdown, similar_items_state, query_region_selector_dropdown],
                    outputs=[search_results_summary_textbox, similar_images_gallery, similar_items_state]
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
                gr.Markdown("""
                # Revers-o: Visual Investigation Tool for Journalists

                ## Key Capabilities:
                
                - **Track Objects Across Multiple Videos**: Follow persons, vehicles, or items through different footage sources
                - **Extract & Analyze Video Frames**: Automatically capture key frames from online or local videos
                - **Object-Based Search**: Find specific objects by appearance rather than metadata
                - **Cross-Source Analysis**: Connect visual evidence across multiple sources and timeframes
                
                ## Journalistic Applications:
                
                - Verify the presence of specific people or objects in multiple videos
                - Establish visual timelines by tracking objects across footage from different sources
                - Identify repeated visual patterns across large volumes of video evidence
                - Support investigative reporting with powerful visual correlation capabilities
                """)


    return demo
