import gradio as gr
import os # For path operations in process_folder_with_mode
import time # For collection naming
from qdrant_client import QdrantClient # For GradioInterface active_client type hint if used

# New module imports
from gradio_ui.interface import GradioInterface
from core.app_state import AppState
from core.database import list_available_databases, delete_database_collection, cleanup_qdrant_connections, verify_repair_database

# Imports for batch processing from the new core module
from core.batch_processing import (
    process_folder_with_progress_advanced,
    process_multiple_folders_with_progress_advanced,
    process_folder_with_progress_whole_images,
    process_multiple_folders_with_progress_whole_images
)
# verify_repair_database and cleanup_qdrant_connections are already imported from core.database

from video_processing.extraction import process_video_folder, VIDEO_PROCESSING_AVAILABLE
from video_processing.download import process_video_urls, YT_DLP_AVAILABLE


def create_multi_mode_interface(pe_model, pe_vit_model, preprocess, device):
    """Create a multi-mode interface with both region-based and whole-image processing"""
    app_state = AppState()
    interface = GradioInterface(pe_model, pe_vit_model, preprocess, device)

    with gr.Blocks(title="Revers-o: GroundedSAM + Perception Encoders Image Similarity Search") as demo:
        gr.Markdown("# 🔍 Revers-o: GroundedSAM + Perception Encoders Image Similarity Search")

        with gr.Tabs() as tabs:
            with gr.TabItem("Quick Start"):
                gr.Markdown("## 🚀 Quick Start Guide")
                # ... (rest of Quick Start tab content - kept short for brevity)
                gr.Markdown("Follow instructions in other tabs.")

            with gr.TabItem("Extract Images"):
                gr.Markdown("## 🎬 Extract Images from Video")
                if VIDEO_PROCESSING_AVAILABLE:
                    with gr.Tabs():
                        with gr.TabItem("📁 From Local Files"):
                            video_input_folder = gr.Textbox(label="Video Folder Path")
                            video_output_folder = gr.Textbox(label="Output Folder Path")
                            max_frames_per_video = gr.Slider(minimum=5, maximum=10000, value=30, step=5, label="Max Frames per Video")
                            scene_threshold = gr.Slider(minimum=10.0, maximum=50.0, value=30.0, step=5.0, label="Scene Detection Sensitivity")
                            extract_video_button = gr.Button("🎬 Extract Images from Videos")
                            video_progress = gr.Textbox(label="Video Processing Status", lines=10)

                            def process_videos_with_progress_wrapper(input_folder, output_folder, max_frames, scene_thresh):
                                if not input_folder or not output_folder: yield "❌ Please specify folders"; return
                                if not os.path.exists(input_folder): yield f"❌ Input folder DNE: {input_folder}"; return
                                os.makedirs(output_folder, exist_ok=True)
                                yield f"📁 Created output: {output_folder}"
                                for msg in process_video_folder(input_folder, output_folder, max_frames, scene_thresh): # from video_processing.extraction
                                    yield msg
                            extract_video_button.click(process_videos_with_progress_wrapper,
                                                       inputs=[video_input_folder, video_output_folder, max_frames_per_video, scene_threshold],
                                                       outputs=[video_progress])

                        with gr.TabItem("🔗 From URLs"):
                            if YT_DLP_AVAILABLE:
                                video_urls_input = gr.Textbox(label="Video URLs", lines=5)
                                url_output_folder = gr.Textbox(label="Output Folder Path")
                                url_max_frames = gr.Slider(minimum=5, maximum=10000, value=30, step=5, label="Max Frames")
                                url_scene_threshold = gr.Slider(minimum=10.0, maximum=50.0, value=30.0, step=5.0, label="Scene Sensitivity")
                                video_quality = gr.Dropdown(choices=["480p", "720p", "1080p", "best"], value="720p", label="Video Quality")
                                extract_urls_button = gr.Button("🔗 Download & Extract Images from URLs")
                                url_progress = gr.Textbox(label="URL Processing Status", lines=10)

                                def process_urls_with_progress_wrapper(urls_text, out_folder, max_fr, scene_thr, qual):
                                    if not urls_text or not out_folder: yield "❌ Specify URLs and output folder"; return
                                    os.makedirs(out_folder, exist_ok=True)
                                    yield f"📁 Created output: {out_folder}"
                                    for msg in process_video_urls(urls_text, out_folder, max_fr, scene_thr, qual): # from video_processing.download
                                        yield msg
                                extract_urls_button.click(process_urls_with_progress_wrapper,
                                                          inputs=[video_urls_input, url_output_folder, url_max_frames, url_scene_threshold, video_quality],
                                                          outputs=[url_progress])
                            else: gr.Markdown("⚠️ **URL video downloading not available.** Install yt-dlp.")
                else: gr.Markdown("⚠️ **Video processing not available.** Install scenedetect, imageio.")

            with gr.TabItem("Create Database"):
                gr.Markdown("## Create a New Database")
                db_folder_path = gr.Textbox(label="Image Folder Path(s)")
                db_collection_name = gr.Textbox(label="Database Name", value="my_image_database")
                db_mode = gr.Radio(choices=["Region Detection (GroundedSAM + PE)", "Whole Image (PE Only)"], value="Region Detection (GroundedSAM + PE)", label="Processing Mode")

                with gr.Group(visible=True) as region_options_group_db: # Renamed to avoid conflict
                    region_prompts_db = gr.Textbox(label="Detection Prompts", value="person . car . building .") # Renamed
                    min_area_ratio_db = gr.Slider(minimum=0.001, maximum=0.1, value=0.01, step=0.001, label="Min Area Ratio") # Renamed
                    max_regions_db = gr.Slider(minimum=1, maximum=20, value=5, step=1, label="Max Regions") # Renamed

                with gr.Group(): # PE Enhancement controls
                    gr.Markdown("### 🔬 **Perception Encoder Enhancements**")
                    db_extraction_mode = gr.Radio(choices=["single", "preset", "multi_layer"], value="single", label="Extraction Mode")
                    with gr.Group(visible=True) as db_single_options:
                        db_embedding_layer = gr.Slider(minimum=1, maximum=50, value=40, step=1, label="Embedding Layer")
                    with gr.Group(visible=False) as db_preset_options:
                        db_preset_name = gr.Dropdown(choices=["object_focused", "spatial_focused", "semantic_focused", "texture_focused"], value="object_focused", label="PE Preset")
                    with gr.Group(visible=False) as db_custom_options:
                        db_custom_layer_1 = gr.Slider(20,50,value=30,step=1,label="L1"); db_custom_weight_1 = gr.Slider(0,1,value=0.3,step=0.1,label="W1")
                        db_custom_layer_2 = gr.Slider(20,50,value=40,step=1,label="L2"); db_custom_weight_2 = gr.Slider(0,1,value=0.4,step=0.1,label="W2")
                        db_custom_layer_3 = gr.Slider(20,50,value=47,step=1,label="L3"); db_custom_weight_3 = gr.Slider(0,1,value=0.3,step=0.1,label="W3")
                    db_pooling_strategy = gr.Dropdown(choices=["top_k", "pe_attention", "pe_spatial", "pe_semantic", "pe_adaptive"], value="top_k", label="Pooling Strategy")
                    db_temperature = gr.Slider(minimum=0.01, maximum=0.2, value=0.07, step=0.01, label="Temperature")

                create_db_button = gr.Button("Process Folder & Create Database")
                db_progress = gr.Textbox(label="Processing Status", lines=10)
                db_done_msg = gr.Markdown(visible=False)

                # Logic for toggling visibility of PE options based on db_extraction_mode
                def toggle_pe_options_db_ui(mode): # Renamed to avoid conflict
                    return gr.update(visible=mode=="single"), gr.update(visible=mode=="preset"), gr.update(visible=mode=="multi_layer")
                db_extraction_mode.change(toggle_pe_options_db_ui, inputs=[db_extraction_mode], outputs=[db_single_options, db_preset_options, db_custom_options])

                # Logic for toggling region options based on db_mode
                def toggle_region_options_ui(mode): # Renamed
                    return gr.update(visible=mode == "Region Detection (GroundedSAM + PE)")
                db_mode.change(toggle_region_options_ui, inputs=[db_mode], outputs=[region_options_group_db])

                def resolve_and_create_database_wrapper(folder_path, collection_name, prompts, mode, min_area, max_reg,
                                                     extraction_m, preset_n, cust_l1, cust_l2, cust_l3,
                                                     cust_w1, cust_w2, cust_w3, pool_strat, temp, embed_layer):
                    layer = embed_layer if extraction_m == "single" else 40 # Simplified logic for layer based on mode

                    # Determine which batch processing function to call
                    is_multiple_folders = ',' in folder_path
                    if mode == "Region Detection (GroundedSAM + PE)":
                        collection_name_final = f"{collection_name}_layer{layer}"
                        process_func = process_multiple_folders_with_progress_advanced if is_multiple_folders else process_folder_with_progress_advanced
                        gen = process_func(folder_path, prompts, collection_name_final, optimal_layer=layer, min_area_ratio=min_area, max_regions=max_reg,
                                           extraction_mode=extraction_m, preset_name=preset_n, custom_layers=[cust_l1,cust_l2,cust_l3],
                                           custom_weights=[cust_w1,cust_w2,cust_w3], pooling_strategy=pool_strat, temperature=temp)
                    else: # Whole Image
                        collection_name_final = f"{collection_name}_whole_images_layer{layer}"
                        process_func = process_multiple_folders_with_progress_whole_images if is_multiple_folders else process_folder_with_progress_whole_images
                        # Note: process_folder_with_progress_whole_images and its multiple folder counterpart need to be updated to accept all PE params
                        gen = process_func(folder_path, collection_name_final, optimal_layer=layer, extraction_mode=extraction_m, preset_name=preset_n,
                                           custom_layers=[cust_l1,cust_l2,cust_l3], custom_weights=[cust_w1,cust_w2,cust_w3],
                                           pooling_strategy=pool_strat, temperature=temp)

                    for message_val, done_visible_val in gen: # Renamed to avoid conflict
                        yield message_val, gr.update(visible=done_visible_val)

                    # Refresh DB list for verify/repair dropdown
                    # This part will be tricky as verify_db_dropdown is defined later
                    # For now, this is a placeholder for the update logic
                    # yield gr.update(), gr.update(choices=app_state.update_available_databases(verbose=True))

                create_db_button.click(resolve_and_create_database_wrapper,
                                       inputs=[db_folder_path, db_collection_name, region_prompts_db, db_mode, min_area_ratio_db, max_regions_db,
                                               db_extraction_mode, db_preset_name, db_custom_layer_1, db_custom_layer_2, db_custom_layer_3,
                                               db_custom_weight_1, db_custom_weight_2, db_custom_weight_3, db_pooling_strategy, db_temperature, db_embedding_layer],
                                       outputs=[db_progress, db_done_msg])

                gr.Markdown("## Database Maintenance")
                verify_db_dropdown = gr.Dropdown(choices=app_state.available_databases if app_state.available_databases else ["No databases found"], label="Select Database to Verify/Repair")
                force_rebuild_checkbox = gr.Checkbox(value=False, label="Force Complete Rebuild")
                verify_db_button = gr.Button("Verify Database")
                repair_db_button = gr.Button("Repair Database")
                verify_repair_status = gr.Textbox(label="Verification/Repair Status")

                verify_db_button.click(verify_repair_database, inputs=[verify_db_dropdown], outputs=[verify_repair_status], fn_name="verify_database_wrapper")
                repair_db_button.click(verify_repair_database, inputs=[verify_db_dropdown, force_rebuild_checkbox], outputs=[verify_repair_status], fn_name="repair_database_wrapper")


            with gr.TabItem("Search Database"):
                gr.Markdown("## Search Database")
                refresh_db_button = gr.Button("Refresh Database List")
                db_dropdown = gr.Dropdown(choices=app_state.available_databases if app_state.available_databases else ["No databases found"], label="Select Database")
                delete_db_button = gr.Button("Delete Selected Database", variant="stop")
                reset_db_button = gr.Button("Reset Database Connection", variant="secondary")

                processing_mode = gr.Radio(choices=["Region Detection (GroundedSAM + PE)", "Whole Image (PE Only)"], value="Region Detection (GroundedSAM + PE)", label="Processing Mode")
                input_image = gr.Image(type="pil", label="Upload Image")

                with gr.Group(visible=True) as region_mode_group:
                    text_prompt_search = gr.Textbox(label="Detection Prompts", value="person . car . building .") # Renamed
                    min_area_ratio_search = gr.Slider(minimum=0.001, maximum=0.1, value=0.01, step=0.001, label="Min Area Ratio") # Renamed
                    max_regions_search = gr.Slider(minimum=1, maximum=20, value=5, step=1, label="Max Regions") # Renamed
                    process_button = gr.Button("Detect Regions")

                with gr.Group(visible=False) as whole_image_mode_group:
                    process_whole_button = gr.Button("Process Whole Image")

                with gr.Group(): # PE Enhancement controls for search
                    gr.Markdown("### 🔬 **Perception Encoder Enhancements (Search)**")
                    search_extraction_mode = gr.Radio(choices=["single", "preset", "multi_layer"], value="single", label="Search Extraction Mode")
                    with gr.Group(visible=True) as search_single_options:
                        search_embedding_layer = gr.Slider(minimum=1, maximum=50, value=40, step=1, label="Search Embedding Layer")
                    with gr.Group(visible=False) as search_preset_options:
                        search_preset_name = gr.Dropdown(choices=["object_focused", "spatial_focused", "semantic_focused", "texture_focused"], value="object_focused", label="Search PE Preset")
                    with gr.Group(visible=False) as search_custom_options:
                        search_custom_layer_1 = gr.Slider(20,50,value=30,step=1,label="S_L1"); search_custom_weight_1 = gr.Slider(0,1,value=0.3,step=0.1,label="S_W1")
                        search_custom_layer_2 = gr.Slider(20,50,value=40,step=1,label="S_L2"); search_custom_weight_2 = gr.Slider(0,1,value=0.4,step=0.1,label="S_W2")
                        search_custom_layer_3 = gr.Slider(20,50,value=47,step=1,label="S_L3"); search_custom_weight_3 = gr.Slider(0,1,value=0.3,step=0.1,label="S_W3")
                    search_pooling_strategy = gr.Dropdown(choices=["top_k", "pe_attention", "pe_spatial", "pe_semantic", "pe_adaptive"], value="top_k", label="Search Pooling Strategy")
                    search_temperature = gr.Slider(minimum=0.01, maximum=0.2, value=0.07, step=0.01, label="Search Temperature")

                detection_info = gr.Textbox(label="Detection Status") # Common for both modes now

                with gr.Group(visible=True) as region_results_group:
                    segmented_output = gr.Image(type="pil", label="Detected Regions")
                    region_dropdown = gr.Dropdown(choices=[], label="Select a Region")
                    region_preview = gr.Image(type="pil", label="Region Preview")
                    similarity_slider = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.01, label="Similarity Threshold")
                    max_results_dropdown = gr.Dropdown(choices=["5","10","20","50"], value="5", label="Max Results")
                    search_button = gr.Button("Search Similar Regions")

                with gr.Group(visible=False) as whole_image_results_group:
                    processed_output = gr.Image(type="pil", label="Processed Image") # Display for whole image processing
                    whole_similarity_slider = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.01, label="Similarity Threshold")
                    whole_max_results_dropdown = gr.Dropdown(choices=["5","10","20","50"], value="5", label="Max Results")
                    whole_search_button = gr.Button("Search Similar Images")

                search_info = gr.Textbox(label="Search Status")
                search_results_output = gr.Image(type="pil", label="Search Results")
                result_selector = gr.Dropdown(choices=[], label="Select Result to View", visible=False) # Initially hidden
                enlarged_result = gr.Image(type="pil", label="Enlarged Result")

                # UI Toggling logic
                def toggle_search_mode_ui(mode): # Renamed
                    is_region_mode = mode == "Region Detection (GroundedSAM + PE)"
                    return gr.update(visible=is_region_mode), gr.update(visible=not is_region_mode), \
                           gr.update(visible=is_region_mode), gr.update(visible=not is_region_mode)
                processing_mode.change(toggle_search_mode_ui, inputs=[processing_mode],
                                       outputs=[region_mode_group, whole_image_mode_group, region_results_group, whole_image_results_group])

                def toggle_pe_options_search_ui(mode): # Renamed
                     return gr.update(visible=mode=="single"), gr.update(visible=mode=="preset"), gr.update(visible=mode=="multi_layer")
                search_extraction_mode.change(toggle_pe_options_search_ui, inputs=[search_extraction_mode],
                                              outputs=[search_single_options, search_preset_options, search_custom_options])

                # Connections
                def refresh_databases_ui(): # Renamed
                    choices = app_state.update_available_databases(verbose=True)
                    return gr.Dropdown(choices=choices if choices else ["No databases found"], value=choices[0] if choices else None)
                refresh_db_button.click(refresh_databases_ui, outputs=[db_dropdown])
                db_dropdown.change(interface.set_active_collection, inputs=[db_dropdown], outputs=[detection_info]) # Use detection_info for status

                def delete_db_wrapper(name): # Renamed
                    s, m = delete_database_collection(name)
                    choices = app_state.update_available_databases(verbose=True)
                    return gr.Dropdown(choices=choices if choices else ["No databases found"], value=choices[0] if choices else None), m
                delete_db_button.click(delete_db_wrapper, inputs=[db_dropdown], outputs=[db_dropdown, detection_info])

                def reset_db_connection_wrapper(): # Renamed
                    cleanup_qdrant_connections(force=True) # From core.database
                    choices = app_state.update_available_databases(verbose=True)
                    return gr.Dropdown(choices=choices if choices else ["No databases found"], value=choices[0] if choices else None), "Database connection reset."
                reset_db_button.click(reset_db_connection_wrapper, outputs=[db_dropdown, detection_info])

                process_button.click(interface.process_image_with_prompt,
                                     inputs=[input_image, text_prompt_search, min_area_ratio_search, max_regions_search,
                                             search_extraction_mode, search_embedding_layer, search_preset_name,
                                             search_custom_layer_1, search_custom_layer_2, search_custom_layer_3,
                                             search_custom_weight_1, search_custom_weight_2, search_custom_weight_3,
                                             search_pooling_strategy, search_temperature],
                                     outputs=[segmented_output, detection_info, region_dropdown, region_preview])
                region_dropdown.change(interface.update_region_preview, inputs=[region_dropdown], outputs=[region_preview])
                search_button.click(interface.search_region,
                                    inputs=[region_dropdown, similarity_slider, max_results_dropdown],
                                    outputs=[search_results_output, search_info, result_selector, result_selector]) # Show result_selector

                process_whole_button.click(interface.process_whole_image,
                                           inputs=[input_image, search_extraction_mode, search_embedding_layer, search_preset_name,
                                                   search_custom_layer_1, search_custom_layer_2, search_custom_layer_3,
                                                   search_custom_weight_1, search_custom_weight_2, search_custom_weight_3,
                                                   search_pooling_strategy, search_temperature],
                                           outputs=[processed_output, detection_info]) # Use common detection_info
                whole_search_button.click(interface.search_whole_image,
                                          inputs=[whole_similarity_slider, whole_max_results_dropdown],
                                          outputs=[search_results_output, search_info, result_selector, result_selector]) # Show result_selector

                result_selector.change(interface.display_enlarged_result, inputs=[result_selector], outputs=[enlarged_result])

            with gr.TabItem("About"):
                gr.Markdown("## About Revers-o")
                # ... (rest of About tab content - kept short for brevity)
                gr.Markdown("Detailed documentation here.")

        if app_state.available_databases: # Initialize active collection for the interface instance
            interface.set_active_collection(app_state.available_databases[0])

    return demo
