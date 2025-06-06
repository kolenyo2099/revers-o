import gradio as gr
from PIL import Image
import numpy as np
from .core import SimpleReverso
from .video_utils import (
    extract_frames_with_progress,
    process_local_videos_with_progress,
)


def create_simple_interface():
    """Create simplified Gradio interface"""

    # Initialize system
    reverso = SimpleReverso()

    # Interface functions
    def detect_and_extract(image, text_prompt, use_direct_pe=False):
        """Detect regions and extract embeddings"""
        if image is None:
            return None, "❌ Please upload an image", None, None

        try:
            if use_direct_pe:
                # Process directly with PE
                embeddings, metadata = reverso.process_image_direct_pe(image)
                result_text = (
                    f"✅ Processed image directly with PE\n"
                    f"🧠 Extracted global image embedding\n"
                    f"🎯 Ready to search!"
                )
                # Create a simple visualization of the full image
                viz_image = image
                # Return empty region options for direct PE
                return (
                    viz_image,
                    result_text,
                    gr.update(choices=[], value=None, visible=False),
                    None,
                )
            else:
                # Detect regions
                num_regions = reverso.detect_regions(image, text_prompt)

                if num_regions == 0:
                    return (
                        None,
                        f"❌ No regions found with prompt: '{text_prompt}'",
                        gr.update(choices=[], value=None, visible=False),
                        None,
                    )

                # Extract embeddings
                embeddings, metadata = reverso.extract_embeddings(image)

                # Create visualization
                viz_image = reverso.visualize_detections(image)

                # Create region options for dropdown
                region_options = []
                for i, meta in enumerate(metadata):
                    confidence = meta.get("confidence", 0.0)
                    detected_class = meta.get("detected_class", "object")
                    region_options.append(
                        f"Region {i+1}: {detected_class} (Confidence: {confidence:.3f})"
                    )

                result_text = (
                    f"✅ Found {num_regions} regions\n"
                    f"🧠 Extracted {len(embeddings)} embeddings\n"
                    f"🎯 Select a region to search with"
                )

                return (
                    viz_image,
                    result_text,
                    gr.update(
                        choices=region_options, value=region_options[0], visible=True
                    ),
                    metadata,
                )

        except Exception as e:
            return (
                None,
                f"❌ Error: {str(e)}",
                gr.update(choices=[], value=None, visible=False),
                None,
            )

    def build_database(folder_path, db_name, db_prompt, use_direct_pe):
        """Build searchable database"""
        if not folder_path or not db_name:
            return "❌ Please provide folder path and database name"

        if not os.path.exists(folder_path):
            return f"❌ Folder not found: {folder_path}"

        try:
            result = reverso.create_database(
                folder_path, db_name, db_prompt, use_direct_pe
            )
            return result
        except Exception as e:
            return f"❌ Error creating database: {str(e)}"

    def search_database(similarity_threshold, max_results, state, selected_region=None):
        """Search for similar regions"""
        try:
            if selected_region is not None:
                # Extract region index from selection (e.g., "Region 1: person" -> 0)
                try:
                    region_index = int(selected_region.split()[1].rstrip(":")) - 1
                    # Use the selected region's embedding
                    if 0 <= region_index < len(reverso.region_embeddings):
                        reverso.region_embeddings = [
                            reverso.region_embeddings[region_index]
                        ]
                    else:
                        print(f"⚠️ Invalid region index: {region_index}")
                except (ValueError, IndexError) as e:
                    print(f"⚠️ Error parsing region index: {e}")
                    # Fall back to first region
                    if reverso.region_embeddings:
                        reverso.region_embeddings = [reverso.region_embeddings[0]]

            result_text, similar_images = reverso.search_similar(
                similarity_threshold, max_results
            )

            # Create a list of image options for the dropdown
            image_options = [
                f"Image {i+1} (Score: {score:.3f})"
                for i, (_, score) in enumerate(similar_images)
                if similar_images[i] is not None
            ]

            # Create a gallery of thumbnails
            thumbnail_gallery = []
            for i, (img, score) in enumerate(similar_images):
                if img is not None:
                    # Create a smaller thumbnail version
                    thumbnail = img.copy()
                    thumbnail.thumbnail((200, 200))  # Resize to thumbnail size
                    thumbnail_gallery.append(thumbnail)

            # Return the results with the new choices and value
            return (
                result_text,
                gr.update(
                    choices=image_options,
                    value=image_options[0] if image_options else None,
                ),
                (
                    similar_images[0][0]
                    if similar_images and similar_images[0] is not None
                    else None
                ),
                similar_images,  # Store images in state
                thumbnail_gallery,  # Return the gallery value directly
            )
        except Exception as e:
            print(f"❌ Search error: {str(e)}")
            import traceback

            traceback.print_exc()
            return (
                f"❌ Search error: {str(e)}",
                gr.update(choices=[], value=None),
                None,
                [],
                [],
            )  # Return empty list for gallery

    def update_similar_image(selected_image, state):
        """Update the displayed similar image based on selection"""
        if not selected_image or not state:
            return None
        try:
            # Extract the image index from the selection (e.g., "Image 1" -> 0)
            index = int(selected_image.split()[1]) - 1
            if 0 <= index < len(state) and state[index] is not None:
                return state[index][0]  # Return the image from the tuple
            return None
        except Exception as e:
            print(f"❌ Error updating image: {str(e)}")
            return None

    def list_available_databases():
        """List all available databases"""
        databases = reverso.list_databases()
        if not databases:
            return "No databases found"
        return "\n".join(f"- {db}" for db in databases)

    def load_selected_database(database_name):
        """Load a selected database"""
        return reverso.load_database(database_name)

    def delete_selected_database(database_name):
        """Delete a selected database"""
        return reverso.delete_database(database_name)

    def unlock_selected_database(database_name):
        """Remove lock file from a selected database"""
        return reverso.unlock_database(database_name)

    def update_region_visualization(selected_region, image):
        """Update the visualization to highlight the selected region"""
        if image is None or selected_region is None:
            return None
        try:
            region_index = int(selected_region.split()[1].rstrip(":")) - 1
            return reverso.visualize_detections(
                image, selected_region_index=region_index
            )
        except Exception as e:
            print(f"❌ Error updating region visualization: {str(e)}")
            return None

    # Create interface
    with gr.Blocks(title="Simple Revers-o: Visual Investigation Tool") as demo:
        # Add state variable at the top level
        similar_images_state = gr.State([])
        detected_image_state = gr.State(None)

        gr.Markdown("# 🔍 Simple Revers-o: Visual Investigation Tool")
        gr.Markdown("**GroundedSAM + PE-Core-L14-336 for visual similarity search**")

        with gr.Tabs():
            # Tab 0: Extract Video Frames (New)
            with gr.TabItem("🎬 Extract Video Frames"):
                gr.Markdown("## Extract Frames from Videos")
                gr.Markdown(
                    "Extract frames from videos for database creation using scene detection"
                )

                # Use tabs for different input methods
                with gr.Tabs():
                    # Tab for URL-based processing
                    with gr.TabItem("🔗 From URLs"):
                        if YT_DLP_AVAILABLE:
                            with gr.Row():
                                with gr.Column():
                                    video_urls = gr.TextArea(
                                        label="Video URLs",
                                        placeholder="https://www.youtube.com/watch?v=..., https://twitter.com/..., etc.\n(comma-separated or one URL per line)",
                                        info="Enter video URLs from YouTube, Twitter, Facebook, Instagram, TikTok, etc.",
                                    )
                                    url_output_folder = gr.Textbox(
                                        label="Output Folder Path",
                                        placeholder="/path/to/save/frames",
                                        info="Folder where extracted frames will be saved",
                                    )
                                    with gr.Row():
                                        frames_per_scene = gr.Slider(
                                            minimum=1,
                                            maximum=10,
                                            value=2,
                                            step=1,
                                            label="Frames per Scene",
                                            info="Number of frames to extract from each detected scene",
                                        )
                                        scene_threshold = gr.Slider(
                                            minimum=10,
                                            maximum=60,
                                            value=30,
                                            step=5,
                                            label="Scene Detection Threshold",
                                            info="Lower values detect more scene changes (more sensitive)",
                                        )
                                    max_quality = gr.Dropdown(
                                        choices=[
                                            "360p",
                                            "480p",
                                            "720p",
                                            "1080p",
                                            "best",
                                        ],
                                        value="720p",
                                        label="Max Video Quality",
                                        info="Higher quality uses more bandwidth and storage",
                                    )
                                    url_extract_btn = gr.Button(
                                        "🎬 Extract Frames from URLs", variant="primary"
                                    )

                                with gr.Column():
                                    url_extraction_status = gr.Markdown(
                                        "URL processing status will appear here"
                                    )

                            # Connect the button to the processing function
                            url_extract_btn.click(
                                extract_frames_with_progress,
                                inputs=[
                                    video_urls,
                                    url_output_folder,
                                    frames_per_scene,
                                    scene_threshold,
                                    max_quality,
                                ],
                                outputs=[url_extraction_status],
                            )
                        else:
                            gr.Markdown("⚠️ **URL video downloading not available.**")
                            gr.Markdown(
                                "Please install yt-dlp to enable URL downloads:"
                            )
                            gr.Code("pip install yt-dlp", language="bash")

                    # Tab for local video processing
                    with gr.TabItem("📁 From Local Files"):
                        with gr.Row():
                            with gr.Column():
                                local_input_folder = gr.Textbox(
                                    label="Video Folder Path",
                                    placeholder="/path/to/videos",
                                    info="Folder containing video files (.mp4, .avi, .mov, etc.)",
                                )
                                local_output_folder = gr.Textbox(
                                    label="Output Folder Path",
                                    placeholder="/path/to/save/frames",
                                    info="Folder where extracted frames will be saved",
                                )
                                with gr.Row():
                                    local_frames_per_scene = gr.Slider(
                                        minimum=1,
                                        maximum=10,
                                        value=2,
                                        step=1,
                                        label="Frames per Scene",
                                        info="Number of frames to extract from each detected scene",
                                    )
                                    local_scene_threshold = gr.Slider(
                                        minimum=10,
                                        maximum=60,
                                        value=30,
                                        step=5,
                                        label="Scene Detection Threshold",
                                        info="Lower values detect more scene changes (more sensitive)",
                                    )
                                local_extract_btn = gr.Button(
                                    "🎬 Extract Frames from Local Videos",
                                    variant="primary",
                                )

                            with gr.Column():
                                local_extraction_status = gr.Markdown(
                                    "Local video processing status will appear here"
                                )

                        # Connect the button to the processing function
                        local_extract_btn.click(
                            process_local_videos_with_progress,
                            inputs=[
                                local_input_folder,
                                local_output_folder,
                                local_frames_per_scene,
                                local_scene_threshold,
                            ],
                            outputs=[local_extraction_status],
                        )

            # Tab 1: Create Database
            with gr.TabItem("🗃️ Create Database"):
                gr.Markdown("## Build a searchable database from your images")

                with gr.Row():
                    with gr.Column():
                        db_folder = gr.Textbox(
                            label="📁 Image Folder Path",
                            placeholder="/path/to/your/images",
                            info="Folder containing images to process",
                        )
                        db_name = gr.Textbox(
                            label="🏷️ Database Name",
                            placeholder="my_investigation_db",
                            info="Name for your database",
                        )
                        db_prompt = gr.Textbox(
                            label="🎯 Detection Prompts",
                            value="person . car . building",
                            info="What to look for (period-separated)",
                        )
                        use_direct_pe = gr.Checkbox(
                            label="🔍 Use Direct PE Processing",
                            value=False,
                            info="Process images directly with PE (no object detection)",
                        )
                        build_btn = gr.Button("🚀 Build Database", variant="primary")

                    with gr.Column():
                        db_status = gr.Textbox(
                            label="📊 Database Status",
                            lines=8,
                            info="Progress and results will appear here",
                        )

                build_btn.click(
                    build_database,
                    inputs=[db_folder, db_name, db_prompt, use_direct_pe],
                    outputs=[db_status],
                )

            # Tab 2: Search Similar
            with gr.TabItem("🔎 Search Similar"):
                gr.Markdown("## Search for similar regions in your database")
                input_image = gr.Image(label="Upload Image", type="pil")
                text_prompt = gr.Textbox(
                    label="Detection Prompt", value="person . car . building"
                )
                use_direct_pe = gr.Checkbox(
                    label="Use Direct PE (no region detection)", value=False
                )
                detect_button = gr.Button("Detect Regions / Extract Embeddings")
                detected_viz = gr.Image(label="Detected Regions", type="pil")
                region_selector = gr.Dropdown(
                    label="Select Region to Search", choices=[], visible=False
                )
                detect_status = gr.Markdown()

                # After detection, update region_selector and detected_viz
                def detect_and_extract_with_state(image, text_prompt, use_direct_pe):
                    viz, status, region_options, metadata = detect_and_extract(
                        image, text_prompt, use_direct_pe
                    )
                    # region_options is a gr.update object
                    return viz, status, region_options, image

                detect_button.click(
                    detect_and_extract_with_state,
                    inputs=[input_image, text_prompt, use_direct_pe],
                    outputs=[
                        detected_viz,
                        detect_status,
                        region_selector,
                        detected_image_state,
                    ],
                )

                # Update visualization when region is selected
                region_selector.change(
                    update_region_visualization,
                    inputs=[region_selector, detected_image_state],
                    outputs=detected_viz,
                )

                # Search controls
                similarity_threshold = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.7,
                    step=0.05,
                    label="🎚️ Similarity Threshold",
                )
                max_results = gr.Dropdown(
                    choices=[3, 5, 10, 20], value=5, label="📊 Max Results"
                )
                search_btn = gr.Button("🎯 Search Database", variant="secondary")
                search_results = gr.Textbox(label="🔍 Search Results", lines=10)
                similar_image_selector = gr.Dropdown(
                    label="Select Similar Image",
                    choices=[],
                    value=None,
                    info="Choose an image to view",
                )
                similar_image_display = gr.Image(
                    label="Selected Similar Image", type="pil"
                )
                thumbnail_gallery = gr.Gallery(
                    label="Search Results Thumbnails", columns=3, height=300
                )

                # Search uses selected region
                search_btn.click(
                    search_database,
                    inputs=[
                        similarity_threshold,
                        max_results,
                        similar_images_state,
                        region_selector,
                    ],
                    outputs=[
                        search_results,
                        similar_image_selector,
                        similar_image_display,
                        similar_images_state,
                        thumbnail_gallery,
                    ],
                )

                similar_image_selector.change(
                    update_similar_image,
                    inputs=[similar_image_selector, similar_images_state],
                    outputs=[similar_image_display],
                )

            # Tab 3: Database Management
            with gr.TabItem("⚙️ Database Management"):
                gr.Markdown("## Manage your databases")

                with gr.Row():
                    with gr.Column():
                        # Automatically list available databases in a dropdown
                        available_databases = reverso.list_databases()
                        db_selector = gr.Dropdown(
                            label="Select Database",
                            choices=available_databases,
                            value=None,
                            info="Choose a database to manage",
                        )

                        with gr.Row():
                            load_btn = gr.Button("📂 Load Database", variant="primary")
                            delete_btn = gr.Button("🗑️ Delete Database", variant="stop")
                            unlock_btn = gr.Button(
                                "🔓 Unlock Database", variant="secondary"
                            )
                            reload_btn = gr.Button(
                                "🔄 Reload List", variant="secondary"
                            )

                        db_status = gr.Textbox(
                            label="Status",
                            lines=3,
                            info="Operation status will appear here",
                        )

                # Connect functions
                load_btn.click(
                    load_selected_database, inputs=[db_selector], outputs=[db_status]
                )

                delete_btn.click(
                    delete_selected_database, inputs=[db_selector], outputs=[db_status]
                )

                unlock_btn.click(
                    unlock_selected_database, inputs=[db_selector], outputs=[db_status]
                )

                def reload_database_list():
                    """Reload the list of available databases"""
                    databases = reverso.list_databases()
                    if not databases:
                        return gr.update(choices=[], value=None), "No databases found"
                    return (
                        gr.update(choices=databases, value=databases[0]),
                        f"✅ Reloaded database list. Found {len(databases)} databases.",
                    )

                reload_btn.click(
                    reload_database_list, inputs=[], outputs=[db_selector, db_status]
                )

            # Tab 4: About
            with gr.TabItem("ℹ️ About"):
                gr.Markdown(
                    """
                ## 🎯 Simple Revers-o: Visual Investigation Tool
                
                **Powered by cutting-edge AI research:**
                - 🤖 **GroundedSAM**: Zero-shot object detection with natural language
                - 🧠 **PE-Core-L14-336**: Meta's Perception Encoder (optimized for investigation)
                - 🎬 **PySceneDetect**: Intelligent scene detection for video keyframe extraction
                
                
                ### 📋 How to Use:
                1. **Extract Video Frames**: Download videos and extract keyframes from detected scenes
                2. **Create Database**: Point to your image folder, set detection prompts
                3. **Search Similar**: Upload query image, detect regions, search database
                4. **Investigate**: Find visually similar regions across your image collection
                
                ### 🎯 Best For:
                - Person identification across image sets
                - Vehicle tracking and matching  
                - Object similarity search
                - Scene and location matching
                - Visual investigation workflows
                
                ### ⚡ Performance:
                - **Model**: PE-Core-L14-336 (1B parameters)
                - **Speed**: ~300ms per image (3x faster than complex approaches)
                - **Memory**: ~4GB VRAM recommended
                - **Quality**: Research-validated optimal layer extraction
                
                *Based on Meta's "Perception Encoder: The best visual embeddings are not at the output of the network" (2025)*
                """
                )

    return demo
