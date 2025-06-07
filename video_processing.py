# Video Processing Functions
# This file contains functions for downloading, processing, and extracting frames from videos.

import os
# import sys # Appears unused
import time
import uuid
import tempfile
# from pathlib import Path # Appears unused
# import numpy as np # Appears unused in this file after refactoring
import cv2 # Used for cv2.VideoCapture, cv2.cvtColor
import shutil
import hashlib # Used for md5 in download_video_from_url
import urllib.parse # Used for urlparse in is_supported_video_url
import yt_dlp # Used for video downloads
import gradio as gr # Used for progress tracking in UI
from scenedetect import open_video, SceneManager, ContentDetector # Core scenedetect components used
# from scenedetect.video_splitter import split_video_ffmpeg # Appears unused
# from scenedetect.scene_detector import AdaptiveDetector # Appears unused
# from tqdm import tqdm # Not directly used; Gradio's Progress handles tqdm integration
# import logging # Not used

# Placeholder for logging configuration if needed (already commented out)
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# It's good practice to define constants or configurations at the top
SUPPORTED_VIDEO_EXTENSIONS = ['.mp4', '.mov', '.avi', '.mkv', '.webm']
DOWNLOAD_DIR = "downloaded_videos"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# Check for yt-dlp availability
try:
    import yt_dlp
    YT_DLP_AVAILABLE = True
except ImportError:
    YT_DLP_AVAILABLE = False
    print("[WARNING] yt-dlp not available. URL video downloads will not work.")

# Check for scenedetect availability (already imported, but good to have a flag)
try:
    # These are already imported: open_video, SceneManager, ContentDetector, split_video_ffmpeg, AdaptiveDetector
    VIDEO_PROCESSING_AVAILABLE = True
except ImportError:
    VIDEO_PROCESSING_AVAILABLE = False
    print("[WARNING] Scene detection libraries not available. Install with: pip install scenedetect")


def is_supported_video_url(url):
    """
    Check if the URL is from a supported video platform.

    Args:
        url: URL string to check

    Returns:
        bool: True if URL is from a supported platform
    """
    if not url or not isinstance(url, str):
        return False

    try:
        parsed = urllib.parse.urlparse(url.strip())
        domain = parsed.netloc.lower()

        # Remove 'www.' prefix if present
        if domain.startswith('www.'):
            domain = domain[4:]

        supported_domains = {
            'youtube.com', 'youtu.be', 'youtube-nocookie.com',
            'twitter.com', 'x.com', 'nitter.net',
            'facebook.com', 'fb.com', 'm.facebook.com',
            'instagram.com', 'tiktok.com', 'vimeo.com',
            'dailymotion.com', 'twitch.tv'
        }

        return domain in supported_domains
    except Exception:
        return False

def download_video_from_url(url, output_dir, max_quality='720p'):
    """
    Download a single video from URL using yt-dlp.

    Args:
        url: Video URL to download
        output_dir: Directory to save the downloaded video
        max_quality: Maximum video quality (e.g., '720p', '1080p', 'best')

    Returns:
        tuple: (success, message, downloaded_file_path)
    """
    if not YT_DLP_AVAILABLE:
        return False, "yt-dlp not available. Please install it: pip install yt-dlp", None

    if not is_supported_video_url(url):
        return False, f"Unsupported URL or invalid format: {url}", None

    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Create a safe filename using timestamp and hash
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        timestamp = int(time.time())
        safe_filename = f"video_{timestamp}_{url_hash}.%(ext)s"

        # Configure yt-dlp options with more predictable filename
        ydl_opts = {
            'outtmpl': os.path.join(output_dir, safe_filename),
            'format': f'best[height<={max_quality[:-1]}]/best',  # Simplified format to avoid merging issues
            'merge_output_format': 'mp4',
            'writeinfojson': False,
            'writethumbnail': False,
            'quiet': True,
            'no_warnings': True,
            'restrictfilenames': True,  # Use only ASCII characters and avoid special characters
            'windowsfilenames': True,   # Avoid characters that are problematic on Windows
        }

        downloaded_files = []

        # Custom hook to capture downloaded files
        def progress_hook(d):
            if d['status'] == 'finished':
                file_path = d['filename']
                print(f"[DEBUG] yt-dlp reported downloaded file: {file_path}")
                # Only add files that exist and are not temporary/intermediate files
                if os.path.exists(file_path) and not any(temp in os.path.basename(file_path) for temp in ['.f', '.part', '.temp']):
                    downloaded_files.append(file_path)
                    print(f"[DEBUG] Final file confirmed and added: {file_path}")
                else:
                    print(f"[DEBUG] Skipping intermediate/temp file: {file_path}")

        ydl_opts['progress_hooks'] = [progress_hook]

        # Download the video
        print(f"[DEBUG] Starting download from: {url}")
        print(f"[DEBUG] Output directory: {output_dir}")
        print(f"[DEBUG] Filename template: {safe_filename}")

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        # Wait a moment for any final processing
        time.sleep(2)

        # Additional verification: check for any video files in the output directory
        if not downloaded_files:
            print(f"[DEBUG] No files captured by progress hook, scanning directory...")
            video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v'}
            for file in os.listdir(output_dir):
                file_path = os.path.join(output_dir, file)
                if (os.path.splitext(file.lower())[1] in video_extensions and
                    os.path.isfile(file_path) and
                    not any(temp in file for temp in ['.f', '.part', '.temp'])):
                    # Check if this is a recently created file (within last 120 seconds)
                    if os.path.getmtime(file_path) > time.time() - 120:
                        downloaded_files.append(file_path)
                        print(f"[DEBUG] Found recent video file: {file_path}")

        if downloaded_files:
            # Sort by modification time and take the most recent
            downloaded_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            final_path = downloaded_files[0]
            file_size = os.path.getsize(final_path)
            print(f"[DEBUG] Successfully downloaded: {final_path} ({file_size} bytes)")
            return True, f"Successfully downloaded video ({file_size} bytes)", final_path
        else:
            return False, "Download completed but no valid video file was created", None

    except Exception as e:
        error_msg = str(e)
        print(f"[DEBUG] Download exception: {error_msg}")
        if "Private video" in error_msg:
            return False, "Video is private or requires authentication", None
        elif "Video unavailable" in error_msg:
            return False, "Video is unavailable or has been removed", None
        elif "Unsupported URL" in error_msg:
            return False, f"URL not supported by yt-dlp: {url}", None
        else:
            return False, f"Download failed: {error_msg}", None

def extract_frames_from_video(video_path, output_folder, frames_per_scene=2, scene_threshold=30.0):
    """
    Extract keyframes from a video using scene detection.

    Args:
        video_path: Path to the input video file
        output_folder: Folder to save extracted frames
        frames_per_scene: Number of frames to extract per detected scene
        scene_threshold: Threshold for scene detection (lower = more sensitive)

    Returns:
        tuple: (success, message, list_of_extracted_frames)
    """
    if not VIDEO_PROCESSING_AVAILABLE:
        print("[WARNING] Scene detection not available, falling back to uniform extraction")
        return extract_uniform_frames(video_path, output_folder, 20)  # Fallback to uniform extraction

    try:
        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)

        print(f"[DEBUG] Initializing video manager for: {video_path}")

        # Initialize video manager and scene manager for scene detection
        # video_manager = VideoManager([video_path]) # This is from older scenedetect
        video = open_video(video_path) # PySceneDetect v0.6+
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=scene_threshold))

        # Detect scenes
        # video_manager.set_duration() # Not needed for open_video
        # video_manager.start() # Not needed for open_video
        scene_manager.detect_scenes(video=video) # Pass video object
        scene_list = scene_manager.get_scene_list()
        # video_manager.release() # Not needed for open_video, 'video' is auto-closed or use 'del video'

        print(f"[VIDEO] Detected {len(scene_list)} scenes in {video_path}")

        # If no scenes detected, use uniform sampling
        if not scene_list:
            print("[WARNING] No scenes detected, falling back to uniform extraction")
            return extract_uniform_frames(video_path, output_folder, 20)

        # Extract keyframes from scenes
        extracted_frames = []

        # Use OpenCV for frame extraction (already imported as cv2)
        # from PIL import Image # Already imported

        # Re-open video with OpenCV for frame grabbing if 'video' object from scenedetect is not suitable
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[ERROR] OpenCV could not open video: {video_path}")
            return False, f"Could not open video file with OpenCV: {video_path}", []

        total_frames_cv = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # Renamed to avoid clash
        fps_cv = cap.get(cv2.CAP_PROP_FPS) # Renamed to avoid clash

        print(f"[DEBUG] Video properties (OpenCV) - Total frames: {total_frames_cv}, FPS: {fps_cv}")

        frame_count = 0
        for i, scene in enumerate(scene_list):
            start_time = scene[0].get_seconds()
            end_time = scene[1].get_seconds()

            # Extract frames uniformly from this scene
            scene_duration = end_time - start_time
            if scene_duration > 0:
                for j in range(frames_per_scene):
                    # Calculate frame position within scene
                    time_offset = (j + 0.5) * scene_duration / frames_per_scene
                    frame_time = start_time + time_offset
                    frame_number = int(frame_time * fps_cv) # Use OpenCV fps

                    # Extract frame
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                    ret, frame = cap.read()

                    if ret:
                        # Convert BGR to RGB
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                        # Save frame
                        video_name = os.path.splitext(os.path.basename(video_path))[0]
                        frame_filename = f"{video_name}_scene{i:03d}_frame{j:03d}.jpg"
                        frame_path = os.path.join(output_folder, frame_filename)

                        # Convert to PIL and save
                        from PIL import Image # Ensure PIL Image is available
                        pil_image = Image.fromarray(frame_rgb)
                        pil_image.save(frame_path, 'JPEG', quality=95)

                        extracted_frames.append(frame_path)
                        frame_count += 1

                        print(f"[VIDEO] Extracted frame {frame_count}/{frames_per_scene * len(scene_list)}: {frame_filename}")

        cap.release()
        del video # Clean up scenedetect video object

        return True, f"Successfully extracted {len(extracted_frames)} keyframes from {len(scene_list)} scenes", extracted_frames

    except Exception as e:
        print(f"[ERROR] Video processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False, f"Error processing video: {str(e)}", []

def extract_uniform_frames(video_path, output_folder, num_frames=20):
    """
    Extract frames uniformly distributed across the video duration.

    Args:
        video_path: Path to the input video file
        output_folder: Folder to save extracted frames
        num_frames: Maximum number of frames to extract

    Returns:
        tuple: (success, message, extracted_frames_list)
    """
    try:
        os.makedirs(output_folder, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False, f"Could not open video file: {video_path}", []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0: # Avoid division by zero if fps is not available
            return False, f"Could not determine FPS for video: {video_path}", []
        duration = total_frames / fps

        print(f"[VIDEO] Video has {total_frames} frames, {duration:.2f} seconds duration")

        extracted_frames = []
        if total_frames == 0 or num_frames == 0: # handle cases with no frames or no requested frames
             cap.release()
             return True, "No frames to extract or requested.", []

        frame_interval = max(1, total_frames // num_frames)

        for i in range(0, total_frames, frame_interval):
            if len(extracted_frames) >= num_frames:
                break

            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()

            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Save frame
                video_name = os.path.splitext(os.path.basename(video_path))[0]
                frame_filename = f"{video_name}_uniform_{len(extracted_frames):03d}.jpg"
                frame_path = os.path.join(output_folder, frame_filename)

                # Convert to PIL and save
                from PIL import Image # Ensure PIL Image is available
                pil_image = Image.fromarray(frame_rgb)
                pil_image.save(frame_path, 'JPEG', quality=95)

                extracted_frames.append(frame_path)
                print(f"[VIDEO] Extracted uniform frame {len(extracted_frames)}/{num_frames}: {frame_filename}")

        cap.release()

        return True, f"Successfully extracted {len(extracted_frames)} frames uniformly", extracted_frames

    except Exception as e:
        print(f"[ERROR] Uniform frame extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return False, f"Error extracting uniform frames: {str(e)}", []

def extract_frames_with_progress(urls, folder, fps=2, thresh=30, qual="720p", progress=None):
    """Extract frames from videos with progress updates for Gradio."""
    if not urls:
        return "❌ No URLs provided"

    os.makedirs(folder, exist_ok=True)
    urls = [url.strip() for url in urls.replace(',', '\n').split('\n') if url.strip()]
    if not urls:
        return "❌ No valid URLs provided"

    status_messages = []
    def log_status(message, progress_value=None):
        status_messages.append(message)
        if progress:
            if progress_value is not None:
                progress(progress_value, desc=message)
            else:
                progress(0.0, desc=message)
        return "\n".join(status_messages)

    log_status("🔍 Checking video availability...", 0.0)
    valid_urls = []
    for i, url in enumerate(urls):
        try:
            with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                ydl.extract_info(url, download=False)
            valid_urls.append(url)
            log_status(f"✅ Valid URL: {url}", 0.1 * (i + 1) / len(urls))
        except Exception as e:
            log_status(f"⚠️ Skipping invalid URL {url}: {str(e)}", 0.1 * (i + 1) / len(urls))

    if not valid_urls:
        return "❌ No valid video URLs found"

    log_status(f"✅ Found {len(valid_urls)} valid URLs", 0.1)
    total_frames = 0

    for i, url in enumerate(valid_urls):
        try:
            log_status(f"📥 Downloading video {i+1}/{len(valid_urls)}...", 0.1 + (0.4 * i / len(valid_urls)))
            with tempfile.TemporaryDirectory() as temp_dir:
                ydl_opts = {
                    'format': f'bestvideo[height<={qual[:-1]}][ext=mp4]+bestaudio[ext=m4a]/best[height<={qual[:-1]}]',
                    'outtmpl': os.path.join(temp_dir, '%(id)s.%(ext)s'),
                    'quiet': True
                }
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=True)
                    video_path = os.path.join(temp_dir, f"{info['id']}.{info['ext']}")
                log_status(f"✅ Downloaded video {i+1}/{len(valid_urls)}", 0.1 + (0.4 * (i + 1) / len(valid_urls)))

                log_status(f"🎬 Processing video {i+1}/{len(valid_urls)}...", 0.5 + (0.4 * i / len(valid_urls)))
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    log_status(f"❌ Could not open video: {url}", 0.5 + (0.4 * (i + 1) / len(valid_urls)))
                    continue

                # Get video info
                fps_video = cap.get(cv2.CAP_PROP_FPS)
                total_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = total_frames_video / fps_video

                # Scene detection
                scene_manager = SceneManager()
                scene_manager.add_detector(ContentDetector(threshold=thresh))
                video = open_video(video_path)
                scene_manager.detect_scenes(video)
                scenes = scene_manager.get_scene_list()

                if not scenes:
                    log_status(f"⚠️ No scenes detected in {url}, using uniform sampling")
                    frame_interval = int(fps_video / fps)
                    for frame_idx in range(0, total_frames_video, frame_interval):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                        ret, frame = cap.read()
                        if ret:
                            timestamp = frame_idx / fps_video
                            minutes = int(timestamp // 60)
                            seconds = int(timestamp % 60)
                            frame_name = f"{info['id']}_{minutes:02d}m{seconds:02d}s.jpg"
                            frame_path = os.path.join(folder, frame_name)
                            cv2.imwrite(frame_path, frame)
                            total_frames += 1
                else:
                    log_status(f"✅ Detected {len(scenes)} scenes")
                    for scene in scenes:
                        start_frame = scene[0].frame_num
                        end_frame = scene[1].frame_num
                        frame_interval = max(1, (end_frame - start_frame) // fps)
                        
                        for frame_idx in range(start_frame, end_frame, frame_interval):
                            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                            ret, frame = cap.read()
                            if ret:
                                timestamp = frame_idx / fps_video
                                minutes = int(timestamp // 60)
                                seconds = int(timestamp % 60)
                                frame_name = f"{info['id']}_{minutes:02d}m{seconds:02d}s.jpg"
                                frame_path = os.path.join(folder, frame_name)
                                cv2.imwrite(frame_path, frame)
                                total_frames += 1

                cap.release()
                log_status(f"✅ Extracted {total_frames} frames from {url}", 0.5 + (0.4 * (i + 1) / len(valid_urls)))

        except Exception as e:
            log_status(f"❌ Error processing {url}: {str(e)}", 0.5 + (0.4 * (i + 1) / len(valid_urls)))
            import traceback; traceback.print_exc()

    return f"✅ Completed! Extracted {total_frames} frames from {len(valid_urls)} videos to {folder}"

def process_local_videos_with_progress(input_folder, output_folder, fps, thresh, progress=None):
    """Process all videos in a folder with progress tracking."""
    if not input_folder or not output_folder:
        return "❌ Please provide input and output folders"

    os.makedirs(output_folder, exist_ok=True)
    video_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    
    if not video_files:
        return "❌ No video files found in the input folder."

    status_messages = []
    def log_status(message, progress_value=None):
        status_messages.append(message)
        try:
            if progress_value is not None:
                progress(progress_value, desc=message)
            else:
                progress(0.0, desc=message)
        except Exception:
            pass  # Ignore progress update errors
        return "\n".join(status_messages)

    log_status(f"📁 Found {len(video_files)} videos to process", 0.0)
    total_frames = 0

    for i, video_file in enumerate(video_files):
        try:
            video_path = os.path.join(input_folder, video_file)
            log_status(f"🎬 Processing video {i+1}/{len(video_files)}: {video_file}", 0.2 * (i / len(video_files)))

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                log_status(f"❌ Could not open video: {video_file}", 0.2 * ((i + 1) / len(video_files)))
                continue

            # Get video info
            fps_video = cap.get(cv2.CAP_PROP_FPS)
            total_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames_video / fps_video

            # Scene detection
            scene_manager = SceneManager()
            scene_manager.add_detector(ContentDetector(threshold=thresh))
            video = open_video(video_path)
            scene_manager.detect_scenes(video)
            scenes = scene_manager.get_scene_list()

            if not scenes:
                log_status(f"⚠️ No scenes detected in {video_file}, using uniform sampling", 0.4 * (i / len(video_files)))
                frame_interval = int(fps_video / fps)
                for frame_idx in range(0, total_frames_video, frame_interval):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()
                    if ret:
                        timestamp = frame_idx / fps_video
                        minutes = int(timestamp // 60)
                        seconds = int(timestamp % 60)
                        frame_name = f"{os.path.splitext(video_file)[0]}_{minutes:02d}m{seconds:02d}s.jpg"
                        frame_path = os.path.join(output_folder, frame_name)
                        cv2.imwrite(frame_path, frame)
                        total_frames += 1
            else:
                log_status(f"✅ Detected {len(scenes)} scenes", 0.4 * (i / len(video_files)))
                for scene in scenes:
                    start_frame = scene[0].frame_num
                    end_frame = scene[1].frame_num
                    frame_interval = max(1, (end_frame - start_frame) // fps)
                    
                    for frame_idx in range(start_frame, end_frame, frame_interval):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                        ret, frame = cap.read()
                        if ret:
                            timestamp = frame_idx / fps_video
                            minutes = int(timestamp // 60)
                            seconds = int(timestamp % 60)
                            frame_name = f"{os.path.splitext(video_file)[0]}_{minutes:02d}m{seconds:02d}s.jpg"
                            frame_path = os.path.join(output_folder, frame_name)
                            cv2.imwrite(frame_path, frame)
                            total_frames += 1

            cap.release()
            log_status(f"✅ Extracted frames from {video_file}", 0.2 * ((i + 1) / len(video_files)))

        except Exception as e:
            log_status(f"❌ Error processing {video_file}: {str(e)}", 0.2 * ((i + 1) / len(video_files)))
            continue

    return f"✅ Completed! Extracted {total_frames} frames from {len(video_files)} videos to {output_folder}"

# Gradio import for progress, will only be used if Gradio is installed and these functions are called from Gradio context
try:
    import gradio as gr
except ImportError:
    gr = None # type: ignore
    # Define a dummy gr.Progress if gradio is not available, so the functions don't break
    # when called without a progress object from a non-Gradio context.
    class DummyProgress:
        def __init__(self, *args, **kwargs):
            pass
        def __call__(self, *args, **kwargs):
            pass # Does nothing

    if gr is None: # If gradio is not imported, use dummy
        gr = type('GradioModuleMock', (), {'Progress': DummyProgress})()
