#!/usr/bin/env python3
"""
Revers-o: GroundedSAM + Perception Encoders Image Similarity Search Application
Main application file containing all core functionality
"""

import argparse
import os
import sys
import time
import gc
import json
import uuid
import math
import io
import tempfile
import subprocess
import shutil
import urllib.parse
import hashlib
from contextlib import nullcontext
from pathlib import Path
from typing import List, Optional

# Core imports
import cv2
import numpy as np
import torch
import torch.nn.functional as F # Added for PE pooling
import torchvision # Added for NMS
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib import gridspec
from PIL import Image
import requests
from io import BytesIO
from tqdm import tqdm
import gradio as gr

# ML/CV imports
from autodistill_grounded_sam import GroundedSAM
from autodistill.detection import CaptionOntology
from qdrant_client import QdrantClient
from qdrant_client.http import models

# Video processing imports
try:
    from scenedetect import VideoManager, SceneManager
    from scenedetect.detectors import ContentDetector
    import imageio
    VIDEO_PROCESSING_AVAILABLE = True
except ImportError as e:
    print(f"[WARNING] Video processing libraries not available: {e}")
    VIDEO_PROCESSING_AVAILABLE = False

# Check for yt-dlp availability
try:
    import yt_dlp
    YT_DLP_AVAILABLE = True
except ImportError:
    YT_DLP_AVAILABLE = False
    print("[WARNING] yt-dlp not available. URL video downloads will not work.")

# Add perception_models to path
sys.path.append('./perception_models')
import core.vision_encoder.pe as pe
import core.vision_encoder.transforms as transforms

# Global variables for models and device
pe_model = None
pe_vit_model = None
preprocess = None
device = None

# =============================================================================
# SIMILARITY FUNCTIONS
# =============================================================================

def temperature_scaled_similarity(query_emb, candidate_emb, temperature=0.07):
    """PE-inspired temperature-scaled cosine similarity"""
    if not isinstance(query_emb, torch.Tensor):
        query_emb = torch.tensor(query_emb, dtype=torch.float32)
    if not isinstance(candidate_emb, torch.Tensor):
        candidate_emb = torch.tensor(candidate_emb, dtype=torch.float32)
    
    # Move to common device, e.g. query_emb's device or CPU
    # If query_emb is on CUDA, candidate_emb should also be moved to CUDA.
    # If both are CPU, this does nothing.
    # If one is CPU and other CUDA, it standardizes to query_emb's device.
    device_to_use = query_emb.device
    candidate_emb = candidate_emb.to(device_to_use)

    if query_emb.ndim == 1:
        query_emb = query_emb.unsqueeze(0) 
    if candidate_emb.ndim == 1: 
        # If candidate_emb was (D), it becomes (1,D).
        # If it was (N,D) for batch comparison, it remains (N,D) due to check query_emb.ndim == 1 for candidate_emb.ndim == 1
        pass # Keep as is, assuming it could be (N,D) for batch comparison
    elif candidate_emb.ndim == 2 and query_emb.shape[-1] == candidate_emb.shape[-1]: # (N,D)
        pass # Correct shape for batched comparison
    else: # Fallback for unexpected shapes, try to make it (1,D)
        candidate_emb = candidate_emb.flatten().unsqueeze(0)
        if candidate_emb.shape[-1] != query_emb.shape[-1]:
            raise ValueError(f"Candidate embedding reshaped to {candidate_emb.shape} but does not match query embedding dim {query_emb.shape[-1]}")


    query_emb_norm = F.normalize(query_emb, p=2, dim=-1)
    candidate_emb_norm = F.normalize(candidate_emb, p=2, dim=-1)
    
    # Cosine similarity will be (1, N) if query is (1,D) and candidates are (N,D)
    # Or (1) if query is (1,D) and candidate is (1,D)
    cosine_sim = F.cosine_similarity(query_emb_norm, candidate_emb_norm, dim=-1) 
    
    # Temperature scaling
    # Ensure temperature is not too close to zero to avoid overflow with exp
    safe_temperature = max(float(temperature), 1e-6)
    scaled_similarity = torch.exp(cosine_sim / safe_temperature)
    
    return scaled_similarity

# =============================================================================
# APPLICATION STATE MANAGEMENT
# =============================================================================

class AppState:
    """Manages the application state for the multi-mode interface"""
    def __init__(self):
        self.mode = "landing"  # "landing", "building", "searching"
        self.available_databases = []
        self.active_database = None
        print(f"[STATUS] Initializing App State")
        self.update_available_databases(verbose=False)
    
    def update_available_databases(self, verbose=True):
        """Update the list of available databases"""
        if verbose:
            print(f"[STATUS] Refreshing database list")
        
        self.available_databases = list_available_databases()
        
        if verbose:
            print(f"[STATUS] Found {len(self.available_databases)} databases")
            
        return self.available_databases

def list_available_databases():
    """Find all existing Qdrant collections in the project directory"""
    collections = []
    
    # Always use the Qdrant API as the primary source of truth
    qdrant_data_dir = os.path.join("image_retrieval_project", "qdrant_data")
    if os.path.exists(qdrant_data_dir):
        try:
            # Connect to Qdrant and get collections directly
            client = QdrantClient(path=qdrant_data_dir)
            collection_list = client.get_collections().collections
            collections = [c.name for c in collection_list]
            client.close()
            print(f"[DEBUG] Found collections via Qdrant API: {collections}")
        except Exception as e:
            print(f"[DEBUG] Error querying Qdrant API: {e}")
            
            # Fallback to filesystem checks if API fails
            # Path 1: Check the 'collection' directory (singular)
            collection_dir = os.path.join(qdrant_data_dir, "collection")
            if os.path.exists(collection_dir):
                collections.extend([d for d in os.listdir(collection_dir) if os.path.isdir(os.path.join(collection_dir, d))])
                print(f"[DEBUG] Found collections in 'collection' directory: {collections}")
            
            # Path 2: Check the 'collections' directory (plural)
            collections_dir = os.path.join(qdrant_data_dir, "collections")
            if os.path.exists(collections_dir):
                plural_collections = [d for d in os.listdir(collections_dir) if os.path.isdir(os.path.join(collections_dir, d))]
                collections.extend(plural_collections)
                print(f"[DEBUG] Found collections in 'collections' directory: {plural_collections}")
    
    # Remove duplicates while preserving order
    unique_collections = []
    for c in collections:
        if c not in unique_collections:
            unique_collections.append(c)
    
    # Filter out any collections that start with "." (hidden folders)
    unique_collections = [c for c in unique_collections if not c.startswith(".")]
    
    if not unique_collections:
        print(f"[STATUS] No database collections found")
    else:
        print(f"[STATUS] Available collections: {unique_collections}")
    
    return unique_collections
    
# Add the new function to delete a database collection
def delete_database_collection(collection_name):
    """Properly delete a database collection using the Qdrant API"""
    if not collection_name or collection_name == "No databases found":
        return False, "No valid collection specified"
        
    try:
        print(f"[STATUS] Attempting to delete collection: {collection_name}")
        qdrant_data_dir = os.path.join("image_retrieval_project", "qdrant_data")
        
        # Connect to Qdrant
        client = QdrantClient(path=qdrant_data_dir)
        
        # Check if collection exists
        collections = client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if collection_name not in collection_names:
            client.close()
            return False, f"Collection '{collection_name}' does not exist"
            
        # Delete the collection
        client.delete_collection(collection_name=collection_name)
        client.close()
        
        print(f"[STATUS] Successfully deleted collection: {collection_name}")
        return True, f"Successfully deleted database: {collection_name}"
    except Exception as e:
        print(f"[ERROR] Failed to delete collection {collection_name}: {e}")
        import traceback
        traceback.print_exc()
        return False, f"Error deleting database: {str(e)}"

# =============================================================================
# DEVICE SETUP AND MODEL LOADING
# =============================================================================

def setup_device():
    """Setup the appropriate compute device"""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Using MPS device: {device}")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {device}")
    else:
        device = torch.device("cpu")
        print(f"Using CPU device: {device}")
    return device

def load_pe_model(device):
    """Load Perception Encoder model"""
    print("Available PE configurations:", pe.CLIP.available_configs())

    # Try different models in order of preference
    model_name = 'PE-Core-G14-448' 
    if model_name in pe.CLIP.available_configs():
        pe_model = pe.CLIP.from_config(model_name, pretrained=True)
    elif "PE-Spatial-L14-336" in pe.CLIP.available_configs():
        pe_model = pe.CLIP.from_config("PE-Core-G14-448", pretrained=True)
    else:
        available_configs = pe.CLIP.available_configs()
        if available_configs:
            pe_model = pe.CLIP.from_config(available_configs[0], pretrained=True)
        else:
            raise Exception("No PE model configurations available")

    # For Perception Encoder, we should prefer VisionTransformer over CLIP
    # when we need intermediate layer access
    if hasattr(pe, 'VisionTransformer') and pe.VisionTransformer.available_configs():
        print("Available PE VisionTransformer configs:", pe.VisionTransformer.available_configs())
        vit_configs = pe.VisionTransformer.available_configs()
        # Prefer PE-Spatial variants for region work
        spatial_configs = [c for c in vit_configs if 'Spatial' in c]
        if spatial_configs:
            vit_model = pe.VisionTransformer.from_config(spatial_configs[0], pretrained=True)
            print(f"Loaded Vision Transformer model: {spatial_configs[0]} for intermediate layer access")
            vit_model = vit_model.to(device)
            # Keep both models - one for feature extraction, one for final embedding if needed
            preprocess = transforms.get_image_transform(pe_model.image_size)
            return pe_model, vit_model, preprocess
    
    pe_model = pe_model.to(device)
    preprocess = transforms.get_image_transform(pe_model.image_size)
    return pe_model, None, preprocess

# =============================================================================
# VIDEO PROCESSING FUNCTIONS
# =============================================================================

def extract_keyframes_from_video(video_path, output_folder, max_frames=30, scene_threshold=30.0):
    """
    Extract keyframes from a video using scene detection and uniform sampling.
    
    Args:
        video_path: Path to the input video file
        output_folder: Folder to save extracted frames
        max_frames: Maximum number of frames to extract
        scene_threshold: Threshold for scene detection (lower = more sensitive)
    
    Returns:
        tuple: (success, message, extracted_frames_list)
    """
    if not VIDEO_PROCESSING_AVAILABLE:
        return False, "Video processing libraries not available. Please install scenedetect and imageio.", []
    
    # Debug: Check if video file exists
    print(f"[DEBUG] Checking video file: {video_path}")
    if not os.path.exists(video_path):
        print(f"[ERROR] Video file does not exist: {video_path}")
        return False, f"Video file not found: {video_path}", []
    
    if not os.path.isfile(video_path):
        print(f"[ERROR] Path is not a file: {video_path}")
        return False, f"Path is not a file: {video_path}", []
    
    file_size = os.path.getsize(video_path)
    print(f"[DEBUG] Video file exists, size: {file_size} bytes")
    
    if file_size == 0:
        print(f"[ERROR] Video file is empty: {video_path}")
        return False, f"Video file is empty: {video_path}", []

    try:
        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        
        print(f"[DEBUG] Initializing video manager for: {video_path}")
        
        # Initialize video manager and scene manager
        video_manager = VideoManager([video_path])
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=scene_threshold))
        
        # Detect scenes
        video_manager.set_duration()
        video_manager.start()
        scene_manager.detect_scenes(frame_source=video_manager)
        scene_list = scene_manager.get_scene_list()
        video_manager.release()
        
        print(f"[VIDEO] Detected {len(scene_list)} scenes in {video_path}")
        
        # If no scenes detected, use uniform sampling
        if not scene_list:
            return extract_uniform_frames(video_path, output_folder, max_frames)
        
        # Extract keyframes from scenes
        extracted_frames = []
        frames_per_scene = max(1, max_frames // len(scene_list))
        
        # Use OpenCV for frame extraction
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[ERROR] OpenCV could not open video: {video_path}")
            return False, f"Could not open video file with OpenCV: {video_path}", []
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"[DEBUG] Video properties - Total frames: {total_frames}, FPS: {fps}")
        
        frame_count = 0
        for i, scene in enumerate(scene_list):
            if frame_count >= max_frames:
                break
                
            start_time = scene[0].get_seconds()
            end_time = scene[1].get_seconds()
            
            # Extract frames uniformly from this scene
            scene_duration = end_time - start_time
            if scene_duration > 0:
                for j in range(frames_per_scene):
                    if frame_count >= max_frames:
                        break
                        
                    # Calculate frame position within scene
                    time_offset = (j + 0.5) * scene_duration / frames_per_scene
                    frame_time = start_time + time_offset
                    frame_number = int(frame_time * fps)
                    
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
                        pil_image = Image.fromarray(frame_rgb)
                        pil_image.save(frame_path, 'JPEG', quality=95)
                        
                        extracted_frames.append(frame_path)
                        frame_count += 1
                        
                        print(f"[VIDEO] Extracted frame {frame_count}/{max_frames}: {frame_filename}")
        
        cap.release()
        
        return True, f"Successfully extracted {len(extracted_frames)} keyframes from {len(scene_list)} scenes", extracted_frames
        
    except Exception as e:
        print(f"[ERROR] Video processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False, f"Error processing video: {str(e)}", []

def extract_uniform_frames(video_path, output_folder, max_frames=30):
    """
    Extract frames uniformly distributed across the video duration.
    
    Args:
        video_path: Path to the input video file
        output_folder: Folder to save extracted frames
        max_frames: Maximum number of frames to extract
    
    Returns:
        tuple: (success, message, extracted_frames_list)
    """
    try:
        os.makedirs(output_folder, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps
        
        print(f"[VIDEO] Video has {total_frames} frames, {duration:.2f} seconds duration")
        
        extracted_frames = []
        frame_interval = max(1, total_frames // max_frames)
        
        for i in range(0, total_frames, frame_interval):
            if len(extracted_frames) >= max_frames:
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
                pil_image = Image.fromarray(frame_rgb)
                pil_image.save(frame_path, 'JPEG', quality=95)
                
                extracted_frames.append(frame_path)
                print(f"[VIDEO] Extracted uniform frame {len(extracted_frames)}/{max_frames}: {frame_filename}")
        
        cap.release()
        
        return True, f"Successfully extracted {len(extracted_frames)} frames uniformly", extracted_frames
        
    except Exception as e:
        print(f"[ERROR] Uniform frame extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return False, f"Error extracting uniform frames: {str(e)}", []

def process_video_folder(input_folder, output_folder, max_frames_per_video=30, scene_threshold=30.0):
    """
    Process all videos in a folder to extract keyframes.
    
    Args:
        input_folder: Folder containing video files
        output_folder: Folder to save extracted frames
        max_frames_per_video: Maximum frames to extract per video
        scene_threshold: Scene detection threshold
    
    Returns:
        Generator yielding progress updates
    """
    if not VIDEO_PROCESSING_AVAILABLE:
        yield "❌ Video processing libraries not available"
        return
    
    # Supported video extensions
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v'}
    
    # Find all video files
    video_files = []
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if os.path.splitext(file.lower())[1] in video_extensions:
                video_files.append(os.path.join(root, file))
    
    if not video_files:
        yield "❌ No video files found in the specified folder"
        return
    
    yield f"📹 Found {len(video_files)} video files to process"
    
    total_extracted = 0
    successful_videos = 0
    
    for i, video_path in enumerate(video_files):
        video_name = os.path.basename(video_path)
        yield f"🎬 Processing video {i+1}/{len(video_files)}: {video_name}"
        
        # Create subfolder for this video's frames
        video_output_folder = os.path.join(output_folder, os.path.splitext(video_name)[0])
        
        try:
            success, message, extracted_frames = extract_keyframes_from_video(
                video_path, video_output_folder, max_frames_per_video, scene_threshold
            )
            
            if success:
                total_extracted += len(extracted_frames)
                successful_videos += 1
                yield f"✅ {video_name}: {message}"
            else:
                yield f"❌ {video_name}: {message}"
                
        except Exception as e:
            yield f"❌ {video_name}: Error - {str(e)}"
    
    yield f"🎉 Processing complete! Successfully processed {successful_videos}/{len(video_files)} videos"
    yield f"📊 Total frames extracted: {total_extracted}"
    yield f"💾 Frames saved to: {output_folder}"

# =============================================================================
# VIDEO URL DOWNLOAD FUNCTIONS
# =============================================================================

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

def process_video_urls(urls_text, output_folder, max_frames_per_video=30, scene_threshold=30.0, max_quality='720p'):
    """
    Process videos from URLs by downloading them first, then extracting keyframes.
    
    Args:
        urls_text: Text containing URLs (one per line)
        output_folder: Folder to save extracted frames  
        max_frames_per_video: Maximum frames to extract per video
        scene_threshold: Scene detection threshold
        max_quality: Maximum video quality for downloads
        
    Returns:
        Generator yielding progress updates
    """
    if not VIDEO_PROCESSING_AVAILABLE:
        yield "❌ Video processing libraries not available"
        return
        
    if not YT_DLP_AVAILABLE:
        yield "❌ yt-dlp not available. Please install it: pip install yt-dlp"
        return
    
    # Parse URLs from input text
    urls = [url.strip() for url in urls_text.split('\n') if url.strip()]
    
    if not urls:
        yield "❌ No URLs provided"
        return
    
    # Filter out invalid URLs
    valid_urls = [url for url in urls if is_supported_video_url(url)]
    invalid_urls = [url for url in urls if not is_supported_video_url(url)]
    
    if invalid_urls:
        yield f"⚠️ Skipping {len(invalid_urls)} invalid/unsupported URLs"
    
    if not valid_urls:
        yield "❌ No valid video URLs found"
        return
    
    yield f"🔗 Found {len(valid_urls)} valid video URLs to process"
    
    # Create temporary directory for downloads
    temp_download_dir = tempfile.mkdtemp(prefix="revers_o_downloads_")
    
    try:
        total_extracted = 0
        successful_videos = 0
        downloaded_files = []
        
        # Download all videos first
        yield f"📥 Starting downloads to temporary directory..."
        
        for i, url in enumerate(valid_urls):
            yield f"📥 Downloading video {i+1}/{len(valid_urls)}: {url}"
            
            try:
                success, message, downloaded_file = download_video_from_url(url, temp_download_dir, max_quality)
                
                if success and downloaded_file:
                    downloaded_files.append(downloaded_file)
                    yield f"✅ Downloaded: {os.path.basename(downloaded_file)}"
                else:
                    yield f"❌ Failed to download {url}: {message}"
                    
            except Exception as e:
                yield f"❌ Error downloading {url}: {str(e)}"
        
        if not downloaded_files:
            yield "❌ No videos were successfully downloaded"
            return
        
        yield f"🎬 Successfully downloaded {len(downloaded_files)} videos. Starting frame extraction..."
        
        # Now process the downloaded videos
        for i, video_path in enumerate(downloaded_files):
            video_name = os.path.basename(video_path)
            yield f"🎬 Processing video {i+1}/{len(downloaded_files)}: {video_name}"
            
            # Create subfolder for this video's frames
            video_output_folder = os.path.join(output_folder, os.path.splitext(video_name)[0])
            
            try:
                success, message, extracted_frames = extract_keyframes_from_video(
                    video_path, video_output_folder, max_frames_per_video, scene_threshold
                )
                
                if success:
                    total_extracted += len(extracted_frames)
                    successful_videos += 1
                    yield f"✅ {video_name}: {message}"
                else:
                    yield f"❌ {video_name}: {message}"
                    
            except Exception as e:
                yield f"❌ {video_name}: Error - {str(e)}"
        
        yield f"🎉 Processing complete! Successfully processed {successful_videos}/{len(downloaded_files)} videos"
        yield f"📊 Total frames extracted: {total_extracted}"
        yield f"💾 Frames saved to: {output_folder}"
        
    finally:
        # Cleanup temporary directory
        try:
            shutil.rmtree(temp_download_dir)
            yield f"🧹 Cleaned up temporary downloads"
        except Exception as e:
            yield f"⚠️ Warning: Could not clean up temporary directory: {str(e)}"

# =============================================================================
# IMAGE PROCESSING FUNCTIONS
# =============================================================================

def download_image(url):
    """Download an image from a URL"""
    response = requests.get(url)
    return Image.open(BytesIO(response.content)).convert("RGB")

def load_local_image(path):
    """Load an image from local storage"""
    return Image.open(path).convert("RGB")

def tokens_to_grid(tokens, model):
    """Convert tokens to spatial grid format"""
    B, N_plus_1, D = tokens.shape
    patch_tokens = tokens[:, 1:, :]  # remove CLS
    N = patch_tokens.shape[1]

    # Smallest square that can hold N
    side = int(np.ceil(np.sqrt(N)))
    grid = side * side

    if N < grid:  # pad missing tokens
        pad_len = grid - N
        pad = torch.zeros(B, pad_len, D,
                          dtype=patch_tokens.dtype,
                          device=patch_tokens.device)
        grid_tokens = torch.cat([patch_tokens, pad], dim=1)
    else:  # drop any extra (globals)
        grid_tokens = patch_tokens[:, :grid, :]

    return grid_tokens.reshape(B, side, side, D)  # (B, H, W, D)

def get_detection_property(detections, index, property_name, default=None):
    """
    Safely get a property from a detection object, handling both tuple-style and attribute-style access.
    
    Args:
        detections: The detections object from GroundedSAM
        index: The index of the detection to access
        property_name: The property name to access (e.g., 'mask', 'confidence', 'class_id')
        default: Default value to return if property can't be accessed
        
    Returns:
        The property value or default if not found
    """
    try:
        # First try attribute-style access (supervision.detection.Detections object)
        if hasattr(detections, property_name):
            prop = getattr(detections, property_name)
            if prop is not None and index < len(prop):
                return prop[index]
                
        # Try direct indexing if it's a list/tuple of objects
        elif isinstance(detections, (list, tuple)) and index < len(detections):
            # If it's an object with the attribute
            if hasattr(detections[index], property_name):
                return getattr(detections[index], property_name)
            # If it's a dict with the key
            elif isinstance(detections[index], dict) and property_name in detections[index]:
                return detections[index][property_name]
                
        # Try accessing specific properties in alternate ways (for different versions)
        if property_name == 'mask' and hasattr(detections, 'masks') and index < len(detections.masks):
            return detections.masks[index]
            
        if property_name == 'class_id' and hasattr(detections, 'class_ids') and index < len(detections.class_ids):
            return detections.class_ids[index]
            
    except Exception as e:
        print(f"[DEBUG] Error accessing {property_name} at index {index}: {e}")
        
    return default

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
    optimal_layer=40, # Retained for single mode, but also new params below
    pooling_strategy="top_k", # Existing parameter
    # NEW PARAMETERS (with backward-compatible defaults)
    extraction_mode="single",     # "single" or "multi_layer" or "preset"
    preset_name="object_focused", # For preset mode
    custom_layers=None,           # For multi_layer mode, default to None then handle
    custom_weights=None,          # For multi_layer mode, default to None then handle
    temperature=0.07,             # New temperature parameter for pooling
    top_k_ratio=0.1,              # Existing top_k_ratio, ensured it's a direct param
    attention_heads=8             # Existing attention_heads, ensured it's a direct param
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

    # Kept Helper Functions
    # The following helper functions are being moved to module level:
    # get_preset_configuration, extract_multi_layer_features, combine_layer_features
    # Their definitions will be removed from here and placed globally.

    # This is the existing top-k implementation, preserved as per subtask.
    def existing_top_k_implementation(masked_features, top_k_ratio=0.1):
        """
        Original top_k pooling implementation.
        """
        feature_norms = masked_features.norm(dim=1)
        k = max(1, int(top_k_ratio * len(feature_norms)))
        top_k_indices = torch.topk(feature_norms, k)[1]
        pooled = masked_features[top_k_indices].mean(dim=0, keepdim=True)
        return pooled

    # Implementation of pe_attention_pooling
    def pe_attention_pooling(masked_features, temperature=0.07, attention_heads=8):
        """
        Perception Encoder Attention Pooling.
        Pools features using multi-head attention mechanism.
        Note: attention_heads parameter is part of the signature, but this simplified
        implementation effectively uses a single head. A full multi-head implementation
        would involve reshaping K, Q, V and computing attention per head.
        """
        if masked_features.numel() == 0:
            # If input is empty, return zero tensor of appropriate shape (1, D)
            # Assuming D is the last dimension of masked_features if it wasn't empty.
            # This requires knowing D. If masked_features can be (0, D), then shape[-1] is D.
            # If masked_features is just (0), D is unknown. Default to a common size or error.
            # For now, let's assume if N=0, D is still accessible from masked_features.shape[-1]
            # However, if masked_features is truly empty (e.g. torch.empty(0)), shape is (0,).
            # A robust way is to expect D as an argument or ensure masked_features is (0,D) not (0,).
            # Given the context, masked_features is likely (N_pixels_in_mask, D_feat_actual).
            # If N_pixels_in_mask is 0, shape is (0, D_feat_actual).
            if masked_features.ndim > 1 and masked_features.shape[-1] > 0:
                D = masked_features.shape[-1]
                return torch.zeros((1, D), device=masked_features.device, dtype=masked_features.dtype)
            else: # Cannot determine D, or truly empty.
                print("[WARN] pe_attention_pooling received empty features with undetermined D. Returning empty tensor.")
                return torch.empty(0, device=masked_features.device, dtype=masked_features.dtype)


        if masked_features.ndim == 2: # (N, D)
            masked_features = masked_features.unsqueeze(0) # (1, N, D)
        
        B, N, D = masked_features.shape
        if N == 0: # Should be caught by numel() check, but as safeguard.
             return torch.zeros((B, D), device=masked_features.device, dtype=masked_features.dtype).unsqueeze(0)


        # Simplified MultiHeadAttention. `attention_heads` is available for future enhancement.
        qkv_layer = torch.nn.Linear(D, D * 3, device=masked_features.device)
        
        qkv = qkv_layer(masked_features) # (B, N, 3*D)
        q, k, v = torch.chunk(qkv, 3, dim=-1) # (B, N, D) each
        
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D) # (B, N, N)
        attention_weights = torch.nn.functional.softmax(attention_scores / temperature, dim=-1) # (B, N, N)
        
        attended_features = torch.matmul(attention_weights, v) # (B, N, D)
        
        pooled = attended_features.mean(dim=1) # (B, D)
        return pooled.unsqueeze(0) # Consistent (1, B, D) or (1, 1, D) output shape

    # Implementation of pe_spatial_pooling
    def pe_spatial_pooling(masked_features, temperature=0.07):
        """
        Perception Encoder Spatial Pooling.
        Weights features by their spatial relevance (e.g., norm) and applies softmax.
        """
        if masked_features.numel() == 0:
            if masked_features.ndim > 1 and masked_features.shape[-1] > 0:
                D = masked_features.shape[-1]
                return torch.zeros((1, D), device=masked_features.device, dtype=masked_features.dtype)
            else:
                print("[WARN] pe_spatial_pooling received empty features with undetermined D. Returning empty tensor.")
                return torch.empty(0, device=masked_features.device, dtype=masked_features.dtype)

        if masked_features.ndim == 2: # (N, D)
            masked_features = masked_features.unsqueeze(0) # (1, N, D)
            
        B, N, D = masked_features.shape
        if N == 0:
            return torch.zeros((B, D), device=masked_features.device, dtype=masked_features.dtype).unsqueeze(0)

        feature_norms = masked_features.norm(dim=-1) # (B, N)
        spatial_weights = torch.nn.functional.softmax(feature_norms / temperature, dim=-1) # (B, N)
        
        pooled = (masked_features * spatial_weights.unsqueeze(-1)).sum(dim=1) # (B, D)
        return pooled.unsqueeze(0)

    # Implementation of pe_semantic_pooling
    def pe_semantic_pooling(masked_features, temperature=0.07):
        """
        Perception Encoder Semantic Pooling.
        Focuses on semantic similarity by using feature dot products.
        """
        if masked_features.numel() == 0:
            if masked_features.ndim > 1 and masked_features.shape[-1] > 0:
                D = masked_features.shape[-1]
                return torch.zeros((1, D), device=masked_features.device, dtype=masked_features.dtype)
            else:
                print("[WARN] pe_semantic_pooling received empty features with undetermined D. Returning empty tensor.")
                return torch.empty(0, device=masked_features.device, dtype=masked_features.dtype)

        if masked_features.ndim == 2: # (N, D)
            masked_features = masked_features.unsqueeze(0) # (1, N, D)
            
        B, N, D = masked_features.shape
        if N == 0:
            return torch.zeros((B, D), device=masked_features.device, dtype=masked_features.dtype).unsqueeze(0)

        normalized_features = torch.nn.functional.normalize(masked_features, p=2, dim=-1)
        similarity_matrix = torch.matmul(normalized_features, normalized_features.transpose(-2, -1)) # (B, N, N)
        
        semantic_weights = torch.nn.functional.softmax(similarity_matrix.mean(dim=-1) / temperature, dim=-1) # (B, N)
        
        pooled = (masked_features * semantic_weights.unsqueeze(-1)).sum(dim=1) # (B, D)
        return pooled.unsqueeze(0)

    # Implementation of pe_adaptive_pooling
    def pe_adaptive_pooling(masked_features, temperature=0.07):
        """
        Perception Encoder Adaptive Pooling.
        Combines spatial and semantic weighting. A simple average for now.
        """
        if masked_features.numel() == 0: # Check if the tensor is empty
            # Try to return a zero tensor of the correct shape (1, D)
            if masked_features.ndim > 1 and masked_features.shape[-1] > 0: # Check if D is obtainable
                D = masked_features.shape[-1]
                return torch.zeros((1, D), device=masked_features.device, dtype=masked_features.dtype)
            else: # Fallback if D cannot be determined (e.g. masked_features was torch.empty(0))
                print("[WARN] pe_adaptive_pooling received empty features with undetermined D. Returning empty tensor.")
                return torch.empty(0, device=masked_features.device, dtype=masked_features.dtype)

        # Ensuring masked_features is (B, N, D)
        if masked_features.ndim == 2: # (N, D)
            masked_features = masked_features.unsqueeze(0) # (1, N, D)
        
        B, N, D = masked_features.shape

        if N == 0: # If there are no features to pool (e.g. mask was empty)
            return torch.zeros((B, D), device=masked_features.device, dtype=masked_features.dtype).unsqueeze(0)

        # Spatial weights
        feature_norms = masked_features.norm(dim=-1, keepdim=True) # (B, N, 1)
        spatial_weights = torch.nn.functional.softmax(feature_norms / temperature, dim=1) # (B, N, 1)
        
        # Semantic proxy weights (using mean feature activation as a simple proxy)
        # Ensure mean is taken over D dimension, result should be (B,N,1) for broadcasting
        semantic_proxy_activations = masked_features.mean(dim=-1, keepdim=True) # (B, N, 1)
        semantic_proxy_weights = torch.nn.functional.softmax(semantic_proxy_activations / temperature, dim=1) # (B, N, 1)
        
        adaptive_weights = (spatial_weights + semantic_proxy_weights) / 2.0 # (B, N, 1)
        
        pooled = (masked_features * adaptive_weights).sum(dim=1) # (B, D)
        return pooled.unsqueeze(0) # Consistent (1, B, D) or (1, 1, D) output shape

    # This is the main dispatcher function as per the subtask.
    # Its signature and functionality are confirmed to match the requirements.
    def apply_spatial_pooling(masked_features, strategy="top_k", top_k_ratio=0.1, 
                             temperature=0.07, attention_heads=8):
        """
        Apply different pooling strategies to masked features.
        Signature and dispatch logic updated as per requirements.
        """
        if masked_features.numel() == 0: # Handle empty input early
            # Attempt to return a zero tensor of shape (1,D)
            # This requires D to be known. If masked_features is (0,D), shape[-1] gives D.
            # If masked_features is truly empty (e.g. torch.empty(0)), D is unknown.
            if masked_features.ndim > 1 and masked_features.shape[-1] > 0:
                D = masked_features.shape[-1]
                return torch.zeros((1, D), device=masked_features.device, dtype=masked_features.dtype)
            else: # Fallback for truly empty or D-undetermined tensors
                print(f"[WARN] apply_spatial_pooling (strategy: {strategy}) received empty features with undetermined D. Returning empty tensor.")
                return torch.empty(0, device=masked_features.device, dtype=masked_features.dtype)


        if strategy == "max":
            if masked_features.ndim == 1:
                 masked_features = masked_features.unsqueeze(0)
            pooled = masked_features.max(dim=0, keepdim=True)[0]
        elif strategy == "top_k":
            pooled = existing_top_k_implementation(masked_features, top_k_ratio)
        elif strategy == "attention": # Original simple attention (remains for backward compatibility or specific use)
            if masked_features.ndim == 1:
                 masked_features = masked_features.unsqueeze(-1) if masked_features.ndim ==1 else masked_features
            # Empty check already done above
            feature_norms = masked_features.norm(dim=1)
            attention_weights = torch.softmax(feature_norms / temperature, dim=0)
            pooled = (masked_features * attention_weights.unsqueeze(1)).sum(dim=0, keepdim=True)
        elif strategy == "average":
            if masked_features.ndim == 1:
                 masked_features = masked_features.unsqueeze(0)
            pooled = masked_features.mean(dim=0, keepdim=True)
        # PE-optimized strategies
        elif strategy == "pe_attention":
            pooled = pe_attention_pooling(masked_features, temperature, attention_heads)
        elif strategy == "pe_spatial":
            pooled = pe_spatial_pooling(masked_features, temperature)
        elif strategy == "pe_semantic":
            pooled = pe_semantic_pooling(masked_features, temperature)
        elif strategy == "pe_adaptive":
            pooled = pe_adaptive_pooling(masked_features, temperature)
        else:  # Fallback to average
            print(f"[WARN] Unknown pooling strategy '{strategy}', defaulting to average pooling.")
            if masked_features.ndim == 1:
                 masked_features = masked_features.unsqueeze(0)
            pooled = masked_features.mean(dim=0, keepdim=True)
        
        # Ensure consistent output shape, e.g., (1, D)
        if pooled.ndim == 1:
            pooled = pooled.unsqueeze(0)
        elif pooled.ndim > 2 : # e.g. if pooling returned (1,1,D)
            pooled = pooled.squeeze(0) # Aim for (1,D) or (D) then unsqueeze
            if pooled.ndim == 1: pooled = pooled.unsqueeze(0)

        return pooled

    def get_detected_class(detections_obj, det_index, class_names_from_ontology):
        # detections_obj is the Detections object from GroundedSAM
        # det_index is the index of the current detection
        # class_names_from_ontology is the list like ['person', 'rocks'] from ontology.classes()

        if hasattr(detections_obj, 'class_id') and \
           detections_obj.class_id is not None and \
           len(detections_obj.class_id) > det_index:
            
            class_id_val = detections_obj.class_id[det_index]

            if isinstance(class_id_val, torch.Tensor):
                class_id_val = class_id_val.item()

            # Corrected type check to include numpy integers
            if isinstance(class_id_val, (int, np.integer)) and 0 <= class_id_val < len(class_names_from_ontology):
                # This is the primary, expected path
                detected_class_name = class_names_from_ontology[class_id_val]
                print(f"[DEBUG get_class] Using class_id {class_id_val} (type {type(class_id_val)}) to get name '{detected_class_name}' from ontology classes list.")
                return detected_class_name
            else:
                print(f"[DEBUG get_class] class_id_val {class_id_val} (type {type(class_id_val)}) is not a valid index for ontology classes list (len {len(class_names_from_ontology)}).")
        else:
            print(f"[DEBUG get_class] detections_obj.class_id not found or invalid for index {det_index}.")

        # Fallback if the above fails (should ideally not be reached if GroundedSAM and ontology work as expected)
        # If there's only one class in the ontology, it's a safe bet. Otherwise, it's a guess.
        default_class = class_names_from_ontology[0] if class_names_from_ontology else "unknown"
        print(f"[DEBUG get_class] Fallback: returning '{default_class}'")
        return default_class


    def cleanup_temp_resources(temp_file_paths_list=None):
        """Clean up temporary resources, including GPU/MPS memory if applicable."""
        if temp_file_paths_list is None:
            temp_file_paths_list = []
        for temp_file in temp_file_paths_list:
            if temp_file and os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                    print(f"[Cleanup] Removed temp file: {temp_file}")
                except Exception as e_file_remove:
                    print(f"[WARN Cleanup] Failed to remove temp file {temp_file}: {e_file_remove}")
        
        # Explicitly call garbage collector
        gc.collect()
        print(f"[Cleanup] Garbage collection called.")
        
        # Clear CUDA cache if available and in use
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"[Cleanup] Cleared CUDA cache.")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            try:
                torch.mps.empty_cache() # For MPS devices
                print(f"[Cleanup] Cleared MPS cache.")
            except Exception as e_mps_cache:
                print(f"[WARN Cleanup] Error clearing MPS cache: {e_mps_cache}")
    
    # Access global models if needed
    global pe_model, pe_vit_model, preprocess, device
    
    pe_model_to_use = pe_model_param if pe_model_param is not None else pe_model
    pe_vit_model_to_use = pe_vit_model_param if pe_vit_model_param is not None else pe_vit_model
    preprocess_to_use = preprocess_param if preprocess_param is not None else preprocess
    device_to_use = device_param if device_param is not None else device

    # Handle default for custom_layers and custom_weights if None
    if custom_layers is None:
        custom_layers = [30,40,47]
    if custom_weights is None:
        custom_weights = [0.3,0.4,0.3]
    
    detections_obj = None 
    # embeddings, metadata_list, labels = [], [], [] # Initialized later
    final_embeddings_list, metadata_list, final_labels = [], [], [] # Use new names to avoid confusion before final assignment
    temp_image_path = None # Ensure temp_image_path is defined for finally

    try:
        print(f"[STATUS] Starting simplified region extraction process...")
        
        # Load and prepare image
        if is_url:
            pil_image = download_image(image_source)
        else:
            if isinstance(image_source, str):
                print(f"[STATUS] Loading image from path: {os.path.basename(image_source)}")
                pil_image = load_local_image(image_source)
            else:
                pil_image = image_source # Assuming it's already a PIL image

        # Resize image if needed for optimal processing
        original_size = pil_image.size
        if max(pil_image.size) > max_image_size:
            pil_image.thumbnail((max_image_size, max_image_size), Image.Resampling.LANCZOS)
            print(f"[STATUS] Resized image from {original_size} to {pil_image.size}")

        image_np = np.array(pil_image)
        image_area = image_np.shape[0] * image_np.shape[1]

        # Create temporary file for GroundedSAM
        temp_dir = tempfile.gettempdir()
        unique_id = str(uuid.uuid4())[:8]
        
        if isinstance(image_source, str) and os.path.isfile(image_source):
            base_filename = os.path.basename(image_source)
        else:
            base_filename = f"uploaded_image_{unique_id}.jpg" # Handle PIL image inputs
            
        temp_image_path = os.path.join(temp_dir, f"gsam_temp_{unique_id}_{base_filename}")
        pil_image.save(temp_image_path, "JPEG", quality=95)
        print(f"[STATUS] Saved temporary image for GroundedSAM: {temp_image_path}")


        prompts = [p.strip() for p in text_prompt.split('.') if p.strip()]
        if not prompts:
            prompts = ["object", "region"] # Default prompts if none provided
            print(f"[WARN] No text prompt provided. Using default prompts: {prompts}")

        print(f"[DEBUG_PROMPT] Initial text_prompt string: '{text_prompt}'")
        print(f"[DEBUG_PROMPT] Derived prompts list: {prompts}")

        # Improved prompt formatting for GroundedSAM
        # Based on official documentation: period-separated format works best for multiple objects
        # Natural language descriptions work well for single descriptive phrases
        if len(prompts) == 1 and len(prompts[0].split()) > 2:
            # Single descriptive phrase - keep natural language format
            formatted_prompt = prompts[0]
            print(f"[DEBUG_PROMPT] Using natural language format: '{formatted_prompt}'")
        else:
            # Multiple objects or simple terms - use period-separated format
            # Since input is already period-separated, just ensure proper formatting
            formatted_prompt = " . ".join(prompts) + " ."
            print(f"[DEBUG_PROMPT] Using period-separated format: '{formatted_prompt}'")

        ontology_dict = {prompt: prompt for prompt in prompts}
        print(f"[DEBUG_PROMPT] Created ontology_dict: {ontology_dict}")
        
        from autodistill_grounded_sam import GroundedSAM # Ensure this import is here
        from autodistill.detection import CaptionOntology
        
        ontology = CaptionOntology(ontology_dict)
        print(f"[DEBUG_PROMPT] Created ontology with classes: {ontology.classes()}")
        
        grounded_sam = GroundedSAM(
            ontology=ontology,
            box_threshold=0.35,
            text_threshold=0.25
        )
        print(f"[STATUS] Initialized GroundedSAM with box_threshold=0.35, text_threshold=0.25")

        # Use the formatted prompt for detection
        detections = grounded_sam.predict(temp_image_path)
        print(f"[STATUS] GroundedSAM detected {len(detections)} regions using prompt: '{formatted_prompt}'")
        
        # Get the class names list directly from the ontology object used by GroundedSAM
        ontology_class_names = ontology.classes()
        print(f"[DEBUG_PROMPT] ontology.classes() returned: {ontology_class_names}")
        print(f"[DEBUG_PROMPT] Type of ontology_class_names: {type(ontology_class_names)}")
        if isinstance(ontology_class_names, list) and ontology_class_names:
            print(f"[DEBUG_PROMPT] Type of first element in ontology_class_names: {type(ontology_class_names[0])}")


        if len(detections) == 0:
            print(f"[STATUS] No detections found by GroundedSAM.")
            # Call cleanup even if no detections, to remove temp image
            cleanup_temp_resources(temp_file_paths_list=[temp_image_path])
            return image_np if image_np is not None else None, [], [], [], [], None


        # Process detections (simplified)
        masks_cpu_np, metadata_list, labels = [], [], [] # Store CPU/NumPy masks for PE
        
        for i in range(min(len(detections), max_regions)):
            try:
                mask_tensor = detections.mask[i]
                raw_confidence = 0.0 # Default

                # --- Start of detailed logging for this detection ---
                print(f"\\n--- [DEBUG_DETECTION {i}] ---")
                
                # Log class_id
                if hasattr(detections, 'class_id') and detections.class_id is not None and len(detections.class_id) > i:
                    raw_class_id = detections.class_id[i]
                    print(f"[DEBUG_DETECTION {i}] Raw detections.class_id[{i}]: {raw_class_id} (Type: {type(raw_class_id)})")
                    if isinstance(raw_class_id, torch.Tensor):
                        print(f"[DEBUG_DETECTION {i}] Raw class_id tensor details: device={raw_class_id.device}, dtype={raw_class_id.dtype}")
                else:
                    print(f"[DEBUG_DETECTION {i}] detections.class_id not available or index out of bounds.")

                # Log confidence
                if hasattr(detections, 'confidence') and detections.confidence is not None and len(detections.confidence) > i:
                    confidence_val = detections.confidence[i]
                    print(f"[DEBUG_DETECTION {i}] Raw detections.confidence[{i}]: {confidence_val} (Type: {type(confidence_val)})")
                    if isinstance(confidence_val, torch.Tensor):
                        raw_confidence = confidence_val.item()
                    elif isinstance(confidence_val, np.ndarray):
                        raw_confidence = float(confidence_val)
                    else:
                        raw_confidence = float(confidence_val)
                else:
                     print(f"[DEBUG_DETECTION {i}] detections.confidence not available or index out of bounds.")


                # Log other potentially useful attributes from detections object
                if hasattr(detections, 'data') and isinstance(detections.data, dict):
                    print(f"[DEBUG_DETECTION {i}] detections.data keys: {list(detections.data.keys())}")
                    for key, value_list in detections.data.items():
                        if isinstance(value_list, (list, np.ndarray)) and len(value_list) > i:
                            item_value = value_list[i]
                            print(f"[DEBUG_DETECTION {i}] detections.data['{key}'][{i}]: {item_value} (Type: {type(item_value)})")
                        elif not isinstance(value_list, (list, np.ndarray)):
                             print(f"[DEBUG_DETECTION {i}] detections.data['{key}'] is not a list/array: {value_list} (Type: {type(value_list)})")


                # Call get_detected_class (which also has internal prints)
                detected_class = get_detected_class(detections, i, ontology_class_names)
                print(f"[DEBUG_DETECTION {i}] Class determined by get_detected_class: '{detected_class}'")
                # --- End of detailed logging for this detection ---

                print(f"[STATUS] Processing detection {i+1}/{min(len(detections), max_regions)}: {detected_class} (Raw Conf: {raw_confidence:.3f})")
                
                # Simplified Validation (Area and non-empty)
                if isinstance(mask_tensor, torch.Tensor):
                    mask_np_cpu = mask_tensor.detach().cpu().numpy() # Ensure mask is on CPU and NumPy
                elif isinstance(mask_tensor, np.ndarray):
                    mask_np_cpu = mask_tensor # Already a NumPy array
                else:
                    print(f"[WARN] Mask tensor for detection {i} is of unexpected type: {type(mask_tensor)}, attempting to convert.")
                    try:
                        mask_np_cpu = np.array(mask_tensor) # Fallback conversion
                    except Exception as e_conv:
                        print(f"[ERROR] Could not convert mask_tensor to NumPy array for detection {i}: {e_conv}")
                        continue # Skip this detection if conversion fails
                
                # Binarize if float, ensure uint8
                if np.issubdtype(mask_np_cpu.dtype, np.floating):
                    mask_processed_cv = (mask_np_cpu > 0.5).astype(np.uint8)
                elif mask_np_cpu.dtype == bool:
                    mask_processed_cv = mask_np_cpu.astype(np.uint8)
                else:
                    mask_processed_cv = mask_np_cpu.astype(np.uint8)

                if np.sum(mask_processed_cv) == 0:
                    print(f"[STATUS] Skipping detection {i+1} ({detected_class}) - Empty mask after processing.")
                    continue

                current_area_pixels = np.sum(mask_processed_cv)
                current_area_ratio = current_area_pixels / image_area if image_area > 0 else 0
                
                # Apply min_area_ratio filter (parameter from function call)
                if min_area_ratio > 0 and current_area_ratio < min_area_ratio:
                    print(f"[STATUS] Skipping detection {i+1} ({detected_class}) - Area ratio {current_area_ratio:.4f} < min_area_ratio {min_area_ratio:.4f}")
                    continue
                
                # Store the processed CPU/NumPy mask for PE
                masks_cpu_np.append(mask_processed_cv) 

                y_indices, x_indices = np.where(mask_processed_cv)
                x_min, x_max = int(x_indices.min()), int(x_indices.max())
                y_min, y_max = int(y_indices.min()), int(y_indices.max())
                
                metadata = {
                    "region_id": str(uuid.uuid4()),
                    "image_source": str(image_source) if isinstance(image_source, str) else "uploaded_image",
                    "filename": base_filename,
                    "bbox": [x_min, y_min, x_max, y_max],
                    "area_ratio": float(current_area_ratio),
                    "detected_class": detected_class,
                    "confidence": float(raw_confidence), # Using raw confidence
                    "type": "region",
                    "detection_method": "grounded_sam_simplified",
                    "layer_used": optimal_layer
                }
                label_text = f"{detected_class} ({raw_confidence:.2f})"
                metadata_list.append(metadata)
                labels.append(label_text)
                print(f"[STATUS] Successfully processed detection {i+1}: {label_text}")

            except Exception as e_det_loop:
                print(f"[ERROR] Error processing detection {i} in simplified loop: {e_det_loop}")
                import traceback
                traceback.print_exc()
                continue
        
        if not masks_cpu_np: # Check if any valid masks were collected
            print(f"[WARNING] No valid regions found after simplified processing and filtering.")
            cleanup_temp_resources(temp_file_paths_list=[temp_image_path])
            return image_np if image_np is not None else None, [], [], [], [], "No valid regions after filtering." # Added message
            
        print(f"[STATUS] Processing {len(masks_cpu_np)} detected regions with PE using mode: '{extraction_mode}'...")
        
        # Determine layers and weights if multi-layer or preset mode
        layers_to_use = []
        weights_to_use = []
        current_pooling_strategy = pooling_strategy # Use the function's pooling_strategy by default

        if extraction_mode == "preset":
            config = get_preset_configuration(preset_name)
            layers_to_use = config["layers"]
            weights_to_use = config["weights"]
            current_pooling_strategy = config["pooling"]  # Override pooling for preset
            print(f"[INFO] Using preset '{preset_name}': layers={layers_to_use}, weights={weights_to_use}, pooling='{current_pooling_strategy}'")
        elif extraction_mode == "multi_layer":
            layers_to_use = custom_layers
            weights_to_use = custom_weights
            print(f"[INFO] Using custom multi-layer: layers={layers_to_use}, weights={weights_to_use}, pooling='{current_pooling_strategy}'")
        else: # single_layer (default)
            print(f"[INFO] Using single layer: {optimal_layer}, pooling='{current_pooling_strategy}'")

        # Preprocess the input image once for PE
        # pe_input = preprocess_to_use(pil_image).unsqueeze(0).to(device_to_use) # Already done if reusing intermediate_features

        # Extract base image-level features (before region masking and pooling)
        raw_image_level_features = None # This will hold [1, N, D] or [1, C, H, W] etc.
        
        with torch.no_grad(): # Ensure no_grad context for all feature extraction
            pe_input_for_vit = preprocess_to_use(pil_image).unsqueeze(0).to(device_to_use)

            if extraction_mode == "single":
                if pe_vit_model_to_use is not None:
                    raw_image_level_features = pe_vit_model_to_use.forward_features(pe_input_for_vit, layer_idx=max(1, optimal_layer))
                    print(f"[STATUS] Extracted single-layer features from ViT layer {max(1, optimal_layer)}")
                elif pe_model_to_use is not None: # Fallback to general PE model (e.g., CLIP)
                    raw_image_level_features = pe_model_to_use.encode_image(pe_input_for_vit) # Usually [1, D_emb]
                    print(f"[STATUS] Extracted single-layer features using general PE model")
                else:
                    cleanup_temp_resources([temp_image_path])
                    return None, [], [], [], [], "No suitable PE model found for feature extraction."
            else: # "preset" or "multi_layer"
                if pe_vit_model_to_use is None:
                    cleanup_temp_resources([temp_image_path])
                    return None, [], [], [], [], "Vision Transformer model (pe_vit_model) required for multi-layer/preset modes."
                
                all_layers_features_dict = extract_multi_layer_features(pil_image, pe_vit_model_to_use, preprocess_to_use, layers_to_use, device_to_use)
                
                if not all_layers_features_dict or all(v is None for v in all_layers_features_dict.values()):
                    cleanup_temp_resources([temp_image_path])
                    return None, [], [], [], [], "Multi-layer feature extraction failed."

                # Align features with weights for combination
                ordered_feature_tensors = []
                temp_weights = []
                for i, layer_idx in enumerate(layers_to_use):
                    if layer_idx in all_layers_features_dict and all_layers_features_dict[layer_idx] is not None:
                        ordered_feature_tensors.append(all_layers_features_dict[layer_idx])
                        temp_weights.append(weights_to_use[i]) # Assumes weights_to_use corresponds to layers_to_use
                    else:
                        print(f"[WARN] Feature for layer {layer_idx} not found or is None. It will be excluded from combination.")

                if not ordered_feature_tensors:
                     cleanup_temp_resources([temp_image_path])
                     return None, [], [], [], [], "No valid features to combine from multi-layer/preset extraction."
                
                # Create a temporary dict for combine_layer_features if it strictly expects a dict
                # where keys don't matter but values (tensors) are in order of weights.
                # Or, adapt combine_layer_features to take list of tensors.
                # For now, assuming combine_layer_features can handle dict keys that might not be contiguous
                # if it aligns features with weights based on the order of `layers_to_use`.
                # The safest is to pass an ordered list of features and corresponding weights.
                # Let's assume combine_layer_features is called with the original dict and list of weights,
                # and it handles the alignment internally based on keys from layers_to_use.
                # This was how `combine_layer_features` was designed in the previous step (takes dict, list).
                # To ensure correct weight mapping for `combine_layer_features` as implemented:
                # The `combine_layer_features` iterates `layer_features_dict.values()`.
                # We must pass a dict whose `values()` are in the order of `temp_weights`.
                temp_dict_for_combine = {i: tensor for i, tensor in enumerate(ordered_feature_tensors)}
                raw_image_level_features = combine_layer_features(temp_dict_for_combine, temp_weights, temperature)


            if raw_image_level_features is None:
                cleanup_temp_resources([temp_image_path])
                return None, [], [], [], [], "Image-level feature extraction failed."

            # Convert raw_image_level_features to spatial grid: [1, H_feat, W_feat, D_feat_actual]
            # This part needs to be robust to various shapes from feature extractors.
            # D_feat_actual will be determined by the model's output.
            if raw_image_level_features.ndim == 3: # [B, N, D] - typical for ViT layers
                # Use pe_model_to_use (CLIP model) for tokens_to_grid, as it defines grid structure
                image_level_spatial_features = tokens_to_grid(raw_image_level_features, pe_model_to_use)
            elif raw_image_level_features.ndim == 4: # Potential [B, C, H, W] or [B, H, W, C]
                # If [B, C, H, W], C is D_feat. Permute to [B, H, W, D_feat]
                # This requires knowing D_feat. Let's assume D_feat is the last dim if not channel dim.
                # For now, assume if 4D, it's already [B, H, W, D] from tokens_to_grid or similar PE structure.
                # This logic needs to be robust. For PE models, features might already be [B,H,W,D].
                # If pe_model_to_use.encode_image returned [B,D] and it was unsqueezed to [B,1,1,D]
                # then tokens_to_grid is not strictly needed but won't harm for 1x1 grid.
                # Let's assume if ndim == 4, it's already the target spatial format [B, H, W, D]
                image_level_spatial_features = raw_image_level_features
            elif raw_image_level_features.ndim == 2 and raw_image_level_features.shape[0] == 1: # [1, D_emb] from encode_image
                image_level_spatial_features = raw_image_level_features.unsqueeze(1).unsqueeze(1) # Becomes [1, 1, 1, D]
            else:
                cleanup_temp_resources([temp_image_path])
                return None, [], [], [], [], f"Unsupported shape for image-level features after extraction/combination: {raw_image_level_features.shape}"

            _B, H_feat, W_feat, D_feat_actual = image_level_spatial_features.shape
            print(f"[STATUS] Prepared image-level spatial features. Grid: {H_feat}x{W_feat}, Dim: {D_feat_actual}")
            
            image_level_spatial_features = image_level_spatial_features.to(device_to_use)
            # pe_model_to_use should already be on device_to_use

            # Loop for masking and pooling
            # final_embeddings_list, temp_metadata, temp_labels, final_masks_for_output already initialized
            # metadata_list and labels are from GroundedSAM loop, need to align
            
            processed_embeddings_count = 0
            for i, mask_cv_np in enumerate(masks_cpu_np): # masks_cpu_np contains original resolution masks
                try:
                    mask_resized_np = cv2.resize(mask_cv_np, (W_feat, H_feat), interpolation=cv2.INTER_NEAREST).astype(bool)
                    if np.sum(mask_resized_np) == 0:
                        print(f"[WARNING] Mask {i} for '{labels[i]}' is empty after resizing, skipping.")
                        continue
                    
                    # Apply mask to image-level spatial features
                    masked_features_for_pooling = image_level_spatial_features[0, mask_resized_np, :] # Results in [N_pixels_in_mask, D_feat_actual]
                    
                    if masked_features_for_pooling.shape[0] == 0:
                        print(f"[WARNING] No features extracted for mask {i} ('{labels[i]}') after spatial selection, skipping.")
                        continue

                    # Apply chosen pooling strategy
                    pooled_embedding = apply_spatial_pooling(
                        masked_features_for_pooling,
                        strategy=current_pooling_strategy,
                        top_k_ratio=top_k_ratio, # Direct param from signature
                        temperature=temperature,   # Direct param from signature
                        attention_heads=attention_heads # Direct param from signature
                    )
                    print(f"[STATUS] Applied '{current_pooling_strategy}' pooling to {masked_features_for_pooling.shape[0]} features for mask '{labels[i]}'")
                    
                    pooled_embedding = pooled_embedding.to(device_to_use) # Ensure it's on device for projection

                    # Final projection (same as before)
                    final_embedding = None
                    # Projection logic depends on whether features are from ViT intermediate or final CLIP
                    # If extraction_mode involved pe_vit_model_to_use or if single layer came from ViT
                    is_vit_features = (extraction_mode != "single") or \
                                      (extraction_mode == "single" and pe_vit_model_to_use is not None and hasattr(pe_vit_model_to_use, 'forward_features'))

                    if is_vit_features and hasattr(pe_model_to_use, 'visual') and \
                       hasattr(pe_model_to_use.visual, 'ln_post') and hasattr(pe_model_to_use.visual, 'proj'):
                        if pe_model_to_use.visual.ln_post.weight.dtype == torch.float32 and pooled_embedding.dtype != torch.float32:
                            pooled_embedding = pooled_embedding.float()
                        normed_features = pe_model_to_use.visual.ln_post(pooled_embedding)
                        projected_embedding = normed_features @ pe_model_to_use.visual.proj
                        final_embedding = projected_embedding[0]
                    elif hasattr(pe_model_to_use, 'proj') and pe_model_to_use.proj is not None and is_vit_features: # Fallback for some CLIPs with ViT features
                         projected_embedding = pooled_embedding @ pe_model_to_use.proj
                         final_embedding = projected_embedding[0]
                    elif not is_vit_features : # Features are from pe_model_to_use.encode_image() or similar final stage
                        final_embedding = pooled_embedding[0]
                    else: # ViT features used, but no standard projection head found
                        print(f"[WARN] ViT-like features used, but no standard projection found. Using pooled features directly for mask '{labels[i]}'.")
                        final_embedding = pooled_embedding[0]


                    if final_embedding is not None and final_embedding.norm().item() > 1e-6:
                        final_embedding = final_embedding / final_embedding.norm(dim=-1, keepdim=True)
                    
                    if final_embedding is not None:
                        final_embedding_np_check = final_embedding.detach().cpu().numpy()
                        if np.isnan(final_embedding_np_check).any() or np.isinf(final_embedding_np_check).any():
                            print(f"[WARN] Embedding for mask '{labels[i]}' contains NaN/inf, skipping.")
                            continue
                        
                        final_embeddings_list.append(final_embedding.detach().cpu())
                        # Keep corresponding metadata and label from the GroundedSAM detection loop
                        # metadata_list and labels were populated in sync with masks_cpu_np
                        # No, metadata_list and labels from outer scope are for ALL initial detections.
                        # We need to append to new lists, based on current 'i'
                        # The metadata_list and labels from the GroundedSAM part are the ones to use.
                        # So, we append to final_metadata, final_labels, final_masks_for_output
                        # The `metadata_list` and `labels` from the detection loop should be used here.
                        # I'll rename them to `initial_metadata_list` and `initial_labels` for clarity.
                        # And then append to the final lists here.
                        # This was not correctly handled in the original code snippet for the loop.
                        # The `metadata_list` and `labels` are from the GroundedSAM part.
                        # `temp_metadata.append(metadata_list[i])` was how it was done.
                        # So, `final_metadata_list.append(metadata_list[i])` and `final_labels.append(labels[i])`
                        # `final_masks_for_output.append(mask_cv_np)`
                        
                        # These lists are populated in the GroundedSAM detection loop.
                        # We need to ensure that the embeddings generated here correspond to the correct entries
                        # in `metadata_list` and `labels` that were populated alongside `masks_cpu_np`.
                        # The current index `i` corresponds to `masks_cpu_np[i]`, `metadata_list[i]`, `labels[i]`.
                        # So, we should append those items to our final lists.
                        # The placeholder code had `temp_metadata.append(metadata_list[i])` which is correct conceptually.
                        
                        # Let's re-initialize final_masks_for_output, final_metadata_list, final_labels before this loop
                        # and append to them here.
                        # This means the `embeddings` list in `return` should be `final_embeddings_list`.
                        # The `metadata` list in `return` should be `final_metadata_list_for_output`.
                        # The `labels` list in `return` should be `final_labels_for_output`.
                        # The `masks` list in `return` should be `final_masks_for_output_list`.
                        # This part of the prompt seems to have simplified the return handling.
                        # The original loop was:
                        # temp_embeddings, temp_metadata, temp_labels, final_masks_for_output
                        # I will use these names for clarity.
                        # So, before this loop:
                        # temp_embeddings, temp_metadata, temp_labels, final_masks_for_output = [], [], [], []
                        # Inside this loop:
                        # temp_embeddings.append(final_embedding.detach().cpu())
                        # temp_metadata.append(metadata_list[i]) # metadata_list from GSAM loop
                        # temp_labels.append(labels[i])         # labels from GSAM loop
                        # final_masks_for_output.append(mask_cv_np)
                        # This is what I'll implement.

                        # (This block is effectively what the old loop did, just that `final_embedding` calculation is more complex)
                        # The prompt has `region_embeddings.append(pooled_embedding)` which is not quite right as it misses projection.
                        # It should be:
                        # final_embeddings_list.append(final_embedding.detach().cpu())
                        # And then the related metadata for this successful embedding.
                        # The existing code already has temp_embeddings, temp_metadata, temp_labels, final_masks_for_output.
                        # I will reuse those variable names.
                        # The loop should iterate, and if an embedding is successfully created, append to these lists.
                        # The `final_embeddings_list` is what I called `temp_embeddings` before.
                        # So, I will use `temp_embeddings.append(final_embedding.detach().cpu())`
                        # and `temp_metadata.append(metadata_list[i])` etc.

                        # This part is fine as is, assuming temp_embeddings, temp_metadata etc. are correctly managed.
                        # The prompt is a bit confusing here with `region_embeddings.append(pooled_embedding)`
                        # vs. the full projection logic. I will follow the full projection.
                        
                        # The existing code structure is:
                        # temp_embeddings = [] ; temp_metadata = [] ; temp_labels = [] ; final_masks_for_output = []
                        # for i, mask_cv_np in enumerate(masks_cpu_np):
                        #    ... compute final_embedding ...
                        #    temp_embeddings.append(final_embedding.detach().cpu())
                        #    temp_metadata.append(metadata_list[i]) # From GSAM loop
                        #    temp_labels.append(labels[i]) # From GSAM loop
                        #    final_masks_for_output.append(mask_cv_np) # From GSAM loop
                        # Then after loop:
                        # embeddings = temp_embeddings
                        # metadata_list = temp_metadata (this was overwriting GSAM metadata_list, careful!)
                        # labels = temp_labels (overwriting GSAM labels)
                        # This needs to be handled carefully.
                        # Let's use new list names for the output:
                        # output_embeddings, output_metadata, output_labels, output_masks = [], [], [], []
                        # Inside loop:
                        # output_embeddings.append(...)
                        # output_metadata.append(metadata_list[i]) # GSAM metadata_list
                        # output_labels.append(labels[i])         # GSAM labels
                        # output_masks.append(mask_cv_np)
                        # Then, in return: image_np, output_masks, output_embeddings, output_metadata, output_labels, None

                        # Okay, let's stick to the variable names from the *existing successful code* for minimal confusion.
                        # Those were: temp_embeddings, temp_metadata, temp_labels, final_masks_for_output.
                        # And after the loop: embeddings = temp_embeddings, metadata_list = temp_metadata, labels = temp_labels.
                        # This implies that the `metadata_list` and `labels` returned by the function are only for regions
                        # that successfully yield an embedding, not all originally detected regions. This is reasonable.

                        # So, the lists `temp_embeddings`, `temp_metadata`, `temp_labels`, `final_masks_for_output`
                        # should be initialized before this loop.
                        
                        # This was the structure:
                        # temp_embeddings = []
                        # temp_metadata = [] (shadows outer scope metadata_list from GSAM)
                        # temp_labels = [] (shadows outer scope labels from GSAM)
                        # final_masks_for_output = []
                        # This is fine.

                        # The prompt's suggested loop starts with `region_embeddings = []`
                        # and appends `pooled_embedding`. This is too early (before projection).
                        # I will use `temp_embeddings` and append `final_embedding.detach().cpu()`.
                        
                        temp_embeddings.append(final_embedding.detach().cpu())
                        temp_metadata.append(metadata_list[i]) # metadata_list is from GSAM part
                        temp_labels.append(labels[i])         # labels is from GSAM part
                        final_masks_for_output.append(mask_cv_np) # current mask from masks_cpu_np
                        processed_embeddings_count +=1


                        print(f"[STATUS] Successfully generated embedding for mask {i} ('{labels[i]}')")
                    else:
                        print(f"[WARN] Final embedding for mask {i} ('{labels[i]}') was None, skipping.")
                        
                except Exception as e_emb_loop:
                    print(f"[ERROR] Error generating embedding for mask {i} ('{labels[i]}'): {e_emb_loop}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            # This was the original assignment after the loop.
            # embeddings = temp_embeddings
            # metadata_list = temp_metadata # This overwrites the original metadata_list from GroundSAM
            # labels = temp_labels         # This overwrites the original labels from GroundSAM
            # This is the correct behavior: the returned metadata and labels should correspond to the returned embeddings.

            if not temp_embeddings: # Check if any embeddings were generated
                print(f"[WARNING] No embeddings generated after PE processing for any region.")
                cleanup_temp_resources(temp_file_paths_list=[temp_image_path])
                return image_np if image_np is not None else None, [], [], [], [], "No embeddings generated."

            print(f"[SUCCESS] Successfully extracted {len(temp_embeddings)} region embeddings.")
            cleanup_temp_resources(temp_file_paths_list=[temp_image_path])
            # Return final_masks_for_output, temp_embeddings, temp_metadata, temp_labels
            return image_np, final_masks_for_output, temp_embeddings, temp_metadata, temp_labels, None

    except Exception as e:
        print(f"[CRITICAL ERROR] in extract_region_embeddings_autodistill: {e}")
        import traceback
        error_message = f"Critical error in simplified pipeline: {e}\\n{traceback.format_exc()}"
        print(error_message)
        # Ensure cleanup is called on any exception
        cleanup_temp_resources(temp_file_paths_list=[temp_image_path])
        return None, [], [], [], [], error_message
    
    finally:
        # General cleanup for any remaining temp files or resources not explicitly handled above
        # The main cleanup logic is now within cleanup_temp_resources, called before returns
        # This finally block can be used for any other last-minute checks if needed.
        # temp_image_path should already be cleaned by calls within try/except
        print(f"[FINALLY] extract_region_embeddings_autodistill finished.")
        # Removed ensure_consistent_dimensions call here

def extract_whole_image_embeddings(
    image_source,
    pe_model_param=None,
    pe_vit_model_param=None,
    preprocess_param=None,
    device_param=None,
    # Removed is_url as image_source can be URL or path, handled by load_local_image/download_image
    max_image_size=800,
    # NEW PARAMETERS (with backward-compatible defaults)
    extraction_mode="single",     # "single" or "multi_layer" or "preset"
    optimal_layer=40,            # For single mode (existing)
    preset_name="object_focused", # For preset mode
    custom_layers=None,           # For multi_layer mode (provide defaults)
    custom_weights=None,          # For multi_layer mode (provide defaults)
    temperature=0.07              # New temperature parameter (primarily for feature combination if applicable)
):
    """
    Extract embeddings for whole images using Perception Encoder.
    Supports single-layer, multi-layer, and preset-based feature extraction.
    """
    # Access global models if needed
    global pe_model, pe_vit_model, preprocess, device
    
    # Use provided models or fall back to globals
    pe_model_to_use = pe_model_param if pe_model_param is not None else pe_model
    pe_vit_model_to_use = pe_vit_model_param if pe_vit_model_param is not None else pe_vit_model
    preprocess_to_use = preprocess_param if preprocess_param is not None else preprocess
    device_to_use = device_param if device_param is not None else device

    # Handle default for custom_layers and custom_weights if None
    if custom_layers is None:
        custom_layers = [30,40,47] # Default custom layers
    if custom_weights is None:
        custom_weights = [0.3,0.4,0.3] # Default custom weights
    
    try:
        # Load the image (pil_image will be used for multi-layer, image_np for metadata)
        if isinstance(image_source, str):
            if image_source.startswith(('http://', 'https://')):
                pil_image = download_image(image_source)
            else:
                pil_image = load_local_image(image_source)
        else: # Assuming it's already a PIL image
            pil_image = image_source

        # Resize image
        original_size = pil_image.size
        if max(pil_image.size) > max_image_size:
            pil_image.thumbnail((max_image_size, max_image_size), Image.Resampling.LANCZOS)
            print(f"[STATUS] Resized image from {original_size} to {pil_image.size}")

        image_np = np.array(pil_image) # For metadata and potential return
        
        # Prepare preprocessed input for models
        pe_input = preprocess_to_use(pil_image).unsqueeze(0).to(device_to_use)

        # Determine layers and weights
        layers_to_use = []
        weights_to_use = []
        actual_layers_used_for_embedding = [] # For metadata

        if extraction_mode == "preset":
            config = get_preset_configuration(preset_name)
            layers_to_use = config["layers"]
            weights_to_use = config["weights"]
            # pooling_strategy = config.get("pooling") # Not used for whole image, but good to know
            print(f"[INFO] extract_whole_image_embeddings: Using preset '{preset_name}': layers={layers_to_use}, weights={weights_to_use}")
            actual_layers_used_for_embedding = layers_to_use
        elif extraction_mode == "multi_layer":
            layers_to_use = custom_layers
            weights_to_use = custom_weights
            print(f"[INFO] extract_whole_image_embeddings: Using custom multi-layer: layers={layers_to_use}, weights={weights_to_use}")
            actual_layers_used_for_embedding = layers_to_use
        else: # single_layer mode
            print(f"[INFO] extract_whole_image_embeddings: Using single layer: {optimal_layer}")
            actual_layers_used_for_embedding = [optimal_layer]

        # Extract features
        raw_features = None # This will hold the features before final processing (pooling, normalization)
        embedding_method_detail = ""

        with torch.no_grad():
            amp_context = nullcontext()
            if torch.backends.mps.is_available() and hasattr(torch.amp, 'autocast'):
                 amp_context = torch.amp.autocast(device_type='mps', dtype=torch.float16)

            with amp_context:
                if extraction_mode == "single":
                    safe_optimal_layer = max(1, optimal_layer)
                    if pe_vit_model_to_use is not None:
                        try:
                            # DINOv2/ViT forward_features usually returns a dict or list of features for specified layers
                            # For a single layer, it might be a list with one tensor, or dict {layer: tensor}
                            # Let's assume it returns features directly for the layer or a dict
                            output = pe_vit_model_to_use.forward_features(pe_input, layer_idx=safe_optimal_layer)
                            if isinstance(output, dict):
                                raw_features = output.get(safe_optimal_layer)
                                if raw_features is None: # Try common DINOv2 keys like 'x_norm_clstoken' or 'x_norm_patchtokens'
                                    raw_features = output.get('x_norm_clstoken', output.get('x_norm_patchtokens'))
                            else: # Assuming direct tensor output or list of tensors
                                raw_features = output[0] if isinstance(output, list) else output
                            embedding_method_detail = f"vit_forward_features_layer_{safe_optimal_layer}"
                            print(f"[STATUS] Extracted features from ViT layer {safe_optimal_layer}")
                        except Exception as e:
                            print(f"[WARN] Error using pe_vit_model.forward_features with layer {safe_optimal_layer}: {e}. Falling back.")
                            # Fallback to pe_model_to_use if ViT fails for single layer
                            if pe_model_to_use:
                                raw_features = pe_model_to_use.encode_image(pe_input)
                                embedding_method_detail = "pe_model_encode_image_fallback"
                                print(f"[STATUS] Used pe_model.encode_image as fallback.")
                            else:
                                raise ValueError("No PE model available for feature extraction.")
                    elif pe_model_to_use is not None: # Only general PE model available
                        raw_features = pe_model_to_use.encode_image(pe_input)
                        embedding_method_detail = "pe_model_encode_image"
                        print(f"[STATUS] Extracted features using general PE model's encode_image.")
                    else:
                        raise ValueError("No PE model available for single layer feature extraction.")
                
                else: # "preset" or "multi_layer"
                    if pe_vit_model_to_use is None:
                        raise ValueError("Vision Transformer model (pe_vit_model) required for multi-layer/preset modes.")
                    
                    # extract_multi_layer_features expects: (pil_image, pe_vit_model, preprocess_fn, layers, device)
                    all_layers_features_dict = extract_multi_layer_features(
                        pil_image, pe_vit_model_to_use, preprocess_to_use, layers_to_use, device_to_use
                    )
                    if not all_layers_features_dict or all(v is None for v in all_layers_features_dict.values()):
                        raise ValueError("Multi-layer feature extraction failed to yield any features.")

                    # Align features with weights for combination
                    ordered_feature_tensors = []
                    temp_weights_for_combine = []
                    valid_layers_in_combination = []
                    for i, layer_idx in enumerate(layers_to_use):
                        if layer_idx in all_layers_features_dict and all_layers_features_dict[layer_idx] is not None:
                            ordered_feature_tensors.append(all_layers_features_dict[layer_idx])
                            temp_weights_for_combine.append(weights_to_use[i])
                            valid_layers_in_combination.append(layer_idx)
                        else:
                            print(f"[WARN] Feature for layer {layer_idx} not found/None. Excluding from combination.")
                    
                    if not ordered_feature_tensors:
                        raise ValueError("No valid features to combine from multi-layer/preset extraction.")
                    
                    # The combine_layer_features expects a dict. Create one with ordered features.
                    temp_dict_for_combine = {idx: tensor for idx, tensor in enumerate(ordered_feature_tensors)}
                    raw_features = combine_layer_features(temp_dict_for_combine, temp_weights_for_combine, temperature)
                    embedding_method_detail = f"{extraction_mode}_combined_{len(valid_layers_in_combination)}_layers"
                    actual_layers_used_for_embedding = valid_layers_in_combination # Update metadata for actual layers used
                    print(f"[STATUS] Combined features from layers: {valid_layers_in_combination} using weights: {temp_weights_for_combine}")

                if raw_features is None:
                    raise ValueError("Feature extraction resulted in None.")

                # Post-process raw_features: Take CLS token or average pool patch tokens for ViTs
                # DINOv2 typically returns patch tokens + CLS token: [B, 1+N, D] or just patch tokens [B,N,D] or CLS [B,D]
                # If it's [B, 1+N, D], CLS is usually features[:, 0]
                # If it's [B, N, D] (patch tokens only), then mean pool: features.mean(dim=1)
                # If it's [B, D] (already CLS or from CLIP encode_image), use as is.
                
                final_embedding = None
                if raw_features.ndim == 3: # [B, N_tokens, D]
                    # Check if CLS token is present (e.g., N_tokens = num_patches + 1)
                    # This is heuristic. Some models might just return patch tokens.
                    # For DINOv2, if 'x_norm_clstoken' was extracted, it's already [B,D].
                    # If 'x_norm_patchtokens' was extracted, it's [B,N,D] and needs pooling.
                    # If a generic forward_features(layer_idx) was used, it's often all tokens from that layer.
                    if "clstoken" in embedding_method_detail.lower() or raw_features.shape[1] == 1 : # Already a CLS token or equivalent
                        final_embedding = raw_features[:, 0] # if shape [B,1,D] squeeze, else it's fine
                        if final_embedding.ndim == 1 and raw_features.shape[0] == 1: # if B=1, result is [D], unsqueeze
                            final_embedding = final_embedding.unsqueeze(0)
                        elif final_embedding.ndim == 2 and raw_features.shape[0] > 1 : # [B,D]
                             pass # Correct shape
                        print(f"[STATUS] Using CLS token or equivalent from raw_features shape {raw_features.shape}")
                    else: # Likely patch tokens or sequence of tokens, average pool them
                        final_embedding = raw_features.mean(dim=1)
                        print(f"[STATUS] Applied mean pooling over token dimension from raw_features shape {raw_features.shape}")
                elif raw_features.ndim == 2: # Already [B, D]
                    final_embedding = raw_features
                    print(f"[STATUS] Using raw_features as is (shape {raw_features.shape})")
                else:
                    raise ValueError(f"Unsupported raw_features shape: {raw_features.shape}")

                # Normalize the final embedding
                final_embedding = torch.nn.functional.normalize(final_embedding, dim=-1)
                print(f"[STATUS] Normalized final embedding, shape: {final_embedding.shape}")
        
        # Create metadata for the whole image
        image_id = str(uuid.uuid4())
        metadata = {
            "region_id": image_id, # Using image_id as region_id for consistency in Qdrant
            "image_source": str(image_source) if isinstance(image_source, str) else "uploaded_image",
            "filename": os.path.basename(str(image_source)) if isinstance(image_source, str) else "uploaded_image",
            "bbox": [0, 0, image_np.shape[1], image_np.shape[0]],
            "area_ratio": 1.0,
            "processing_type": "whole_image",
            "embedding_config": { # New field for detailed config
                "extraction_mode": extraction_mode,
                "layers_used": actual_layers_used_for_embedding,
                "weights_used": weights_to_use if extraction_mode != "single" else "N/A",
                "temperature": temperature,
                "embedding_method_detail": embedding_method_detail
            }
        }
        
        embedding_cpu = final_embedding.cpu().detach()
        
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
        
        print(f"[STATUS] Successfully extracted whole image embedding. Shape: {embedding_cpu.shape}, Method: {embedding_method_detail}")
        return image_np, embedding_cpu, metadata, "Whole Image" # image_np is the numpy version of pil_image
        
    except Exception as e:
        print(f"[ERROR] Error extracting whole image embedding: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

# =============================================================================
# DATABASE FUNCTIONS
# =============================================================================

def setup_qdrant(collection_name, vector_size, max_retries=5, retry_delay=3.0):
    """Setup Qdrant collection for storing image region embeddings with retry logic"""
    persist_path = "./image_retrieval_project/qdrant_data"
    os.makedirs(persist_path, exist_ok=True)
    
    # Try to connect with retries
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                print(f"Retry attempt {attempt}/{max_retries} connecting to Qdrant...")
                
            client = QdrantClient(path=persist_path)
            print(f"Successfully connected to local storage at {persist_path}")
            
            try:
                # Check collections
                collections = client.get_collections().collections
                collection_names = [collection.name for collection in collections]
                
                if collection_name not in collection_names:
                    # Create the collection with standard vector configuration
                    client.create_collection(
                        collection_name=collection_name,
                        vectors_config=models.VectorParams(
                            size=vector_size,
                            distance=models.Distance.COSINE
                        ),
                        # Ensure we have proper indexing for fast search
                        optimizers_config=models.OptimizersConfigDiff(
                            memmap_threshold=10000  # Use memmap for large collections
                        )
                    )
                    print(f"✅ Created new collection: {collection_name}")
                else:
                    print(f"✅ Using existing collection: {collection_name}")
                    try:
                        # Verify the collection has the correct vector size
                        collection_info = client.get_collection(collection_name=collection_name)
                        existing_vector_size = collection_info.config.params.vectors.size
                        if existing_vector_size != vector_size:
                            print(f"⚠️ Warning: Collection {collection_name} has vector size {existing_vector_size}, but requested size is {vector_size}")
                            print(f"This might cause issues when searching. Consider using a different collection name.")
                    except Exception as inner_e:
                        print(f"⚠️ Warning: Could not verify vector size: {inner_e}")
                
                return client
                
            except Exception as inner_e:
                print(f"Error working with collection: {inner_e}")
                client.close()
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                continue
                
        except RuntimeError as e:
            if "already accessed by another instance" in str(e):
                print(f"⚠️ Database is locked by another process. Waiting {retry_delay} seconds before retry {attempt+1}/{max_retries}...")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    print(f"❌ Failed to access database after {max_retries} attempts - database is locked!")
                    print(f"Please ensure no other instances of the application are running.")
                    return None
            else:
                print(f"❌ Error connecting to Qdrant: {e}")
                import traceback
                traceback.print_exc()
                return None
        except Exception as e:
            print(f"❌ Unexpected error in setup_qdrant: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    return None

def store_embeddings_in_qdrant(client, collection_name, embeddings, metadata_list):
    """Store embeddings in Qdrant collection"""
    # Check if client is valid
    if client is None:
        print(f"❌ Cannot store embeddings: Database client is not connected")
        return []
        
    if not embeddings or not metadata_list:
        print(f"❌ Nothing to store: Empty embeddings or metadata")
        return []
        
    try:
        print(f"[STATUS] Processing {len(embeddings)} embeddings for storage in {collection_name}...")
        
        # First, ensure all embeddings are in the correct format
        embedding_vectors = []
        for emb in embeddings:
            # Check if embedding is already a tensor, get numpy array
            if isinstance(emb, torch.Tensor):
                emb_numpy = emb.detach().cpu().numpy()
            else:
                emb_numpy = np.array(emb)
                
            # If we have a 1D array with shape (D,), reshape to (1, D)
            if len(emb_numpy.shape) == 1:
                emb_vector = emb_numpy
            # If we have a 2D array with shape (1, D), extract the inner vector
            elif len(emb_numpy.shape) == 2 and emb_numpy.shape[0] == 1:
                emb_vector = emb_numpy[0]
            else:
                # For any other shapes, flatten to ensure 1D
                emb_vector = emb_numpy.flatten()
                
            embedding_vectors.append(emb_vector)

        point_ids = [metadata["region_id"] for metadata in metadata_list]

        # Convert numpy types in metadata to Python native types
        sanitized_metadata = []
        for metadata in metadata_list:
            sanitized = {}
            for key, value in metadata.items():
                if isinstance(value, np.integer):
                    sanitized[key] = int(value)
                elif isinstance(value, np.floating):
                    sanitized[key] = float(value)
                elif isinstance(value, np.ndarray):
                    sanitized[key] = value.tolist()
                elif isinstance(value, list):
                    sanitized[key] = [int(x) if isinstance(x, np.integer) else
                                     float(x) if isinstance(x, np.floating) else x
                                     for x in value]
                else:
                    sanitized[key] = value
            sanitized_metadata.append(sanitized)

        # Prepare points for upsert
        points = []
        for i in range(len(embedding_vectors)):
            points.append(models.PointStruct(
                id=point_ids[i],
                vector=embedding_vectors[i],
                payload=sanitized_metadata[i]
            ))

        if not points:
            print(f"⚠️ Warning: No valid points to store after processing")
            return []

        # Upsert points in batches
        batch_size = 100
        stored_count = 0
        
        for i in range(0, len(points), batch_size):
            batch = points[i:i+batch_size]
            try:
                client.upsert(
                    collection_name=collection_name,
                    points=batch
                )
                stored_count += len(batch)
                print(f"[STATUS] Stored batch {i//batch_size + 1}/{(len(points)-1)//batch_size + 1} ({len(batch)} points)")
            except Exception as batch_error:
                print(f"❌ Error storing batch {i//batch_size + 1}: {batch_error}")
                continue

        print(f"✅ Successfully stored {stored_count}/{len(points)} embeddings in collection: {collection_name}")
        return point_ids
    except Exception as e:
        print(f"❌ Error in store_embeddings_in_qdrant: {e}")
        import traceback
        traceback.print_exc()
        return []

def visualize_detected_regions(image, masks, labels, title="Detected Regions", figsize=(10, 10)):
    """Create a visualization of detected regions in an image"""
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(image)

    # Draw each mask with a different color
    for i, mask in enumerate(masks):
        color = np.random.random(3)

        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        cv2.drawContours(image, contours, -1, color * 255, 2)

        M = cv2.moments(mask.astype(np.uint8))
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            label_text = labels[i] if i < len(labels) else f"Region {i+1}"
            ax.text(cx, cy, label_text, color='white',
                   bbox=dict(facecolor='black', alpha=0.7))

    ax.set_title(title)
    ax.axis('off')
    plt.tight_layout()

    return fig

# =============================================================================
# MAIN PROCESSING PIPELINE
# =============================================================================

def process_folder_for_region_database(
    folder_path,
    collection_name,
    text_prompts,
    pe_model,
    pe_vit_model, 
    preprocess,
    device,
    optimal_layer=40,
    min_area_ratio=0.01,
    max_regions=10,
    visualize_samples=True,
    sample_count=3,
    checkpoint_interval=10,
    resume_from_checkpoint=True,
    pooling_strategy="top_k"
):
    """Process all images in a folder and build a searchable database"""
    print(f"Starting to process folder: {folder_path}")
    print(f"Looking for objects matching: {text_prompts}")
    print(f"Using embedding layer: {optimal_layer}")
    print(f"Using pooling strategy: {pooling_strategy}")

    if not os.path.exists(folder_path):
        print(f"❌ Error: Folder not found at {folder_path}")
        return None

    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_paths = []

    for file in os.listdir(folder_path):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_paths.append(os.path.join(folder_path, file))

    if not image_paths:
        print(f"❌ Error: No images found in {folder_path}")
        return None

    print(f"📁 Found {len(image_paths)} images to process")

    # Initialize tracking variables
    processed_count = 0
    failed_count = 0
    skipped_count = 0
    total_regions = 0

    # Initialize database variables
    client = None
    vector_size = None
    initialization_complete = False
    
    # Include layer in collection name
    region_collection = f"{collection_name}_layer{optimal_layer}"
    print(f"📊 Using collection name with layer: {region_collection}")

    start_time = time.time()

    # Process images
    for idx, image_path in enumerate(image_paths):
        print(f"\nProcessing image {idx+1}/{len(image_paths)}: {os.path.basename(image_path)}")

        try:
            # Extract regions
            image, masks, embeddings, metadata, labels, error_message = extract_region_embeddings_autodistill(
                image_path,
                text_prompt=text_prompts,
                pe_model_param=pe_model,
                pe_vit_model_param=pe_vit_model,
                preprocess_param=preprocess,
                device_param=device,
                is_url=False,
                min_area_ratio=min_area_ratio,
                max_regions=max_regions,
                optimal_layer=optimal_layer,
                pooling_strategy=pooling_strategy
            )

            if image is not None and len(embeddings) > 0:
                print(f"✅ Found {len(embeddings)} valid regions in image {idx+1}")

                # Initialize database if not done yet
                if not initialization_complete:
                    # FIX: Properly extract vector dimension from embedding tensor
                    first_embedding = embeddings[0]
                    if len(first_embedding.shape) == 2 and first_embedding.shape[0] == 1:
                        # Shape is [1, D], extract D
                        vector_size = first_embedding.shape[1]
                    elif len(first_embedding.shape) == 1:
                        # Shape is [D], use D
                        vector_size = first_embedding.shape[0]
                    else:
                        # Flatten for safety
                        vector_size = first_embedding.numel()
                    
                    print(f"📊 Vector dimension: {vector_size}")

                    client = setup_qdrant(region_collection, vector_size)
                    if client is None:
                        print("❌ Failed to initialize Qdrant client.")
                        return None

                    initialization_complete = True

                # Store embeddings
                store_embeddings_in_qdrant(client, region_collection, embeddings, metadata)
                total_regions += len(embeddings)
                processed_count += 1
            else:
                print(f"⚠️ No valid regions found in image {idx+1}, skipping")
                skipped_count += 1

        except Exception as e:
            print(f"❌ Error processing image {idx+1}: {e}")
            failed_count += 1
            continue

        # Print progress
        if (idx+1) % 5 == 0 or idx == len(image_paths) - 1:
            elapsed_time = time.time() - start_time
            print(f"\n📊 Progress: {idx+1}/{len(image_paths)} images, {total_regions} total regions")
            print(f"⏱️ Time elapsed: {elapsed_time:.1f}s")
            print(f"✅ Processed: {processed_count}, ⏭️ Skipped: {skipped_count}, ❌ Failed: {failed_count}")

    # Print final statistics
    elapsed_time = time.time() - start_time
    print("\n📊 Processing Complete!")
    print(f"⏱️ Total time: {elapsed_time:.1f}s")
    print(f"🔍 Images processed: {processed_count}/{len(image_paths)}")
    print(f"⏭️ Images skipped (no regions): {skipped_count}") 
    print(f"❌ Images failed: {failed_count}")
    print(f"📦 Total regions stored: {total_regions}")
    print(f"📊 Embeddings created with layer: {optimal_layer}")

    return client

def process_folder_for_whole_image_database(
    folder_path,
    collection_name,
    pe_model,
    pe_vit_model, 
    preprocess,
    device,
    optimal_layer=40,
    checkpoint_interval=10,
    resume_from_checkpoint=True
):
    """Process all images in a folder and build a searchable database of whole image embeddings"""
    print(f"Starting to process folder for whole image embeddings: {folder_path}")

    if not os.path.exists(folder_path):
        print(f"❌ Error: Folder not found at {folder_path}")
        return None

    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_paths = []

    for file in os.listdir(folder_path):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_paths.append(os.path.join(folder_path, file))

    if not image_paths:
        print(f"❌ Error: No images found in {folder_path}")
        return None

    print(f"📁 Found {len(image_paths)} images to process")

    # Initialize tracking variables
    processed_count = 0
    failed_count = 0
    total_images = 0

    # Initialize database variables
    client = None
    vector_size = None
    initialization_complete = False

    start_time = time.time()

    # Create a special collection name for whole images with layer information
    whole_image_collection = f"{collection_name}_whole_images_layer{optimal_layer}"
    print(f"📊 Using collection name: {whole_image_collection}")

    # Process images
    for idx, image_path in enumerate(image_paths):
        print(f"\nProcessing image {idx+1}/{len(image_paths)}: {os.path.basename(image_path)}")

        try:
            # Extract whole image embedding
            image, embedding, metadata, label = extract_whole_image_embeddings(
                image_path,
                pe_model_param=pe_model,
                pe_vit_model_param=pe_vit_model,
                preprocess_param=preprocess,
                device_param=device,
                optimal_layer=optimal_layer
            )

            if image is not None and embedding is not None:
                print(f"✅ Successfully extracted embedding for image {idx+1}")

                # Initialize database if not done yet
                if not initialization_complete:
                    # FIX: Properly extract vector dimension from embedding tensor
                    # For whole image, embedding is a single tensor, not a list
                    if len(embedding.shape) == 2 and embedding.shape[0] == 1:
                        # Shape is [1, D], extract D
                        vector_size = embedding.shape[1]
                    elif len(embedding.shape) == 1:
                        # Shape is [D], use D
                        vector_size = embedding.shape[0]
                    else:
                        # Flatten for safety
                        vector_size = embedding.numel()
                    
                    print(f"📊 Vector dimension: {vector_size}")

                    client = setup_qdrant(whole_image_collection, vector_size)
                    if client is None:
                        print("❌ Failed to initialize Qdrant client.")
                        return None

                    initialization_complete = True

                # Store embedding
                store_embeddings_in_qdrant(client, whole_image_collection, [embedding], [metadata])
                total_images += 1
                processed_count += 1
            else:
                print(f"⚠️ Failed to extract embedding for image {idx+1}, skipping")
                failed_count += 1

        except Exception as e:
            print(f"❌ Error processing image {idx+1}: {e}")
            failed_count += 1
            continue

        # Print progress
        if (idx+1) % 5 == 0 or idx == len(image_paths) - 1:
            elapsed_time = time.time() - start_time
            print(f"\n📊 Progress: {idx+1}/{len(image_paths)} images, {total_images} total processed")
            print(f"⏱️ Time elapsed: {elapsed_time:.1f}s")
            print(f"✅ Processed: {processed_count}, ❌ Failed: {failed_count}")

    # Print final statistics
    elapsed_time = time.time() - start_time
    print("\n📊 Processing Complete!")
    print(f"⏱️ Total time: {elapsed_time:.1f}s")
    print(f"🔍 Images processed: {processed_count}/{len(image_paths)}")
    print(f"❌ Images failed: {failed_count}")
    print(f"📊 Embeddings created with layer: {optimal_layer}")

    return client

# =============================================================================
# BATCH PROCESSING FUNCTIONS
# =============================================================================

def process_folder_with_progress(folder_path, prompts, collection_name):
    """Process folder with progress updates for the UI"""
    try:
        print(f"[STATUS] Starting folder processing: {folder_path}")
        print(f"[STATUS] Using collection: {collection_name}")
        print(f"[STATUS] Using prompts: {prompts}")
        
        # Get global model variables
        global pe_model, pe_vit_model, preprocess, device
        
        # Split prompts if provided as period-separated string
        prompt_list = [p.strip() for p in prompts.split(".")] if prompts else ["person", "building", "car", "text", "object"]
        print(f"[STATUS] Parsed prompts: {prompt_list}")
        
        # Get file list
        image_files = [f for f in os.listdir(folder_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
        total = len(image_files)
        
        print(f"[STATUS] Found {total} images to process in folder")
        
        if total == 0:
            print(f"[STATUS] No images found in folder {folder_path}")
            yield "No images found in folder", gr.update(visible=False)
            return
        
        # Process images with progress updates
        client = None
        processed = 0
        skipped = 0
        errors = 0
        total_regions = 0
        
        # Update Gradio with initial status
        yield f"🔍 Starting to process {total} images in {folder_path}", gr.update(visible=False)
        
        print(f"[STATUS] Beginning image processing loop")
        for i, img_file in enumerate(image_files):
            # Process image
            img_path = os.path.join(folder_path, img_file)
            
            # Yield progress update with percentage
            progress_pct = ((i+1) / total) * 100
            yield f"🔄 Processing: {i+1}/{total} images ({progress_pct:.1f}%)\n📄 Current: {img_file}", gr.update(visible=False)
            
            print(f"[STATUS] Processing image {i+1}/{total}: {img_file}")
            
            try:
                # Extract region embeddings - use period-separated format
                image, masks, embeddings, metadata, labels, error_message = extract_region_embeddings_autodistill(
                    img_path,
                    " . ".join(prompt_list) + " .",
                    pe_model,
                    pe_vit_model,
                    preprocess,
                    device,
                    min_area_ratio=0.005,
                    max_regions=5
                )
                
                if image is not None and len(embeddings) > 0:
                    print(f"[STATUS] Found {len(embeddings)} regions in {img_file}")
                    # Setup Qdrant client if needed
                    if client is None:
                        # FIX: Properly extract vector dimension from embedding tensor
                        first_embedding = embeddings[0]
                        if len(first_embedding.shape) == 2 and first_embedding.shape[0] == 1:
                            # Shape is [1, D], extract D
                            vector_size = first_embedding.shape[1]
                        elif len(first_embedding.shape) == 1:
                            # Shape is [D], use D
                            vector_size = first_embedding.shape[0]
                        else:
                            # Flatten for safety
                            vector_size = first_embedding.numel()
                        
                        print(f"[STATUS] Setting up database with vector size {vector_size}")
                        client = setup_qdrant(collection_name, vector_size)
                    
                    # Store embeddings
                    print(f"[STATUS] Storing {len(embeddings)} embeddings in database")
                    store_embeddings_in_qdrant(client, collection_name, embeddings, metadata)
                    processed += 1
                    total_regions += len(embeddings)
                else:
                    print(f"[STATUS] No valid regions found in {img_file}")
                    skipped += 1
            except Exception as e:
                print(f"[ERROR] Error processing {img_path}: {e}")
                import traceback
                traceback.print_exc()
                errors += 1
            
            # Every few images, yield a progress update with detailed stats
            if (i + 1) % 3 == 0 or i == total - 1:
                stats = (
                    f"📊 Progress: {i+1}/{total} images ({(i+1)/total*100:.1f}%)\n"
                    f"✅ Processed: {processed} images\n"
                    f"⏭️ Skipped: {skipped} images (no regions found)\n"
                    f"❌ Errors: {errors} images\n"
                    f"🔍 Total regions found: {total_regions}"
                )
                yield stats, gr.update(visible=i == total-1)
                
        print(f"[STATUS] Processing complete - processed: {processed}, skipped: {skipped}, errors: {errors}, total regions: {total_regions}")
        
        if client is not None:
            print(f"[STATUS] Closing database connection")
            client.close()
        
        final_message = (
            f"✅ Complete! Processing summary:\n\n"
            f"📁 Total images: {total}\n"
            f"✅ Successfully processed: {processed}\n"
            f"⏭️ Skipped (no regions): {skipped}\n"
            f"❌ Errors: {errors}\n"
            f"🔍 Total regions stored: {total_regions}\n\n"
            f"📦 Database collection: '{collection_name}'"
        )
        print(f"[STATUS] {final_message}")
        yield final_message, gr.update(visible=True)
    except Exception as e:
        error_message = f"❌ Error processing folder: {str(e)}"
        print(f"[ERROR] {error_message}")
        import traceback
        traceback.print_exc()
        yield error_message, gr.update(visible=False)

def process_folder_with_progress_advanced(folder_path, prompts, collection_name, 
                                  optimal_layer=40, min_area_ratio=0.01, max_regions=5, 
                                  resume_from_checkpoint=True, pooling_strategy="top_k"):
    """Process folder with progress updates for the UI with advanced parameters"""
    try:
        print(f"[STATUS] Starting folder processing: {folder_path}")
        print(f"[STATUS] Using collection: {collection_name}")
        print(f"[STATUS] Using prompts: {prompts}")
        print(f"[STATUS] Advanced parameters: optimal_layer={optimal_layer}, min_area_ratio={min_area_ratio}, " 
              f"max_regions={max_regions}, resume_from_checkpoint={resume_from_checkpoint}, pooling_strategy={pooling_strategy}")
        
        # Input validation
        if not folder_path or not os.path.exists(folder_path):
            error_msg = f"❌ Error: Folder '{folder_path}' does not exist or is invalid"
            print(f"[ERROR] {error_msg}")
            yield error_msg, gr.update(visible=False)
            return
            
        if not collection_name or not isinstance(collection_name, str):
            collection_name = f"image_database_{int(time.time())}"
            print(f"[WARNING] Invalid collection name provided, using: {collection_name}")
        
        # Ensure parameters are in valid ranges
        optimal_layer = max(1, int(optimal_layer))  # Only ensure minimum layer of 1
        min_area_ratio = max(0.001, min(float(min_area_ratio), 0.5))  # Between 0.001-0.5
        max_regions = max(1, min(int(max_regions), 20))  # Between 1-20
        
        # Get global model variables
        global pe_model, pe_vit_model, preprocess, device
        
        # Process and validate prompts
        if not prompts or not isinstance(prompts, str) or prompts.strip() == "":
            prompts = "person . building . car . text . object ."
            print(f"[WARNING] No valid prompts provided, using defaults: {prompts}")
            
        # Split prompts if provided as period-separated string
        prompt_list = [p.strip().lower() for p in prompts.split(".") if p.strip()]
        
        # Ensure we have valid prompts
        if not prompt_list:
            prompt_list = ["person", "building", "car", "text", "object"]
        
        print(f"[STATUS] Parsed prompts: {prompt_list}")
        
        # Get file list - create a stable sorting to ensure consistent processing order
        image_files = sorted([f for f in os.listdir(folder_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))])
        total = len(image_files)
        
        print(f"[STATUS] Found {total} images to process in folder")
        
        if total == 0:
            print(f"[STATUS] No images found in folder {folder_path}")
            yield "No images found in folder", gr.update(visible=False)
            return
        
        # Process images with progress updates
        client = None
        processed = 0
        skipped = 0
        errors = 0
        total_regions = 0
        vector_dimension = None
        
        # Force database cleanup before starting to avoid connection issues
        from threading import Thread
        cleanup_thread = Thread(target=cleanup_qdrant_connections, args=(True,))
        cleanup_thread.start()
        cleanup_thread.join(timeout=10)
        
        # Use collection name provided (should already include layer info from caller)
        collection_with_layer = collection_name
        
        # Update Gradio with initial status
        yield (f"🔍 Starting to process {total} images in {folder_path}\n"
              f"📊 Parameters:\n"
              f"  - Semantic Layer: {optimal_layer}\n"
              f"  - Min Region Size: {min_area_ratio}\n"
              f"  - Max Regions: {max_regions}\n"
              f"  - Resume from checkpoint: {resume_from_checkpoint}\n"
              f"  - Collection name: {collection_with_layer}"), gr.update(visible=False)
        
        print(f"[STATUS] Beginning image processing loop")
        for i, img_file in enumerate(image_files):
            # Process image
            img_path = os.path.join(folder_path, img_file)
            
            # Yield progress update with percentage
            progress_pct = ((i+1) / total) * 100
            yield f"🔄 Processing: {i+1}/{total} images ({progress_pct:.1f}%)\n📄 Current: {img_file}", gr.update(visible=False)
            
            print(f"[STATUS] Processing image {i+1}/{total}: {img_file}")
            
            # Skip non-image files silently
            if not os.path.isfile(img_path) or not img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                print(f"[WARNING] Skipping non-image file: {img_file}")
                skipped += 1
                continue
                
            # Skip files that are too small or potentially corrupt
            try:
                img_size = os.path.getsize(img_path)
                if img_size < 1000:  # Files smaller than 1KB are likely invalid
                    print(f"[WARNING] Skipping very small file ({img_size} bytes): {img_file}")
                    skipped += 1
                    continue
            except Exception as e:
                print(f"[WARNING] Error checking file size: {e}")
            
            try:
                # Add memory management to avoid memory leaks
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                
                # Extract region embeddings with custom parameters
                image, masks, embeddings, metadata, labels, error_message = extract_region_embeddings_autodistill(
                    img_path,
                    " . ".join(prompt_list) + " .",
                    pe_model_param=pe_model,
                    pe_vit_model_param=pe_vit_model,
                    preprocess_param=preprocess,
                    device_param=device,
                    min_area_ratio=min_area_ratio,
                    max_regions=max_regions,
                    optimal_layer=optimal_layer,
                    pooling_strategy=pooling_strategy
                )
                
                # Check if there was an error during processing
                if error_message is not None:
                    print(f"[ERROR] Error processing {img_file}: {error_message}")
                    errors += 1
                    # Show error in UI progress updates
                    if (i + 1) % 1 == 0 or i == total - 1:  # Update every image when there are errors
                        error_stats = (
                            f"❌ Error processing {img_file}:\n{error_message}\n\n"
                            f"📊 Progress: {i+1}/{total} images ({(i+1)/total*100:.1f}%)\n"
                            f"✅ Processed: {processed} images\n"
                            f"⏭️ Skipped: {skipped} images (no regions found)\n"
                            f"❌ Errors: {errors} images\n"
                            f"🔍 Total regions found: {total_regions}"
                        )
                        yield error_stats, gr.update(visible=False)
                    continue
                
                if image is not None and len(embeddings) > 0:
                    print(f"[STATUS] Found {len(embeddings)} regions in {img_file}")
                    
                    # Additional validation for database creation
                    valid_embeddings = []
                    valid_metadata = []
                    invalid_count = 0
                    
                    for idx, (emb, meta) in enumerate(zip(embeddings, metadata)):
                        # Skip regions with missing or invalid data
                        if emb is None or meta is None:
                            print(f"[WARNING] Region {idx} has None embedding or metadata, skipping")
                            invalid_count += 1
                            continue
                            
                        # Validate embedding dimensions
                        if vector_dimension is not None:
                            # Use the same logic as dimension detection to extract actual feature dimension
                            if len(emb.shape) == 2 and emb.shape[0] == 1:
                                # Shape is [1, D], extract D
                                actual_dimension = emb.shape[1]
                            elif len(emb.shape) == 1:
                                # Shape is [D], use D
                                actual_dimension = emb.shape[0]
                            else:
                                # Flatten for safety
                                actual_dimension = emb.numel()
                            
                            if actual_dimension != vector_dimension:
                                print(f"[WARNING] Region {idx} has inconsistent embedding dimension: expected {vector_dimension}, got {actual_dimension}")
                                invalid_count += 1
                                continue
                        
                        # Validate the region metadata
                        if "bbox" not in meta or len(meta["bbox"]) != 4:
                            print(f"[WARNING] Region {idx} has invalid bbox, skipping")
                            invalid_count += 1
                            continue
                            
                        # Skip regions with suspiciously high confidence (1.0 exactly)
                        phrase = meta.get("phrase", "")
                        if ": 1.00" in phrase:
                            print(f"[WARNING] Region {idx} has perfect 1.00 confidence, likely false positive: {phrase}")
                            invalid_count += 1
                            continue
                        
                        # Add timestamp to metadata to track when it was processed
                        meta["processed_timestamp"] = time.time()
                        meta["source_filename"] = img_file
                            
                        # Include only valid embeddings
                        valid_embeddings.append(emb)
                        valid_metadata.append(meta)
                    
                    # If we filtered out bad regions, update counts
                    if invalid_count > 0:
                        print(f"[STATUS] Filtered out {invalid_count}/{len(embeddings)} invalid regions")
                        embeddings = valid_embeddings
                        metadata = valid_metadata
                    
                    # Setup Qdrant client if needed
                    if len(embeddings) > 0:
                        # Store the vector dimension for consistency checking
                        if vector_dimension is None:
                            # FIX: Properly extract vector dimension from embedding tensor
                            # The embeddings are shape [1, D] or [D], we want D
                            first_embedding = embeddings[0]
                            
                            # EXTENSIVE DEBUG OUTPUT
                            print(f"[DEBUG] Raw first_embedding: {first_embedding}")
                            print(f"[DEBUG] first_embedding type: {type(first_embedding)}")
                            print(f"[DEBUG] first_embedding shape: {first_embedding.shape}")
                            print(f"[DEBUG] first_embedding.shape has {len(first_embedding.shape)} dimensions")
                            if len(first_embedding.shape) >= 1:
                                print(f"[DEBUG] first_embedding.shape[0] = {first_embedding.shape[0]}")
                            if len(first_embedding.shape) >= 2:
                                print(f"[DEBUG] first_embedding.shape[1] = {first_embedding.shape[1]}")
                            
                            if len(first_embedding.shape) == 2 and first_embedding.shape[0] == 1:
                                # Shape is [1, D], extract D
                                vector_dimension = first_embedding.shape[1]
                                print(f"[DEBUG] Using 2D logic: vector_dimension = {vector_dimension}")
                            elif len(first_embedding.shape) == 1:
                                # Shape is [D], use D
                                vector_dimension = first_embedding.shape[0]
                                print(f"[DEBUG] Using 1D logic: vector_dimension = {vector_dimension}")
                            else:
                                # Flatten for safety
                                vector_dimension = first_embedding.numel()
                                print(f"[DEBUG] Using fallback logic: vector_dimension = {vector_dimension}")
                            
                            print(f"[STATUS] Established vector dimension: {vector_dimension}")
                            print(f"[DEBUG] First embedding shape: {first_embedding.shape}")
                        
                        if client is None:
                            print(f"[STATUS] Setting up database with vector size {vector_dimension}")
                            client = setup_qdrant(collection_with_layer, vector_dimension)
                            
                            # If we still couldn't create a client, try aggressive cleanup
                            if client is None:
                                print(f"[WARNING] Failed to create database client, attempting recovery...")
                                cleanup_qdrant_connections(force=True)
                                time.sleep(2)
                                client = setup_qdrant(collection_with_layer, vector_dimension)
                                
                                if client is None:
                                    error_msg = "❌ Failed to create database connection after cleanup attempts."
                                    print(f"[ERROR] {error_msg}")
                                    return error_msg, gr.update(visible=False)
                        
                        # Add layer information to each region's metadata
                        for md in metadata:
                            md["embedding_layer"] = optimal_layer
                            md["processing_parameters"] = {
                                "min_area_ratio": min_area_ratio,
                                "max_regions": max_regions
                            }
                        
                        # Store embeddings in batches with retry mechanism
                        retry_count = 0
                        max_retries = 3
                        stored_successfully = False
                        actual_stored_count = 0
                        
                        while not stored_successfully and retry_count < max_retries:
                            try:
                                # Store embeddings
                                print(f"[STATUS] Storing {len(embeddings)} embeddings in database")
                                point_ids = store_embeddings_in_qdrant(client, collection_with_layer, embeddings, metadata)
                                actual_stored_count = len(point_ids)
                                stored_successfully = actual_stored_count > 0
                                
                                if stored_successfully:
                                    processed += 1
                                    total_regions += actual_stored_count
                                    print(f"[STATUS] Successfully stored {actual_stored_count} regions in database")
                                else:
                                    print(f"[WARNING] Failed to store embeddings in database (0 stored), attempt {retry_count + 1}/{max_retries}")
                                    
                            except Exception as e:
                                print(f"[ERROR] Error storing embeddings, attempt {retry_count + 1}/{max_retries}: {e}")
                                import traceback
                                traceback.print_exc()
                                
                            if not stored_successfully and retry_count < max_retries - 1:
                                # Sleep before retrying
                                time.sleep(1 + retry_count)
                                retry_count += 1
                                # Try to recreate client
                                try:
                                    client.close()
                                except:
                                    pass
                                client = setup_qdrant(collection_with_layer, vector_dimension)
                            else:
                                break
                                
                        if not stored_successfully:
                            errors += 1
                            print(f"[ERROR] Failed to store regions for {img_file} after {max_retries} attempts - NO REGIONS STORED")
                    else:
                        print(f"[STATUS] No valid regions remain after filtering for {img_file}")
                        skipped += 1
                else:
                    print(f"[STATUS] No valid regions found in {img_file}")
                    skipped += 1
            except Exception as e:
                print(f"[ERROR] Error processing {img_path}: {e}")
                import traceback
                traceback.print_exc()
                errors += 1
            
            # Every few images, yield a progress update with detailed stats
            if (i + 1) % 3 == 0 or i == total - 1:
                stats = (
                    f"📊 Progress: {i+1}/{total} images ({(i+1)/total*100:.1f}%)\n"
                    f"✅ Processed: {processed} images\n"
                    f"⏭️ Skipped: {skipped} images (no regions found)\n"
                    f"❌ Errors: {errors} images\n"
                    f"🔍 Total regions found: {total_regions}\n\n"
                    f"🧠 Using semantic layer: {optimal_layer}\n"
                    f"🔍 Min region size: {min_area_ratio}\n"
                    f"📏 Max regions per image: {max_regions}\n"
                    f"📦 Collection: {collection_with_layer}"
                )
                yield stats, gr.update(visible=i == total-1)
                
        print(f"[STATUS] Processing complete - processed: {processed}, skipped: {skipped}, errors: {errors}, total regions: {total_regions}")
        
        if client is not None:
            print(f"[STATUS] Closing database connection")
            try:
                client.close()
            except Exception as e:
                print(f"[WARNING] Error closing client: {e}")
        
        # Final stats
        if processed == 0 and total > 0:
            if errors > 0:
                final_message = (
                    f"⚠️ Warning: No images were successfully processed!\n\n"
                    f"📁 Total images: {total}\n"
                    f"❌ Errors: {errors} images\n"
                    f"⏭️ Skipped: {skipped} images\n\n"
                    f"Please check the console for error messages."
                )
            else:
                final_message = (
                    f"⚠️ Warning: No regions were found in any images!\n\n"
                    f"📁 Total images: {total}\n"
                    f"⏭️ Skipped: {skipped} images (no regions found)\n\n"
                    f"Try adjusting the parameters:\n"
                    f"- Decrease Min Region Size (currently {min_area_ratio})\n"
                    f"- Add more detection prompts (currently {prompts})\n"
                )
        else:
            final_message = (
                f"✅ Complete! Processing summary:\n\n"
                f"📁 Total images: {total}\n"
                f"✅ Successfully processed: {processed}\n"
                f"⏭️ Skipped: {skipped} images (no regions found)\n"
                f"❌ Errors: {errors}\n"
                f"🔍 Total regions stored: {total_regions}\n\n"
                f"🧠 Semantic layer used: {optimal_layer}\n"
                f"🔍 Min region size: {min_area_ratio}\n"
                f"📏 Max regions per image: {max_regions}\n\n"
                f"📦 Database collection: '{collection_with_layer}'"
            )
        print(f"[STATUS] {final_message}")
        yield final_message, gr.update(visible=True)
    except Exception as e:
        error_message = f"❌ Error processing folder: {str(e)}"
        print(f"[ERROR] {error_message}")
        import traceback
        traceback.print_exc()
        yield error_message, gr.update(visible=False)

def process_folder_with_progress_whole_images(folder_path, collection_name, 
                                    optimal_layer=40, resume_from_checkpoint=True):
    """Process folder with whole image embeddings for the UI"""
    try:
        print(f"[STATUS] Starting whole image processing for folder: {folder_path}")
        print(f"[STATUS] Using collection: {collection_name}")
        print(f"[STATUS] Using embedding layer: {optimal_layer}")
        
        # Get global model variables
        global pe_model, pe_vit_model, preprocess, device
        
        # Get file list
        image_files = [f for f in os.listdir(folder_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
        total = len(image_files)
        
        print(f"[STATUS] Found {total} images to process in folder")
        
        if total == 0:
            print(f"[STATUS] No images found in folder {folder_path}")
            yield "No images found in folder", gr.update(visible=False)
            return
        
        # Process images with progress updates
        client = None
        processed = 0
        errors = 0
        
        # Use collection name provided (should already include layer info from caller)
        whole_image_collection = collection_name
        
        # Update Gradio with initial status
        yield f"🔍 Starting to process {total} whole images in {folder_path}\n📊 Collection: {whole_image_collection}\n🧠 Using embedding layer: {optimal_layer}", gr.update(visible=False)
        
        print(f"[STATUS] Beginning whole image processing loop")
        for i, img_file in enumerate(image_files):
            # Process image
            img_path = os.path.join(folder_path, img_file)
            
            # Yield progress update with percentage
            progress_pct = ((i+1) / total) * 100
            yield f"🔄 Processing: {i+1}/{total} images ({progress_pct:.1f}%)\n📄 Current: {img_file}", gr.update(visible=False)
            
            print(f"[STATUS] Processing image {i+1}/{total}: {img_file}")
            
            try:
                # Extract whole image embedding
                image, embedding, metadata, label = extract_whole_image_embeddings(
                    img_path,
                    pe_model,
                    pe_vit_model,
                    preprocess,
                    device,
                    optimal_layer=optimal_layer
                )
                
                if image is not None and embedding is not None:
                    print(f"[STATUS] Successfully extracted embedding for {img_file}")
                    # Setup Qdrant client if needed
                    if client is None:
                        # FIX: Properly extract vector dimension from embedding tensor
                        # For whole image, embedding is a single tensor, not a list
                        if len(embedding.shape) == 2 and embedding.shape[0] == 1:
                            # Shape is [1, D], extract D
                            vector_size = embedding.shape[1]
                        elif len(embedding.shape) == 1:
                            # Shape is [D], use D
                            vector_size = embedding.shape[0]
                        else:
                            # Flatten for safety
                            vector_size = embedding.numel()
                        
                        print(f"[STATUS] Setting up database with vector size {vector_size}")
                        client = setup_qdrant(whole_image_collection, vector_size)
                    
                    # Store embedding
                    print(f"[STATUS] Storing embedding in database")
                    store_embeddings_in_qdrant(client, whole_image_collection, [embedding], [metadata])
                    processed += 1
                else:
                    print(f"[STATUS] Failed to extract embedding for {img_file}")
            except Exception as e:
                print(f"[ERROR] Error processing {img_path}: {e}")
                import traceback
                traceback.print_exc()
                errors += 1
                
            # Provide periodic updates
            if (i+1) % 5 == 0 or i == len(image_files) - 1:
                yield f"🔄 Processing: {i+1}/{total} images ({progress_pct:.1f}%)\n✅ Processed: {processed}\n❌ Errors: {errors}\n🧠 Using embedding layer: {optimal_layer}", gr.update(visible=False)
        
        # Final update
        if processed > 0:
            final_message = f"✅ Processing complete!\n📊 Processed {processed}/{total} images\n🗄️ Collection: {whole_image_collection}\n🧠 Using embedding layer: {optimal_layer}"
            print(f"[STATUS] {final_message}")
            yield final_message, gr.update(visible=True)
        else:
            error_message = f"❌ Processing failed. No images were successfully processed."
            print(f"[STATUS] {error_message}")
            yield error_message, gr.update(visible=False)
        
    except Exception as e:
        error_message = f"❌ Error: {str(e)}"
        print(f"[ERROR] Batch processing error: {e}")
        import traceback
        traceback.print_exc()
        yield error_message, gr.update(visible=False)

# =============================================================================
# GRADIO INTERFACE CLASSES
# =============================================================================

class GradioInterface:
    """Gradio interface for the Grounded SAM Region Search application"""
    
    def __init__(self, pe_model, pe_vit_model, preprocess, device):
        """Initialize the interface with required models"""
        self.pe_model = pe_model
        self.pe_vit_model = pe_vit_model
        self.preprocess = preprocess
        self.device = device
        
        # State variables
        self.detected_regions = {
            "image": None,
            "masks": [],
            "embeddings": [],
            "metadata": [],
            "labels": []
        }
        
        # Add whole image state
        self.whole_image = {
            "image": None,
            "embedding": None,
            "metadata": None,
            "label": None
        }
        
        self.image_cache = {}
        self.search_result_images = []
        self.active_client = None
        self.all_db_embeddings = None # For storing all embeddings from selected DB
        self.all_db_metadata = None   # For storing all metadata from selected DB
        
        # Get available collections
        self.available_collections = list_available_databases()
        # Set default collection based on available databases
        if self.available_collections:
            self.active_collection = self.available_collections[0]
            print(f"[STATUS] Using default collection: {self.active_collection}")
        else:
            self.active_collection = "grounded_image_regions"
            print(f"[STATUS] No existing collections found. Using default name: {self.active_collection}")
    
    def set_active_collection(self, collection_name):
        """Set the active collection for search operations"""
        if collection_name and collection_name != "No databases found":
            if collection_name == self.active_collection:
                # No change needed
                print(f"[DEBUG] Collection already set to {collection_name}")
                return f"Using database: {collection_name}"
                
            previous = self.active_collection
            
            # Store the full collection name exactly as provided
            # This ensures search operations use the correct collection name with layer suffix
            self.active_collection = collection_name
            print(f"[STATUS] Changed active collection from {previous} to {collection_name}")
            
            # Close existing client if database changed
            if self.active_client is not None:
                print(f"[STATUS] Closing previous database connection")
                try:
                    self.active_client.close()
                except Exception as e:
                    print(f"[WARN] Error closing Qdrant client: {e}")
                self.active_client = None
            
            # Reset stored embeddings and metadata
            self.all_db_embeddings = None
            self.all_db_metadata = None

            try:
                print(f"[STATUS] Connecting to Qdrant to load all embeddings for collection: {collection_name}")
                # Ensure client is re-initialized if it was closed or is None
                self.active_client = QdrantClient(path="./image_retrieval_project/qdrant_data")
                
                # Check if collection exists
                collections_info = self.active_client.get_collections()
                if not any(c.name == collection_name for c in collections_info.collections):
                    message = f"Error: Collection '{collection_name}' not found in Qdrant."
                    print(f"[ERROR] {message}")
                    return message

                collection_info = self.active_client.get_collection(collection_name=collection_name)
                vector_size = collection_info.config.params.vectors.size
                total_points = collection_info.points_count
                
                if total_points == 0:
                    print(f"[INFO] Collection '{collection_name}' is empty. No embeddings to load.")
                    return f"Set active database to: {collection_name} (0 embeddings loaded)"

                print(f"[STATUS] Loading {total_points} embeddings of size {vector_size} from '{collection_name}'...")
                
                all_vectors = []
                all_payloads = []
                
                # Use scroll API to fetch all points
                offset = None
                processed_count = 0
                while True:
                    response = self.active_client.scroll(
                        collection_name=collection_name,
                        offset=offset,
                        limit=256, # Adjust batch size as needed
                        with_payload=True,
                        with_vectors=True
                    )
                    points = response[0] # Actual points
                    next_offset = response[1] # Next offset for pagination

                    if not points:
                        break
                    
                    for point in points:
                        all_vectors.append(point.vector)
                        all_payloads.append(point.payload)
                    
                    processed_count += len(points)
                    print(f"[STATUS] Loaded {processed_count}/{total_points} embeddings...")
                    
                    if next_offset is None:
                        break
                    offset = next_offset
                
                if all_vectors:
                    self.all_db_embeddings = torch.tensor(all_vectors, dtype=torch.float32)
                    self.all_db_metadata = all_payloads
                    print(f"[SUCCESS] Loaded {len(self.all_db_embeddings)} embeddings into memory for local search.")
                    # Move to device if GPU is available and embeddings are successfully loaded
                    if self.device and self.all_db_embeddings is not None:
                        try:
                            self.all_db_embeddings = self.all_db_embeddings.to(self.device)
                            print(f"[STATUS] Moved all_db_embeddings to device: {self.device}")
                        except Exception as e:
                            print(f"[WARN] Failed to move all_db_embeddings to device {self.device}: {e}. Using CPU.")
                else:
                    print(f"[WARN] No vectors found in collection '{collection_name}' despite points_count={total_points}.")
                    self.all_db_embeddings = None # Ensure it's None if no vectors
                    self.all_db_metadata = []


            except Exception as e:
                error_msg = f"Error loading embeddings from {collection_name}: {e}"
                print(f"[ERROR] {error_msg}")
                import traceback
                traceback.print_exc()
                self.all_db_embeddings = None # Ensure reset on error
                self.all_db_metadata = None
                return error_msg # Return error message to UI

            return f"Set active database to: {collection_name} ({len(self.all_db_embeddings) if self.all_db_embeddings is not None else 0} embeddings loaded)"
        return "No database selected"

    def process_image_with_prompt(self, 
                                image_pil_or_np, # Gradio might pass NumPy array or PIL
                                text_prompt, 
                                min_area_ratio=0.01, 
                                max_regions=5, 
                                # NEW PARAMETERS from UI
                                extraction_mode="single", 
                                optimal_layer_ui=40,
                                preset_name_ui="object_focused",
                                custom_layer_1=30, custom_weight_1=0.3,
                                custom_layer_2=40, custom_weight_2=0.4,
                                custom_layer_3=47, custom_weight_3=0.3,
                                pooling_strategy_ui="top_k",
                                temperature_ui=0.07, 
                                top_k_ratio_ui=0.1, 
                                attention_heads_ui=8
                                ):
        """Process an uploaded image with a text prompt to detect regions"""
        if isinstance(image_pil_or_np, np.ndarray):
            image_pil = Image.fromarray(image_pil_or_np)
        else:
            image_pil = image_pil_or_np # Assuming it's already PIL

        custom_layers_list = [cl for cl in [custom_layer_1, custom_layer_2, custom_layer_3] if cl is not None]
        custom_weights_list = [cw for cw in [custom_weight_1, custom_weight_2, custom_weight_3] if cw is not None]
        
        # Adjust weights list to match the number of valid layers if necessary
        if len(custom_layers_list) < len(custom_weights_list):
            custom_weights_list = custom_weights_list[:len(custom_layers_list)]
        elif len(custom_weights_list) < len(custom_layers_list): # Should not happen if UI enforces it
            # Fill remaining weights with default if fewer weights than layers
            default_weight = 0.1 
            custom_weights_list.extend([default_weight] * (len(custom_layers_list) - len(custom_weights_list)))


        print(f"Gradio process_image_with_prompt:")
        print(f"  extraction_mode='{extraction_mode}', optimal_layer_ui={optimal_layer_ui}")
        print(f"  preset_name_ui='{preset_name_ui}'")
        print(f"  custom_layers_list={custom_layers_list}, custom_weights_list={custom_weights_list}")
        print(f"  pooling_strategy_ui='{pooling_strategy_ui}', temperature_ui={temperature_ui}")
        print(f"  top_k_ratio_ui={top_k_ratio_ui}, attention_heads_ui={attention_heads_ui}")
        print(f"  min_area_ratio={min_area_ratio}, max_regions={max_regions}")


        # Extract regions from image
        image_np, masks, embeddings, metadata, labels, error_message = extract_region_embeddings_autodistill(
            image_source=image_pil, # Pass the PIL image
            text_prompt=text_prompt,
            pe_model_param=self.pe_model,
            pe_vit_model_param=self.pe_vit_model,
            preprocess_param=self.preprocess,
            device_param=self.device,
            is_url=False, # Direct image upload
            min_area_ratio=min_area_ratio,
            max_regions=max_regions,
            # max_image_size is handled by extract_region_embeddings_autodistill default
            
            # NEWLY PASSED PARAMETERS
            extraction_mode=extraction_mode,
            optimal_layer=optimal_layer_ui, 
            preset_name=preset_name_ui,
            custom_layers=custom_layers_list,
            custom_weights=custom_weights_list,
            pooling_strategy=pooling_strategy_ui, 
            temperature=temperature_ui,
            top_k_ratio=top_k_ratio_ui,
            attention_heads=attention_heads_ui
        )

        # Store results
        self.detected_regions = {
            "image": image_np,
            "masks": masks,
            "embeddings": embeddings,
            "metadata": metadata,
            "labels": labels
        }

        if len(masks) == 0:
            return None, f"No regions found with prompt: '{text_prompt}'", gr.Dropdown(choices=[], value=None), None

        # Create visualization
        fig = visualize_detected_regions(image_np, masks, labels)
        
        # Create dropdown choices
        choices = [f"Region {i+1}: {label}" for i, label in enumerate(labels)]

        # Save figure to buffer
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)

        segmented_image = Image.open(buf)

        # Create preview of first region
        region_preview = None
        if len(masks) > 0:
            region_preview = self.create_region_preview(image_np, masks[0], labels[0])

        return segmented_image, f"Found {len(masks)} regions", gr.Dropdown(choices=choices, value=choices[0] if choices else None), region_preview

    def create_region_preview(self, image, mask, label=None):
        """Creates a visualization of a single region with enhanced visibility"""
        preview = image.copy()
        
        # Create a mask for highlighting
        mask_3d = np.stack([mask, mask, mask], axis=2)
        
        # Make the region much brighter while dimming the surroundings
        highlighted = np.where(mask_3d, np.minimum(preview * 1.8, 255), preview * 0.3)

        # Find contours for drawing a border
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Draw a thicker green border around the region
        cv2.drawContours(highlighted, contours, -1, (0, 255, 0), 3)

        # Create the figure with a border
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(highlighted.astype(np.uint8))
        
        # Show the label if provided
        if label:
            ax.set_title(label, fontsize=12, pad=10)
            
        # Add a border to the figure
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(3)
            spine.set_color('#01579b')  # Blue border color
            
        ax.axis('on')  # Show border
        ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, 
                      labelbottom=False, labelleft=False)  # Hide ticks but keep border

        # Convert to image
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)

        return Image.open(buf)

    def search_region(self, region_selection, similarity_threshold=0.5, max_results=5, temperature_gr=0.07): # Added temperature_gr, removed optimal_layer
        """Search for similar regions using in-memory embeddings and temperature-scaled similarity."""
        print(f"[STATUS] Starting local search for similar regions with temperature: {temperature_gr}")

        if region_selection is None or self.detected_regions["image"] is None:
            print(f"[STATUS] No region selected or no detected regions available for query.")
            return None, "Please select a region first.", gr.update(visible=False), gr.update(choices=[], value=None)

        if self.all_db_embeddings is None or self.all_db_metadata is None:
            error_msg = "Database embeddings not loaded. Please select a database/collection first."
            print(f"[ERROR] {error_msg}")
            return None, error_msg, gr.update(visible=False), gr.update(choices=[], value=None)
        
        if self.all_db_embeddings.numel() == 0:
            error_msg = f"The selected database '{self.active_collection}' is empty."
            print(f"[ERROR] {error_msg}")
            return None, error_msg, gr.update(visible=False), gr.update(choices=[], value=None)

        try:
            region_idx = int(region_selection.split(":")[0].replace("Region ", "")) - 1
            if not (0 <= region_idx < len(self.detected_regions["embeddings"])):
                print(f"[ERROR] Invalid region index: {region_idx}")
                return None, "Invalid region selection.", gr.update(visible=False), gr.update(choices=[], value=None)

            query_embedding = self.detected_regions["embeddings"][region_idx]
            if not isinstance(query_embedding, torch.Tensor): # Ensure it's a tensor
                query_embedding = torch.tensor(query_embedding, dtype=torch.float32)
            
            # Move query embedding to the same device as db embeddings (which should be self.device or CPU)
            query_embedding = query_embedding.to(self.all_db_embeddings.device)

            print(f"[STATUS] Retrieved query embedding for region {region_idx}, shape: {query_embedding.shape}, device: {query_embedding.device}")
            print(f"[STATUS] Database embeddings shape: {self.all_db_embeddings.shape}, device: {self.all_db_embeddings.device}")

            # Calculate temperature-scaled similarity scores against all DB embeddings
            # The temperature_scaled_similarity function handles normalization andunsqueeze for query
            scores_tensor = temperature_scaled_similarity(query_embedding, self.all_db_embeddings, temperature_gr)
            
            # Scores_tensor is 1D if query_embedding was (D) and all_db_embeddings (N,D)
            # Or (N) if query_embedding was (1,D) and all_db_embeddings (N,D)
            # Squeeze if it's (1,N) from unsqueezed query vs (N,D) candidates
            if scores_tensor.ndim == 2 and scores_tensor.shape[0] == 1:
                scores_tensor = scores_tensor.squeeze(0)

            if scores_tensor.numel() != len(self.all_db_metadata):
                error_msg = f"Mismatch between number of scores ({scores_tensor.numel()}) and metadata entries ({len(self.all_db_metadata)})."
                print(f"[ERROR] {error_msg}")
                return None, error_msg, gr.update(visible=False), gr.update(choices=[], value=None)

            # Combine scores with metadata
            results_with_scores = []
            for i, score_val in enumerate(scores_tensor):
                # Create a Qdrant-like ScoredPoint structure for compatibility
                # The ID can be derived from metadata if available, else use index
                point_id = self.all_db_metadata[i].get("region_id", str(uuid.uuid4()))
                
                # Ensure payload is a dictionary (it should be)
                payload = self.all_db_metadata[i]
                if not isinstance(payload, dict):
                    payload = {"data": payload} # Basic fallback

                # Add collection type and name to payload for visualization consistency
                payload["collection_type"] = "region" if "_layer" in self.active_collection and "_whole_images_layer" not in self.active_collection else "whole_image"
                payload["collection_name"] = self.active_collection
                
                scored_point = models.ScoredPoint(
                    id=point_id,
                    version=0, # Dummy version
                    score=score_val.item(), # Actual score from temp_scaled_similarity
                    payload=payload,
                    vector=None # Not returning vector in search results display
                )
                results_with_scores.append(scored_point)

            # Sort results by score (descending)
            results_with_scores.sort(key=lambda x: x.score, reverse=True)
            
            # Apply similarity threshold
            # Note: The meaning of 'similarity_threshold' changes with temp-scaled scores.
            # These scores are not bounded [0,1] like cosine similarity.
            # For now, applying it directly. May need adjustment based on typical score range.
            final_results = [res for res in results_with_scores if res.score >= float(similarity_threshold)]
            
            # Limit to max_results
            final_results = final_results[:int(max_results)]

            print(f"[STATUS] Found {len(final_results)} similar regions after filtering and sorting.")

            if not final_results:
                return None, f"No similar items found above threshold {similarity_threshold} with current temperature.", gr.update(visible=False), gr.update(choices=[], value=None)

            return self.create_unified_search_results_visualization(
                final_results,
                self.detected_regions["image"],
                self.detected_regions["masks"][region_idx],
                self.detected_regions["labels"][region_idx],
                "region"
            )

        except Exception as e:
            error_message = f"Error during local region search: {str(e)}"
            print(f"[ERROR] {error_message}")
            import traceback
            traceback.print_exc()
            return None, error_message, gr.update(visible=False), gr.update(choices=[], value=None)

    def search_whole_image(self, similarity_threshold=0.5, max_results=5, temperature_gr=0.07): # Added temperature_gr, removed optimal_layer
        """Search for similar whole images using in-memory embeddings and temperature-scaled similarity."""
        print(f"[STATUS] Starting local search for similar whole images with temperature: {temperature_gr}")

        if self.whole_image["image"] is None or self.whole_image["embedding"] is None:
            print(f"[STATUS] No processed whole image available for query.")
            return None, "Please process an image first to get its whole embedding.", gr.update(visible=False), gr.update(choices=[], value=None)

        if self.all_db_embeddings is None or self.all_db_metadata is None:
            error_msg = "Database embeddings not loaded. Please select a database/collection first."
            print(f"[ERROR] {error_msg}")
            return None, error_msg, gr.update(visible=False), gr.update(choices=[], value=None)

        if self.all_db_embeddings.numel() == 0:
            error_msg = f"The selected database '{self.active_collection}' is empty."
            print(f"[ERROR] {error_msg}")
            return None, error_msg, gr.update(visible=False), gr.update(choices=[], value=None)
            
        try:
            query_embedding = self.whole_image["embedding"]
            if not isinstance(query_embedding, torch.Tensor): # Ensure it's a tensor
                query_embedding = torch.tensor(query_embedding, dtype=torch.float32)

            # Move query embedding to the same device as db embeddings
            query_embedding = query_embedding.to(self.all_db_embeddings.device)
            
            print(f"[STATUS] Retrieved query embedding for whole image, shape: {query_embedding.shape}, device: {query_embedding.device}")
            print(f"[STATUS] Database embeddings shape: {self.all_db_embeddings.shape}, device: {self.all_db_embeddings.device}")

            scores_tensor = temperature_scaled_similarity(query_embedding, self.all_db_embeddings, temperature_gr)
            
            if scores_tensor.ndim == 2 and scores_tensor.shape[0] == 1: # Ensure 1D tensor of scores
                scores_tensor = scores_tensor.squeeze(0)

            if scores_tensor.numel() != len(self.all_db_metadata):
                error_msg = f"Mismatch between number of scores ({scores_tensor.numel()}) and metadata entries ({len(self.all_db_metadata)})."
                print(f"[ERROR] {error_msg}")
                return None, error_msg, gr.update(visible=False), gr.update(choices=[], value=None)

            results_with_scores = []
            for i, score_val in enumerate(scores_tensor):
                point_id = self.all_db_metadata[i].get("region_id", str(uuid.uuid4())) # region_id is used as point ID
                
                payload = self.all_db_metadata[i]
                if not isinstance(payload, dict): payload = {"data": payload}

                payload["collection_type"] = "whole_image" # Assuming only whole images in a whole image search context
                payload["collection_name"] = self.active_collection
                
                scored_point = models.ScoredPoint(
                    id=point_id,
                    version=0, 
                    score=score_val.item(),
                    payload=payload,
                    vector=None 
                )
                results_with_scores.append(scored_point)

            results_with_scores.sort(key=lambda x: x.score, reverse=True)
            
            final_results = [res for res in results_with_scores if res.score >= float(similarity_threshold)]
            final_results = final_results[:int(max_results)]

            print(f"[STATUS] Found {len(final_results)} similar whole images after filtering and sorting.")

            if not final_results:
                return None, f"No similar whole images found above threshold {similarity_threshold} with current temperature.", gr.update(visible=False), gr.update(choices=[], value=None)

            return self.create_unified_search_results_visualization(
                final_results,
                self.whole_image["image"],
                None,  # No mask for whole image query
                self.whole_image.get("label", "Whole Query Image"),
                "whole_image"
            )

        except Exception as e:
            error_message = f"Error during local whole image search: {str(e)}"
            print(f"[ERROR] {error_message}")
            import traceback
            traceback.print_exc()
            return None, error_message, gr.update(visible=False), gr.update(choices=[], value=None)

    def create_unified_search_results_visualization(self, filtered_results, query_image, query_mask=None, query_label="Query", query_type="region"):
        """Create visualization of search results with unified display for both region and whole image results"""
        print(f"[STATUS] Creating unified search results visualization for {len(filtered_results)} results...")
        self.search_result_images = []

        # Determine optimal grid layout based on number of results
        num_results = len(filtered_results)
        if num_results <= 3:
            grid_rows = 1
            grid_cols = num_results 
        elif num_results <= 6:
            grid_rows = 2
            grid_cols = 3
        else:
            grid_rows = 3
            grid_cols = 4  # Maximum 12 results displayed well
        
        # Calculate figure size (width, height) with better proportions
        fig_width = min(18, 6 + 3 * grid_cols)  # Limit max width
        fig_height = 4 + 3 * grid_rows
        
        # Prepare figure for visualization
        print(f"[STATUS] Setting up visualization figure with {grid_rows}x{grid_cols} grid...")
        fig = plt.figure(figsize=(fig_width, fig_height))
        
        # Create a layout with a dedicated area for query and proper spacing
        gs = gridspec.GridSpec(grid_rows + 1, grid_cols, height_ratios=[1] + [3] * grid_rows)
        
        # Query section (top row, spans all columns)
        ax_query = fig.add_subplot(gs[0, :])
        
        # Display the query differently based on type
        if query_type == "region" and query_mask is not None:
            # For regions, highlight the region in the query image
            highlighted = query_image.copy()
            mask_3d = np.stack([query_mask, query_mask, query_mask], axis=2)
            highlighted = np.where(mask_3d, np.minimum(highlighted * 1.5, 255), highlighted * 0.4)
            
            # Add contour
            contours, _ = cv2.findContours(
                query_mask.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(highlighted, contours, -1, (0, 255, 0), 2)
            
            ax_query.imshow(highlighted.astype(np.uint8))
            ax_query.set_title(f"Query Region: {query_label}", fontsize=14, pad=10)
        else:
            # For whole images, just show the image
            ax_query.imshow(query_image)
            ax_query.set_title(f"Query: {query_label}", fontsize=14, pad=10)
            
        ax_query.axis('off')

        # Create radio choices for result selection
        radio_choices = []
        result_info = ["# Search Results\n\n"]

        # Results section (grid layout below query)
        print(f"[STATUS] Processing {len(filtered_results)} result images...")
        for i, result in enumerate(filtered_results):
            if i >= grid_rows * grid_cols:  # Limit to grid capacity
                print(f"[STATUS] Only displaying first {grid_rows * grid_cols} results due to space constraints")
                break
                
            # Calculate row and column for this result
            row = (i // grid_cols) + 1  # +1 because query is in row 0
            col = i % grid_cols
            
            print(f"[STATUS] Processing result {i+1}/{len(filtered_results)}, placing at position ({row},{col})...")
            ax = fig.add_subplot(gs[row, col])

            try:
                # Extract metadata
                metadata = result.payload
                image_source = metadata.get("image_source", "Unknown")
                score = result.score
                
                # Get collection type (region or whole_image)
                collection_type = metadata.get("collection_type", "unknown")
                collection_name = metadata.get("collection_name", "unknown")
                
                print(f"[STATUS] Result {i+1} - Source: {os.path.basename(image_source) if isinstance(image_source, str) else 'embedded_image'}, "
                      f"Score: {score:.4f}, Type: {collection_type}")
                
                # Get filename for display
                if isinstance(image_source, str):
                    filename = os.path.basename(image_source)
                else:
                    filename = "embedded_image"
                
                # Load image
                if image_source in self.image_cache:
                    print(f"[STATUS] Using cached image for {filename}")
                    img = self.image_cache[image_source]
                else:
                    print(f"[STATUS] Loading image from {filename}")
                    try:
                        img = np.array(load_local_image(image_source))
                        self.image_cache[image_source] = img
                    except Exception as e:
                        print(f"[WARNING] Could not load image {filename}: {e}")
                        ax.text(0.5, 0.5, f"Error loading image", ha='center', va='center')
                        ax.axis('off')
                        continue

                # Display the result differently based on collection type
                if collection_type == "region":
                    # Extract and display region using bbox
                    bbox = metadata.get("bbox", [0, 0, 100, 100])
                    x_min, y_min, x_max, y_max = bbox
                    
                    # Ensure bbox is within image bounds
                    if x_min >= img.shape[1] or y_min >= img.shape[0]:
                        print(f"[WARNING] Invalid bbox for {filename}: {bbox}")
                        region = img  # Show whole image as fallback
                    else:
                        # Clip values to image bounds
                        x_min = max(0, min(x_min, img.shape[1]-1))
                        y_min = max(0, min(y_min, img.shape[0]-1))
                        x_max = max(x_min+1, min(x_max, img.shape[1]))
                        y_max = max(y_min+1, min(y_max, img.shape[0]))
                        
                        print(f"[STATUS] Extracting region from bbox: {[x_min, y_min, x_max, y_max]}")
                        region = img[y_min:y_max, x_min:x_max]
                    
                    # Display the region with a highlight border
                    ax.imshow(region)
                    
                    # Add a border around the region to make it stand out
                    border_width = 3  
                    for spine in ax.spines.values():
                        spine.set_visible(True)
                        spine.set_linewidth(border_width)
                        spine.set_color('#01579b')  # Blue border color
                    
                    # Get detected class name from metadata (improved labeling)
                    detected_class = metadata.get("detected_class", metadata.get("phrase", "Object"))
                    confidence = metadata.get("confidence", "")
                    confidence_str = f" ({confidence:.2f})" if confidence else ""
                    
                    # Create a badge showing it's a region with the detected class
                    badge_props = dict(boxstyle="round,pad=0.3", fc="#e1f5fe", ec="#01579b", alpha=0.8)
                    badge_text = f"{detected_class.upper()}"
                    ax.text(0.04, 0.04, badge_text, transform=ax.transAxes, 
                            fontsize=9, weight='bold', color='#01579b',
                            verticalalignment='top', horizontalalignment='left',
                            bbox=badge_props)
                    
                    # Store region for possible enlargement
                    self.search_result_images.append({
                        "image": region,
                        "bbox": bbox,
                        "metadata": {
                            "image_source": image_source,
                            "filename": filename,
                            "detected_class": detected_class,
                            "confidence": confidence,
                            "score": score,
                            "type": "region",
                            "collection_name": collection_name
                        }
                    })
                    
                    # Create radio choice label with actual detected class
                    choice_label = f"Result {i+1}: {filename} - {detected_class}{confidence_str}"
                    
                    # Add to result info text with better details
                    result_info.append(f"**Result {i+1}:** {filename}\n- Type: Region ({detected_class})\n- Score: {score:.4f}\n")
                    
                else:  # whole_image
                    # Display the whole image
                    ax.imshow(img)
                    
                    # Create a badge showing it's a whole image
                    badge_props = dict(boxstyle="round,pad=0.3", fc="#e8f5e9", ec="#2e7d32", alpha=0.8)
                    badge_text = "WHOLE IMAGE"
                    ax.text(0.04, 0.04, badge_text, transform=ax.transAxes, 
                            fontsize=9, weight='bold', color='#2e7d32',
                            verticalalignment='top', horizontalalignment='left',
                            bbox=badge_props)
                    
                    # Store whole image for possible enlargement
                    self.search_result_images.append({
                        "image": img,
                        "metadata": {
                            "image_source": image_source,
                            "filename": filename,
                            "detected_class": "whole_image",
                            "score": score,
                            "type": "whole_image",
                            "collection_name": collection_name
                        }
                    })
                    
                    # Create radio choice label
                    choice_label = f"Result {i+1}: {filename} (Whole Image)"
                    
                    # Add to result info text
                    result_info.append(f"**Result {i+1}:** {filename}\n- Type: Whole Image\n- Score: {score:.4f}\n")
                
                # Add to radio choices
                radio_choices.append(choice_label)
                
                # Create a clearer title with result number and score
                title = f"Result {i+1}: {score:.2f}"
                ax.set_title(title, fontsize=12, pad=5)
                
                # Add filename in a less obtrusive way - smaller text at the bottom
                ax.text(0.5, 0.03, f"{filename}", transform=ax.transAxes, 
                        ha='center', va='bottom', fontsize=8, 
                        bbox=dict(facecolor='white', alpha=0.6, pad=2,
                                 edgecolor='lightgray', boxstyle='round'))
                ax.axis('off')

            except Exception as e:
                print(f"[ERROR] Error displaying result {i+1}: {e}")
                import traceback
                traceback.print_exc()
                ax.text(0.5, 0.5, f"Error loading image {i+1}", ha='center', va='center')
                ax.axis('off')
                self.search_result_images.append(None)
                continue

        # Adjust layout to prevent overlapping and provide better spacing
        plt.tight_layout(pad=2.0)
        fig.subplots_adjust(wspace=0.3, hspace=0.3)

        print(f"[STATUS] Generating final visualization image...")
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150)
        buf.seek(0)
        plt.close(fig)

        results_image = Image.open(buf)
        print(f"[STATUS] Search results visualization complete with {len(radio_choices)} result options")
        return results_image, "\n".join(result_info), gr.update(visible=True), gr.update(choices=radio_choices, value=None)

    def display_enlarged_result(self, result_selection):
        """Displays an enlarged version of the selected search result with region highlighted in full image"""
        if result_selection is None or not self.search_result_images:
            return None
            
        try:
            # Extract result index from selection
            result_idx = int(result_selection.split(":")[0].replace("Result ", "")) - 1
            
            if result_idx < 0 or result_idx >= len(self.search_result_images) or self.search_result_images[result_idx] is None:
                return None
                
            # Get the selected result data
            result_data = self.search_result_images[result_idx]
            
            # Check if result_data is a dictionary (expected format)
            if not isinstance(result_data, dict):
                print(f"[WARNING] Unexpected result_data type: {type(result_data)}")
                return None
                
            # Safely handle metadata
            if "metadata" not in result_data:
                print(f"[WARNING] No 'metadata' key in result_data: {list(result_data.keys())}")
                # Create default metadata
                metadata = {
                    "filename": f"Result {result_idx+1}",
                    "phrase": "Unknown",
                    "score": 0.0,
                    "image_source": None
                }
            else:
                metadata = result_data["metadata"]
                
            # Get metadata values safely with defaults
            filename = metadata.get("filename", f"Result {result_idx+1}")
            phrase = metadata.get("detected_class", "Unknown") # Changed "phrase" to "detected_class"
            score = metadata.get("score", 0.0)
            image_source = metadata.get("image_source", None)
            result_type = metadata.get("type", "region")
            
            # Create a more visually appealing enlarged display
            fig = plt.figure(figsize=(12, 10), constrained_layout=True)
            gs = gridspec.GridSpec(1, 1, figure=fig)
            ax = fig.add_subplot(gs[0, 0])
            
            # Load the full original image if we have a source
            if image_source and os.path.exists(image_source):
                try:
                    # Load the full original image
                    full_image = np.array(load_local_image(image_source))
                    
                    if result_type == "region":
                        # Get bounding box if available
                        bbox = result_data.get("bbox", None)
                        
                        # Create a copy of the image to draw on
                        display_image = full_image.copy()
                        
                        # If we have a valid bounding box, highlight the region
                        if bbox and len(bbox) == 4:
                            x_min, y_min, x_max, y_max = bbox
                            
                            # Ensure bbox is within image bounds
                            x_min = max(0, min(x_min, display_image.shape[1]-1))
                            y_min = max(0, min(y_min, display_image.shape[0]-1))
                            x_max = max(x_min+1, min(x_max, display_image.shape[1]))
                            y_max = max(y_min+1, min(y_max, display_image.shape[0]))
                            
                            # Draw rectangle around region
                            cv2.rectangle(display_image, 
                                         (int(x_min), int(y_min)), 
                                         (int(x_max), int(y_max)), 
                                         (0, 255, 0), 3)
                                         
                            # Highlight the region slightly
                            alpha = 0.3
                            roi = display_image[y_min:y_max, x_min:x_max]
                            highlighted_roi = np.clip(roi * 1.3, 0, 255).astype(np.uint8)
                            display_image[y_min:y_max, x_min:x_max] = cv2.addWeighted(
                                highlighted_roi, alpha, roi, 1-alpha, 0)
                    else:
                        # For whole image matches, just use the full image
                        display_image = full_image
                        
                    # Show the full image with highlighted region
                    ax.imshow(display_image)
                except Exception as e:
                    print(f"[WARNING] Error loading full image: {e}")
                    # Fallback to the region image
                    region = result_data.get("image", None)
                    if region is not None:
                        ax.imshow(region)
            else:
                # If no source image available, use the region image
                region = result_data.get("image", None)
                if region is not None:
                    ax.imshow(region)
                else:
                    ax.text(0.5, 0.5, "Image not available", 
                           ha='center', va='center', fontsize=14)
            
            ax.axis('off')
            
            # Add a comprehensive, well-formatted title
            title = f"Result {result_idx+1}: {filename}"
            ax.set_title(title, fontsize=16, pad=15)
            
            # Add detailed metadata information in a visually appealing box
            info_text = (
                f"Phrase: {phrase}\n"
                f"Score: {score:.4f}" if isinstance(score, (float, int)) else f"Score: {score}"
            )
            
            # Create a text box with better styling
            props = dict(boxstyle='round,pad=1', facecolor='white', alpha=0.8, edgecolor='lightgray')
            ax.text(0.5, 0.03, info_text, transform=ax.transAxes, 
                    ha='center', va='bottom', fontsize=12,
                    bbox=props, linespacing=1.5)
            
            # Convert the figure to an image
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=200)  # Higher DPI for better quality
            buf.seek(0)
            plt.close(fig)
            
            return Image.open(buf)
            
        except Exception as e:
            print(f"[ERROR] Error displaying enlarged result: {e}")
            import traceback
            traceback.print_exc()
            return None

    def build_interface(self):
        """Build the Gradio interface"""
        # Create the interface with tabs for different modes
        with gr.Blocks(title="Grounded SAM Region Search") as demo:
            gr.Markdown("# 🔍 Grounded SAM Region Search")
            gr.Markdown("Upload an image, detect regions, and search for similar regions in your image database.")

            # Processing mode selector (new)
            gr.Markdown("## Processing Mode")
            with gr.Row():
                processing_mode = gr.Radio(
                    choices=["Region Detection (GroundedSAM + PE)", "Whole Image (PE Only)"],
                    value="Region Detection (GroundedSAM + PE)",
                    label="Select Processing Mode"
                )
            
            # Common file upload for both modes
            with gr.Row():
                input_image = gr.Image(type="pil", label="Upload Image")
            
            # Region-based mode components
            with gr.Group(visible=True) as region_mode_group:
                with gr.Row():
                    text_prompt = gr.Textbox(
                        placeholder="Examples: 'person . car . building .' OR 'all the chairs in the room' OR 'a person with pink clothes'",
                        label="Detection Prompts",
                        info="💡 Use period-separated for multiple objects (person . car . building .) OR natural language for specific descriptions (all the chairs, a person with red shirt)",
                        value="person . car . building ."
                    )
                
                # Add parameter controls before the process button
                    gr.Markdown("#### Advanced PE Configuration") # Title for the new section
                    extraction_mode_radio = gr.Radio(
                        choices=["single", "preset", "multi_layer"],
                        value="single",
                        label="Feature Extraction Mode",
                        interactive=True
                    )
                    
                    with gr.Group(visible=True) as single_options_group: # Visible by default
                        optimal_layer_slider_ui = gr.Slider( 
                            minimum=1, maximum=47, value=40, step=1, 
                            label="PE Layer (for Single Mode)",
                            info="Layer of the PE-ViT model to use for 'single' mode.",
                            interactive=True
                        )

                    with gr.Group(visible=False) as preset_options_group:
                        preset_name_dropdown_ui = gr.Dropdown(
                            choices=["object_focused", "spatial_focused", "semantic_focused"],
                            value="object_focused",
                            label="Preset Configuration",
                            info="Select a predefined multi-layer configuration.",
                            interactive=True
                        )

                    with gr.Group(visible=False) as custom_options_group:
                        gr.Markdown("Define up to 3 layers and their weights for 'multi_layer' mode. Ensure weights roughly sum to 1 or they will be normalized.")
                        custom_layers_sliders_ui = []
                        custom_weights_sliders_ui = []
                        default_custom_layers = [30, 40, 47] # Default example values
                        default_custom_weights = [0.3, 0.4, 0.3] # Default example values
                        for i in range(3): 
                            with gr.Row():
                                with gr.Column(scale=2):
                                    layer_slider = gr.Slider(
                                        minimum=1, maximum=47, value=default_custom_layers[i], step=1,
                                        label=f"Custom Layer {i+1}",
                                        interactive=True
                                    )
                                    custom_layers_sliders_ui.append(layer_slider)
                                with gr.Column(scale=1):
                                    weight_slider = gr.Slider(
                                        minimum=0.0, maximum=1.0, value=default_custom_weights[i], step=0.01,
                                        label=f"Weight {i+1}",
                                        interactive=True
                                    )
                                    custom_weights_sliders_ui.append(weight_slider)
                    
                    gr.Markdown("#### Detection & Pooling Parameters") # Title for existing and new pooling params
                with gr.Row():
                    with gr.Column():
                            min_area_ratio = gr.Slider( # Existing parameter
                            minimum=0.001, maximum=0.1, value=0.01, step=0.001,
                            label="Minimum Area Ratio",
                            info="Minimum size of regions to detect (as a fraction of image size)"
                        )
                    with gr.Column():
                            max_regions = gr.Slider( # Existing parameter
                            minimum=1, maximum=20, value=5, step=1,
                            label="Maximum Regions",
                            info="Maximum number of regions to detect per image"
                        )
                    
                    with gr.Row():
                        pooling_strategy_dropdown_ui = gr.Dropdown(
                            choices=["top_k", "pe_attention", "pe_spatial", "pe_semantic", "pe_adaptive", "average", "max", "attention"],
                            value="top_k", # Default from process_image_with_prompt
                            label="Pooling Strategy",
                            interactive=True
                        )
                        temperature_slider_ui = gr.Slider(
                            minimum=0.01, maximum=1.0, value=0.07, step=0.01,
                            label="Temperature (pooling/combination)",
                            interactive=True
                        )
                    
                    with gr.Row():
                        top_k_ratio_slider_ui = gr.Slider(
                            minimum=0.01, maximum=1.0, value=0.1, step=0.01,
                            label="Top-K Ratio (for 'top_k' pooling)",
                            interactive=True
                        )
                        attention_heads_slider_ui = gr.Slider(
                            minimum=1, maximum=16, value=8, step=1,
                            label="Attention Heads (for 'pe_attention')",
                            interactive=True
                        )

                with gr.Row():
                    process_button = gr.Button("Detect Regions", variant="primary")
            
            # Whole-image mode components
            with gr.Group(visible=False) as whole_image_mode_group:
                # Add embedding layer for whole image processing too
                with gr.Row():
                    whole_image_layer = gr.Slider(
                        minimum=1, maximum=50, value=40, step=1,
                        label="Embedding Layer",
                        info="Layer of the Perception Encoder to use for processing"
                    )
                with gr.Row():
                    process_whole_button = gr.Button("Process Whole Image", variant="primary")
            
            # Status and results area
            with gr.Row():
                with gr.Column():
                    # Results for region-based mode
                    with gr.Group(visible=True) as region_results_group:
                        detection_info = gr.Textbox(label="Detection Status")
                        segmented_output = gr.Image(type="pil", label="Detected Regions")
                        region_dropdown = gr.Dropdown(choices=[], label="Select a Region")
                        region_preview = gr.Image(type="pil", label="Region Preview")
                        
                        with gr.Row():
                            with gr.Column():
                                similarity_slider = gr.Slider(
                                    minimum=0.0, maximum=1.0, value=0.5, step=0.01,
                                    label="Similarity Threshold"
                                )
                            with gr.Column():
                                max_results_dropdown = gr.Dropdown(
                                    choices=["5", "10", "20", "50"], value="5",
                                    label="Max Results"
                                )
                        
                        # Remove the embedding layer slider from here since we moved it above
                        search_button = gr.Button("Search Similar Regions", variant="primary")
                    
                    # Results for whole-image mode
                    with gr.Group(visible=False) as whole_image_results_group:
                        whole_image_info = gr.Textbox(label="Processing Status")
                        processed_output = gr.Image(type="pil", label="Processed Image")
                        
                        with gr.Row():
                            with gr.Column():
                                whole_similarity_slider = gr.Slider(
                                    minimum=0.0, maximum=1.0, value=0.5, step=0.01,
                                    label="Similarity Threshold"
                                )
                            with gr.Column():
                                whole_max_results_dropdown = gr.Dropdown(
                                    choices=["5", "10", "20", "50"], value="5",
                                    label="Max Results"
                                )
                                    
                        whole_search_button = gr.Button("Search Similar Images", variant="primary")
                    
                    # Common search results area
                    search_info = gr.Textbox(label="Search Status")
                    search_results_output = gr.Image(type="pil", label="Search Results")
                    
                    with gr.Group(visible=False) as button_section:
                        result_selector = gr.Dropdown(choices=[], label="Select Result to View")
                    
                    enlarged_result = gr.Image(type="pil", label="Enlarged Result")
            
            # Function to toggle visibility based on mode
            def toggle_mode(mode):
                if mode == "Region Detection (GroundedSAM + PE)":
                    return gr.update(visible=True), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
                else:  # Whole Image mode
                    return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=True)
            
            # Connect mode selector to toggle visibility
            processing_mode.change(
                toggle_mode,
                inputs=[processing_mode],
                outputs=[
                    region_mode_group, 
                    whole_image_mode_group,
                    region_results_group,
                    whole_image_results_group
                ]
            )
            
            # Connect buttons to functions for region-based processing
            
            # Define the visibility update function for extraction modes
            def update_extraction_visibility(mode_value):
                return {
                    single_options_group: gr.update(visible=(mode_value == "single")),
                    preset_options_group: gr.update(visible=(mode_value == "preset")),
                    custom_options_group: gr.update(visible=(mode_value == "multi_layer")),
                }

            extraction_mode_radio.change(
                fn=update_extraction_visibility,
                inputs=[extraction_mode_radio],
                outputs=[single_options_group, preset_options_group, custom_options_group]
            )

            process_button.click(
                self.process_image_with_prompt,
                inputs=[
                    input_image, text_prompt, 
                    min_area_ratio, max_regions, # Existing params for detection
                    extraction_mode_radio,         # New
                    optimal_layer_slider_ui,       # New (replaces old embedding_layer)
                    preset_name_dropdown_ui,       # New
                    custom_layers_sliders_ui[0], custom_weights_sliders_ui[0], # New
                    custom_layers_sliders_ui[1], custom_weights_sliders_ui[1], # New
                    custom_layers_sliders_ui[2], custom_weights_sliders_ui[2], # New
                    pooling_strategy_dropdown_ui,  # New
                    temperature_slider_ui,         # New
                    top_k_ratio_slider_ui,         # New
                    attention_heads_slider_ui      # New
                ],
                outputs=[segmented_output, detection_info, region_dropdown, region_preview]
            )

            region_dropdown.change(
                self.update_region_preview,
                inputs=[region_dropdown],
                outputs=[region_preview]
            )

            search_button.click(
                self.search_region,
                # Pass temperature_slider_ui. Note: embedding_layer was removed from search_region's signature.
                inputs=[region_dropdown, similarity_slider, max_results_dropdown, temperature_slider_ui],
                outputs=[search_results_output, search_info, button_section, result_selector]
            )
            
            # Connect buttons to functions for whole-image processing
            # (process_whole_button's call to self.process_whole_image does not need temperature for query generation itself,
            # but extract_whole_image_embeddings it calls *does* accept temperature for multi-layer combination.
            # However, the UI for whole_image_mode_group currently only has whole_image_layer, not the full advanced PE config block.
            # This is consistent with the prompt focusing temperature on the *search* part for whole images.)
            process_whole_button.click(
                self.process_whole_image, # process_whole_image itself doesn't take temperature_ui directly for its own logic
                                         # but the extract_whole_image_embeddings it calls has a temperature parameter.
                                         # The UI doesn't currently expose a specific temperature for *processing* the whole image query if it were multi-layer.
                                         # This is fine as per prompt's focus on search similarity.
                inputs=[input_image, whole_image_layer], # Assuming whole_image_layer is optimal_layer for single, or used in preset/multi if UI was extended
                outputs=[processed_output, whole_image_info]
            )
            
            whole_search_button.click(
                self.search_whole_image,
                # Pass temperature_slider_ui. Note: whole_image_layer (as optimal_layer) was removed from search_whole_image's signature.
                inputs=[whole_similarity_slider, whole_max_results_dropdown, temperature_slider_ui],
                outputs=[search_results_output, search_info, button_section, result_selector]
            )
            
            # Connect result selector for both modes
            result_selector.change(
                self.display_enlarged_result,
                inputs=[result_selector],
                outputs=[enlarged_result]
            )

        return demo

    def process_whole_image(self, image, optimal_layer=40):
        """Process an uploaded image using only PE Encoder without region detection"""
        if isinstance(image, np.ndarray):
            image_pil = Image.fromarray(image)
        else:
            image_pil = image

        print(f"[STATUS] Processing whole image with embedding layer: {optimal_layer}")
            
        # Extract whole image embedding
        image_np, embedding, metadata, label = extract_whole_image_embeddings(
            image_pil,
            pe_model_param=self.pe_model,
            pe_vit_model_param=self.pe_vit_model,
            preprocess_param=self.preprocess,
            device_param=self.device,
            optimal_layer=optimal_layer
        )

        # Store results
        self.whole_image = {
            "image": image_np,
            "embedding": embedding,
            "metadata": metadata,
            "label": label,
            "layer_used": metadata.get("layer_used", optimal_layer) if metadata else optimal_layer
        }

        if embedding is None:
            return image_np, "Failed to process image"

        # Create a simple visualization with border
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(image_np)
        
        # Use the actual layer from metadata if available
        actual_layer = metadata.get("layer_used", optimal_layer) if metadata else optimal_layer
        ax.set_title(f"Whole Image Processed (Layer {actual_layer})")
        
        # Add a green border to show it's been processed
        border_width = 5
        plt.gca().spines['top'].set_linewidth(border_width)
        plt.gca().spines['bottom'].set_linewidth(border_width)
        plt.gca().spines['left'].set_linewidth(border_width)
        plt.gca().spines['right'].set_linewidth(border_width)
        plt.gca().spines['top'].set_color('green')
        plt.gca().spines['bottom'].set_color('green')
        plt.gca().spines['left'].set_color('green')
        plt.gca().spines['right'].set_color('green')
        ax.axis('on')  # Show axis for border visibility
        ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
        
        # Save figure to buffer
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)

        processed_image = Image.open(buf)
        
        # Use actual layer in the response message too
        return processed_image, f"Image processed successfully using layer {actual_layer}"

    def update_region_preview(self, region_selection):
        """Updates the region preview when a region is selected"""
        if region_selection is None or self.detected_regions["image"] is None:
            return None

        region_idx = int(region_selection.split(":")[0].replace("Region ", "")) - 1
        mask = self.detected_regions["masks"][region_idx]
        label = self.detected_regions["labels"][region_idx]

        return self.create_region_preview(self.detected_regions["image"], mask, label)

# =============================================================================
# MULTI-MODE INTERFACE
# =============================================================================

def create_multi_mode_interface(pe_model, pe_vit_model, preprocess, device):
    """Create a multi-mode interface with both region-based and whole-image processing"""
    # Initialize the app state to track databases
    app_state = AppState()
    
    # Initialize the interface with the models
    interface = GradioInterface(pe_model, pe_vit_model, preprocess, device)
    
    # Create a comprehensive multi-mode Gradio interface with tabs
    with gr.Blocks(title="Revers-o: GroundedSAM + Perception Encoders Image Similarity Search") as demo:
        gr.Markdown("# 🔍 Revers-o: GroundedSAM + Perception Encoders Image Similarity Search")
        
        # Create mode tabs
        with gr.Tabs() as tabs:
            # Tab 0: Quick Start Instructions
            with gr.TabItem("Quick Start"):
                gr.Markdown("## 🚀 Quick Start Guide")
                gr.Markdown("**New to Revers-o? Follow these simple steps to get started:**")
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("""
                        ### 1. 🎥 Extract Images from Video (Optional)
                        If you have videos to analyze:
                        - Go to the **"Extract Images"** tab
                        - Enter your video folder path
                        - Set output folder for extracted frames
                        - Click "Extract Images from Videos"
                        
                        ### 2. 🗃️ Create Database
                        - Go to the **"Create Database"** tab
                        - Enter path to your images folder
                        - Choose detection prompts (e.g., "person . car . building")
                        - Click "Process Folder & Create Database"
                        
                        ### 3. 🔍 Search for Matches
                        - Go to the **"Search Database"** tab
                        - Upload an image containing what you want to find
                        - Describe what to look for in the image
                        - Click "Detect Regions" then "Search Similar Regions"
                        """)
                    
                    with gr.Column():
                        gr.Markdown("""
                        ### 💡 Tips for Best Results
                        
                        **Good Detection Prompts:**
                        - "person wearing uniform"
                        - "vehicle with license plate"
                        - "building with red door"
                        - "crowd of people"
                        
                        **Processing Modes:**
                        - **Region Detection**: Find specific objects/people
                        - **Whole Image**: Find similar scenes/compositions
                        
                        **Parameter Guidelines:**
                        - **Similarity Threshold**: 0.3-0.7 for most searches
                        - **Embedding Layer**: 40 is optimal for most cases
                        - **Max Regions**: 5-10 for detailed analysis
                        
                        **⚠️ Start Small:** Test with a few images first to understand how the system works before processing large collections.
                        """)
                
                gr.Markdown("---")
                gr.Markdown("### 📖 Need More Help?")
                gr.Markdown("Check the **About** tab for detailed documentation and troubleshooting tips.")
            
            # Tab 1: Extract Images from Video
            with gr.TabItem("Extract Images"):
                gr.Markdown("## 🎬 Extract Images from Video")
                gr.Markdown("Process videos to extract keyframes and scenes that can be used to create image databases.")
                
                if VIDEO_PROCESSING_AVAILABLE:
                    # Add tabs for different input methods
                    with gr.Tabs():
                        # Tab for folder-based processing (existing functionality)
                        with gr.TabItem("📁 From Local Files"):
                            with gr.Row():
                                with gr.Column():
                                    video_input_folder = gr.Textbox(
                                        label="Video Folder Path", 
                                        placeholder="/path/to/videos",
                                        info="Folder containing video files (.mp4, .avi, .mov, .mkv, .wmv, .flv, .webm, .m4v)"
                                    )
                                    video_output_folder = gr.Textbox(
                                        label="Output Folder Path", 
                                        placeholder="/path/to/extracted/images",
                                        info="Folder where extracted images will be saved"
                                    )
                                    
                                    with gr.Row():
                                        with gr.Column():
                                            max_frames_per_video = gr.Slider(
                                                minimum=5, maximum=100, value=30, step=5,
                                                label="Max Frames per Video",
                                                info="Maximum number of keyframes to extract from each video"
                                            )
                                        with gr.Column():
                                            scene_threshold = gr.Slider(
                                                minimum=10.0, maximum=50.0, value=30.0, step=5.0,
                                                label="Scene Detection Sensitivity",
                                                info="Lower values = more sensitive scene detection (more scenes)"
                                            )
                                    
                                    extract_video_button = gr.Button("🎬 Extract Images from Videos", variant="primary")
                                
                                with gr.Column():
                                    video_progress = gr.Textbox(
                                        label="Video Processing Status",
                                        lines=10,
                                        max_lines=15,
                                        info="Real-time progress updates will appear here"
                                    )
                        
                        # Tab for URL-based processing (new functionality)
                        with gr.TabItem("🔗 From URLs"):
                            if YT_DLP_AVAILABLE:
                                with gr.Row():
                                    with gr.Column():
                                        gr.Markdown("### Supported: youtube, twitter, facebook, instagram, tiktok, vimeo, some others")                                        
                                        video_urls_input = gr.Textbox(
                                            label="Video URLs",
                                            lines=5,
                                            placeholder="https://www.youtube.com/watch?v=...\nhttps://twitter.com/user/status/...\nhttps://www.tiktok.com/@user/video/...\n\n(Enter one URL per line)",
                                            info="Enter video URLs, one per line. Supports YouTube, Twitter, Facebook, Instagram, TikTok, Vimeo, and more."
                                        )
                                        
                                        url_output_folder = gr.Textbox(
                                            label="Output Folder Path",
                                            placeholder="/path/to/extracted/images",
                                            info="Folder where extracted images will be saved"
                                        )
                                        
                                        with gr.Row():
                                            with gr.Column():
                                                url_max_frames = gr.Slider(
                                                    minimum=5, maximum=100, value=30, step=5,
                                                    label="Max Frames per Video",
                                                    info="Maximum number of keyframes to extract from each video"
                                                )
                                            with gr.Column():
                                                url_scene_threshold = gr.Slider(
                                                    minimum=10.0, maximum=50.0, value=30.0, step=5.0,
                                                    label="Scene Detection Sensitivity",
                                                    info="Lower values = more sensitive scene detection"
                                                )
                                        
                                        with gr.Row():
                                            with gr.Column():
                                                video_quality = gr.Dropdown(
                                                    choices=["480p", "720p", "1080p", "best"],
                                                    value="720p",
                                                    label="Video Quality",
                                                    info="Maximum video quality to download (higher quality = larger files)"
                                                )
                                        
                                        extract_urls_button = gr.Button("🔗 Download & Extract Images from URLs", variant="primary")
                                    
                                    with gr.Column():
                                        url_progress = gr.Textbox(
                                            label="URL Processing Status",
                                            lines=10,
                                            max_lines=15,
                                            info="Download and processing updates will appear here"
                                        )
                            else:
                                gr.Markdown("⚠️ **URL video downloading not available.**")
                                gr.Markdown("Please install yt-dlp to enable URL downloads:")
                                gr.Code("pip install yt-dlp", language="bash")
                    
                    # Video processing function with progress updates (existing)
                    def process_videos_with_progress(input_folder, output_folder, max_frames, scene_thresh):
                        """Process videos and yield progress updates"""
                        if not input_folder or not output_folder:
                            yield "❌ Please specify both input and output folder paths"
                            return
                        
                        if not os.path.exists(input_folder):
                            yield f"❌ Input folder does not exist: {input_folder}"
                            return
                        
                        try:
                            # Create output folder if it doesn't exist
                            os.makedirs(output_folder, exist_ok=True)
                            yield f"📁 Created output folder: {output_folder}"
                            
                            # Process videos
                            for progress_msg in process_video_folder(input_folder, output_folder, max_frames, scene_thresh):
                                yield progress_msg
                                
                        except Exception as e:
                            yield f"❌ Error during video processing: {str(e)}"
                    
                    # URL processing function with progress updates (new)
                    def process_urls_with_progress(urls_text, output_folder, max_frames, scene_thresh, quality):
                        """Process URLs and yield progress updates"""
                        if not urls_text or not urls_text.strip():
                            yield "❌ Please provide at least one video URL"
                            return
                        
                        if not output_folder:
                            yield "❌ Please specify an output folder path"
                            return
                        
                        try:
                            # Create output folder if it doesn't exist
                            os.makedirs(output_folder, exist_ok=True)
                            yield f"📁 Created output folder: {output_folder}"
                            
                            # Process URLs
                            for progress_msg in process_video_urls(urls_text, output_folder, max_frames, scene_thresh, quality):
                                yield progress_msg
                                
                        except Exception as e:
                            yield f"❌ Error during URL processing: {str(e)}"
                    
                    # Connect existing video processing button
                    extract_video_button.click(
                        process_videos_with_progress,
                        inputs=[video_input_folder, video_output_folder, max_frames_per_video, scene_threshold],
                        outputs=[video_progress]
                    )
                    
                    # Connect new URL processing button (only if yt-dlp is available)
                    if YT_DLP_AVAILABLE:
                        extract_urls_button.click(
                            process_urls_with_progress,
                            inputs=[video_urls_input, url_output_folder, url_max_frames, url_scene_threshold, video_quality],
                            outputs=[url_progress]
                        )
                else:
                    gr.Markdown("⚠️ **Video processing not available.** Please install required dependencies:")
                    gr.Code("pip install scenedetect imageio imageio-ffmpeg", language="bash")
            
            # Tab 2: Create Database
            with gr.TabItem("Create Database"):
                gr.Markdown("## Create a New Database")
                gr.Markdown("Process images in a folder to create a searchable database.")
                
                with gr.Row():
                    with gr.Column():
                        db_folder_path = gr.Textbox(
                            label="Image Folder Path(s)", 
                            placeholder="/path/to/images1, /path/to/images2, /path/to/images3",
                            info="💡 Enter a single folder path OR multiple folder paths separated by commas"
                        )
                        db_collection_name = gr.Textbox(label="Database Name", value="my_image_database", 
                                                      placeholder="Name for your database collection")
                        
                        db_mode = gr.Radio(
                            choices=["Region Detection (GroundedSAM + PE)", "Whole Image (PE Only)"],
                            value="Region Detection (GroundedSAM + PE)",
                            label="Database Processing Mode"
                        )
                        
                        # Region-specific options
                        with gr.Group(visible=True) as region_options_group:
                            region_prompts = gr.Textbox(
                                placeholder="Examples: 'person . car . building .' OR 'all the chairs in the room' OR 'a person with pink clothes'",
                                label="Detection Prompts",
                                info="💡 Use period-separated for multiple objects (person . car . building .) OR natural language for specific descriptions (all the chairs, a person with red shirt)",
                                value="person . car . building ."
                            )
                            
                            # Add pooling strategy selection
                            pooling_strategy = gr.Dropdown(
                                choices=["top_k", "max", "attention", "average"],
                                value="top_k",
                                label="Feature Pooling Strategy",
                                info="How to combine spatial features from detected regions"
                            )
                            
                            # Add region detection parameters
                            with gr.Row():
                                with gr.Column():
                                    min_area_ratio = gr.Slider(
                                        minimum=0.001, maximum=0.1, value=0.01, step=0.001,
                                        label="Minimum Area Ratio",
                                        info="Minimum size of regions to detect (as a fraction of image size)"
                                    )
                                with gr.Column():
                                    max_regions = gr.Slider(
                                        minimum=1, maximum=20, value=5, step=1,
                                        label="Maximum Regions",
                                        info="Maximum number of regions to detect per image"
                                    )
                            
                        # Common options for both modes
                        with gr.Row():
                            with gr.Column():
                                layer_slider = gr.Slider(
                                    minimum=1, maximum=50, value=40, step=1,
                                    label="Embedding Layer",
                                    info="Layer of the Perception Encoder to use (1-50)"
                                )
                        
                        # Process button
                        create_db_button = gr.Button("Process Folder & Create Database", variant="primary")
                    
                    with gr.Column():
                        # Display the logo image
                        # gr.Image("big_logo.png", show_label=False, container=False)
                        
                        db_progress = gr.Textbox(label="Processing Status")
                        db_done_msg = gr.Markdown(visible=False)
                
                # Add section for database verification and repair
                gr.Markdown("## Database Maintenance", elem_id="db_maintenance")
                gr.Markdown("Verify and repair existing databases if you encounter issues with region detection or search quality.")
                
                with gr.Row():
                    with gr.Column():
                        verify_db_dropdown = gr.Dropdown(
                            choices=app_state.available_databases if app_state.available_databases else ["No databases found"],
                            value=app_state.available_databases[0] if app_state.available_databases else None,
                            label="Select Database to Verify/Repair"
                        )
                    with gr.Column():
                        force_rebuild_checkbox = gr.Checkbox(
                            value=False,
                            label="Force Complete Rebuild",
                            info="Enable this to force a complete rebuild of the database (slower but more thorough)"
                        )
                
                with gr.Row():
                    with gr.Column():
                        verify_db_button = gr.Button("Verify Database", variant="secondary")
                    with gr.Column():
                        repair_db_button = gr.Button("Repair Database", variant="primary")
                
                verify_repair_status = gr.Textbox(label="Verification/Repair Status")
                        
                # Function to toggle region options based on mode
                def toggle_region_options(mode):
                    if mode == "Region Detection (GroundedSAM + PE)":
                        return gr.update(visible=True)
                    else:
                        return gr.update(visible=False)
                
                db_mode.change(toggle_region_options, inputs=[db_mode], outputs=[region_options_group])
                
                # Connect database verification functions
                def verify_database(collection_name):
                    """Verify the selected database"""
                    if collection_name and collection_name != "No databases found":
                        print(f"[STATUS] Verifying database: {collection_name}")
                        success, message = verify_repair_database(collection_name, force_rebuild=False)
                        if success:
                            return f"✅ {message}"
                        else:
                            return f"❌ {message}"
                    else:
                        return "No database selected for verification"
                
                def repair_database(collection_name, force_rebuild):
                    """Repair the selected database"""
                    if collection_name and collection_name != "No databases found":
                        print(f"[STATUS] Repairing database: {collection_name} (force_rebuild={force_rebuild})")
                        success, message = verify_repair_database(collection_name, force_rebuild=force_rebuild)
                        
                        # Refresh databases after repair
                        app_state.update_available_databases(verbose=True)
                        
                        if success:
                            return f"✅ {message}"
                        else:
                            return f"❌ {message}"
                    else:
                        return "No database selected for repair"
                
                # Connect verification buttons
                verify_db_button.click(
                    verify_database,
                    inputs=[verify_db_dropdown],
                    outputs=[verify_repair_status]
                )
                
                repair_db_button.click(
                    repair_database,
                    inputs=[verify_db_dropdown, force_rebuild_checkbox],
                    outputs=[verify_repair_status]
                )
                
                # Refresh button for database verification dropdown
                def refresh_verify_databases():
                    """Refresh the database list for verification"""
                    app_state.update_available_databases(verbose=True)
                    if app_state.available_databases:
                        return gr.Dropdown(choices=app_state.available_databases, value=app_state.available_databases[0])
                    else:
                        return gr.Dropdown(choices=["No databases found"], value=None)
                
                # Connect build database functions
                def process_folder_with_mode(folder_path, collection_name, prompts, layer, mode, min_area_ratio=0.01, max_regions=5, pooling_strategy="top_k"):
                    """Process folder(s) based on selected mode - supports single folder or comma-separated multiple folders"""
                    print(f"[STATUS] Starting folder processing with mode: {mode}")
                    print(f"[STATUS] Using pooling strategy: {pooling_strategy}")
                    print(f"[STATUS] Input folder path(s): {folder_path}")
                    
                    # Check if multiple folders are provided (contains comma)
                    is_multiple_folders = ',' in folder_path
                    
                    if is_multiple_folders:
                        print(f"[STATUS] Detected multiple folder paths, using multi-folder processing")
                        
                        if mode == "Region Detection (GroundedSAM + PE)":
                            # Modify collection name to include mode and layer
                            collection_name = f"{collection_name}_layer{layer}"
                            print(f"[STATUS] Using region collection name: {collection_name}")
                            # Process with region detection for multiple folders
                            gen = process_multiple_folders_with_progress_advanced(
                                folder_path, 
                                prompts, 
                                collection_name,
                                optimal_layer=layer,
                                min_area_ratio=min_area_ratio,
                                max_regions=max_regions,
                                pooling_strategy=pooling_strategy
                            )
                        else:
                            # Modify collection name to include mode and layer
                            collection_name = f"{collection_name}_whole_images_layer{layer}"
                            print(f"[STATUS] Using whole image collection name: {collection_name}")
                            # Process with whole image for multiple folders
                            gen = process_multiple_folders_with_progress_whole_images(
                                folder_path,
                                collection_name,
                                optimal_layer=layer
                            )
                    else:
                        print(f"[STATUS] Detected single folder path, using single-folder processing")
                        
                        if mode == "Region Detection (GroundedSAM + PE)":
                            # Modify collection name to include mode and layer
                            collection_name = f"{collection_name}_layer{layer}"
                            print(f"[STATUS] Using region collection name: {collection_name}")
                            # Process with region detection for single folder
                            gen = process_folder_with_progress_advanced(
                                folder_path, 
                                prompts, 
                                collection_name,
                                optimal_layer=layer,
                                min_area_ratio=min_area_ratio,
                                max_regions=max_regions,
                                pooling_strategy=pooling_strategy
                            )
                        else:
                            # Modify collection name to include mode and layer
                            collection_name = f"{collection_name}_whole_images_layer{layer}"
                            print(f"[STATUS] Using whole image collection name: {collection_name}")
                            # Process with whole image for single folder
                            gen = process_folder_with_progress_whole_images(
                                folder_path,
                                collection_name,
                                optimal_layer=layer
                            )
                    
                    # Return first message
                    result = next(gen)
                    
                    # Process remaining generator values 
                    for message, done_visible in gen:
                        yield message, gr.update(visible=done_visible)
                        
                    # After completing database creation, refresh verify dropdown
                    verify_db_dropdown.choices = app_state.update_available_databases(verbose=True)
                    if app_state.available_databases:
                        verify_db_dropdown.value = app_state.available_databases[0]
                
                create_db_button.click(
                    process_folder_with_mode,
                    inputs=[db_folder_path, db_collection_name, region_prompts, layer_slider, db_mode, min_area_ratio, max_regions, pooling_strategy],
                    outputs=[db_progress, db_done_msg]
                )
            
            # Tab 3: Search Database
            with gr.TabItem("Search Database"):
                gr.Markdown("## Search Database")
                gr.Markdown("Upload an image, detect regions, and search for similar regions in your database.")
                
                # Database selection
                with gr.Row():
                    with gr.Column(scale=1):
                        # Refresh button for database list
                        refresh_db_button = gr.Button("Refresh Database List")
                    with gr.Column(scale=2):
                        db_dropdown = gr.Dropdown(
                            choices=app_state.available_databases if app_state.available_databases else ["No databases found"],
                            value=app_state.available_databases[0] if app_state.available_databases else None,
                            label="Select Database"
                        )
                    with gr.Column(scale=1):
                        # Delete database button
                        delete_db_button = gr.Button("Delete Selected Database", variant="stop")
                
                # Add row for reset database connection button
                with gr.Row():
                    with gr.Column(scale=3):
                        # Empty space for alignment
                        gr.Markdown("") 
                    with gr.Column(scale=1):
                        # Reset database connection button
                        reset_db_button = gr.Button("Reset Database Connection", variant="secondary")
                
                # Processing mode selector
                with gr.Row():
                    processing_mode = gr.Radio(
                        choices=["Region Detection (GroundedSAM + PE)", "Whole Image (PE Only)"],
                        value="Region Detection (GroundedSAM + PE)",
                        label="Select Processing Mode"
                    )
                
                # Common file upload for both modes
                with gr.Row():
                    input_image = gr.Image(type="pil", label="Upload Image")
                
                # Region-based mode components
                with gr.Group(visible=True) as region_mode_group:
                    with gr.Row():
                        text_prompt = gr.Textbox(
                            placeholder="Examples: 'person . car . building .' OR 'all the chairs in the room' OR 'a person with pink clothes'",
                            label="Detection Prompts",
                            info="💡 Use period-separated for multiple objects (person . car . building .) OR natural language for specific descriptions (all the chairs, a person with red shirt)",
                            value="person . car . building ."
                        )
                    
                    # Add parameter controls before the process button
                    with gr.Row():
                        with gr.Column():
                            embedding_layer = gr.Slider(
                                minimum=1, maximum=50, value=40, step=1,
                                label="Embedding Layer",
                                info="Layer of the Perception Encoder to use for detection"
                            )
                        with gr.Column():
                            min_area_ratio = gr.Slider(
                                minimum=0.001, maximum=0.1, value=0.01, step=0.001,
                                label="Minimum Area Ratio",
                                info="Minimum size of regions to detect (as a fraction of image size)"
                            )
                        with gr.Column():
                            max_regions = gr.Slider(
                                minimum=1, maximum=20, value=5, step=1,
                                label="Maximum Regions",
                                info="Maximum number of regions to detect per image"
                            )
                    
                    with gr.Row():
                        process_button = gr.Button("Detect Regions", variant="primary")
                
                # Whole-image mode components
                with gr.Group(visible=False) as whole_image_mode_group:
                    # Add embedding layer for whole image processing too
                    with gr.Row():
                        whole_image_layer = gr.Slider(
                            minimum=1, maximum=50, value=40, step=1,
                            label="Embedding Layer",
                            info="Layer of the Perception Encoder to use for processing"
                        )
                    with gr.Row():
                        process_whole_button = gr.Button("Process Whole Image", variant="primary")
                
                # Status and results area
                with gr.Row():
                    with gr.Column():
                        # Results for region-based mode
                        with gr.Group(visible=True) as region_results_group:
                            detection_info = gr.Textbox(label="Detection Status")
                            segmented_output = gr.Image(type="pil", label="Detected Regions")
                            region_dropdown = gr.Dropdown(choices=[], label="Select a Region")
                            region_preview = gr.Image(type="pil", label="Region Preview")
                            
                            with gr.Row():
                                with gr.Column():
                                    similarity_slider = gr.Slider(
                                        minimum=0.0, maximum=1.0, value=0.5, step=0.01,
                                        label="Similarity Threshold"
                                    )
                                with gr.Column():
                                    max_results_dropdown = gr.Dropdown(
                                        choices=["5", "10", "20", "50"], value="5",
                                        label="Max Results"
                                    )
                            
                            # Remove the embedding layer slider from here since we moved it above
                            search_button = gr.Button("Search Similar Regions", variant="primary")
                        
                        # Results for whole-image mode
                        with gr.Group(visible=False) as whole_image_results_group:
                            whole_image_info = gr.Textbox(label="Processing Status")
                            processed_output = gr.Image(type="pil", label="Processed Image")
                            
                            with gr.Row():
                                with gr.Column():
                                    whole_similarity_slider = gr.Slider(
                                        minimum=0.0, maximum=1.0, value=0.5, step=0.01,
                                        label="Similarity Threshold"
                                    )
                                with gr.Column():
                                    whole_max_results_dropdown = gr.Dropdown(
                                        choices=["5", "10", "20", "50"], value="5",
                                        label="Max Results"
                                    )
                                    
                            whole_search_button = gr.Button("Search Similar Images", variant="primary")
                        
                        # Common search results area
                        search_info = gr.Textbox(label="Search Status")
                        search_results_output = gr.Image(type="pil", label="Search Results")
                        
                        with gr.Group(visible=False) as button_section:
                            result_selector = gr.Dropdown(choices=[], label="Select Result to View")
                        
                        enlarged_result = gr.Image(type="pil", label="Enlarged Result")
                    
                    # Function to toggle visibility based on mode
                    def toggle_mode(mode):
                        if mode == "Region Detection (GroundedSAM + PE)":
                            return gr.update(visible=True), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
                        else:  # Whole Image mode
                            return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=True)
                    
                    # Connect mode selector to toggle visibility
                    processing_mode.change(
                        toggle_mode,
                        inputs=[processing_mode],
                        outputs=[
                            region_mode_group, 
                            whole_image_mode_group,
                            region_results_group,
                            whole_image_results_group
                        ]
                    )
                    
                    # Function to refresh database list
                    def refresh_databases():
                        app_state.update_available_databases(verbose=True)
                        if app_state.available_databases:
                            print(f"[STATUS] Updated dropdown with {len(app_state.available_databases)} databases")
                            return gr.Dropdown(choices=app_state.available_databases, value=app_state.available_databases[0])
                        else:
                            print(f"[STATUS] No databases found to display in dropdown")
                            return gr.Dropdown(choices=["No databases found"], value=None)
                    
                    # Connect refresh button
                    refresh_db_button.click(refresh_databases, inputs=[], outputs=[db_dropdown])
                    
                    # Function to set active database
                    def set_active_database(collection_name):
                        if collection_name and collection_name != "No databases found":
                            interface.set_active_collection(collection_name)
                            return f"Set active database to: {collection_name}"
                        return "No database selected"
                    
                    # Connect database dropdown
                    db_dropdown.change(set_active_database, inputs=[db_dropdown], outputs=[detection_info])
                    
                    # Function to delete the selected database
                    def delete_selected_database(collection_name):
                        if collection_name and collection_name != "No databases found":
                            success, message = delete_database_collection(collection_name)
                            if success:
                                # Refresh the database list after successful deletion
                                app_state.update_available_databases(verbose=True)
                                if app_state.available_databases:
                                    return gr.Dropdown(choices=app_state.available_databases, value=app_state.available_databases[0]), f"{message}"
                                else:
                                    return gr.Dropdown(choices=["No databases found"], value=None), f"{message}"
                            return gr.update(), f"{message}"
                        return gr.update(), "No database selected for deletion"
                    
                    # Connect delete button
                    delete_db_button.click(delete_selected_database, inputs=[db_dropdown], outputs=[db_dropdown, detection_info])
                    
                    # Function to reset the database connection
                    def reset_database_connection():
                        """Reset the database connection by forcefully cleaning up locks"""
                        try:
                            print(f"[STATUS] Forcefully resetting database connection...")
                            
                            # Close any existing client if present
                            if interface.active_client is not None:
                                try:
                                    interface.active_client.close()
                                    interface.active_client = None
                                    print(f"[STATUS] Closed existing database client")
                                except Exception as e:
                                    print(f"[WARNING] Error closing client: {e}")
                            
                            # Run aggressive cleanup
                            success = cleanup_qdrant_connections(force=True)
                            
                            # Refresh database list
                            app_state.update_available_databases(verbose=True)
                            
                            if success:
                                message = "✅ Database connection has been reset successfully."
                                if app_state.available_databases:
                                    return gr.Dropdown(choices=app_state.available_databases, value=app_state.available_databases[0]), message
                                else:
                                    return gr.Dropdown(choices=["No databases found"], value=None), message
                            else:
                                # If the cleanup wasn't successful
                                message = "⚠️ Database reset partially completed. You may need to restart the application if issues persist."
                                if app_state.available_databases:
                                    return gr.Dropdown(choices=app_state.available_databases, value=app_state.available_databases[0]), message
                                else:
                                    return gr.Dropdown(choices=["No databases found"], value=None), message
                            
                        except Exception as e:
                            print(f"[ERROR] Error during database reset: {e}")
                            import traceback
                            traceback.print_exc()
                            return gr.update(), f"❌ Error resetting database: {str(e)}"
                    
                    # Connect reset button
                    reset_db_button.click(reset_database_connection, inputs=[], outputs=[db_dropdown, detection_info])
                    
                    # Connect buttons to functions for region-based processing
                    process_button.click(
                        interface.process_image_with_prompt,
                        inputs=[input_image, text_prompt, min_area_ratio, max_regions, embedding_layer],
                        outputs=[segmented_output, detection_info, region_dropdown, region_preview]
                    )

                    region_dropdown.change(
                        interface.update_region_preview,
                        inputs=[region_dropdown],
                        outputs=[region_preview]
                    )

                    search_button.click(
                        interface.search_region,
                        inputs=[region_dropdown, similarity_slider, max_results_dropdown, embedding_layer],
                        outputs=[search_results_output, search_info, button_section, result_selector]
                    )
                    
                    # Connect buttons to functions for whole-image processing
                    process_whole_button.click(
                        interface.process_whole_image,
                        inputs=[input_image, whole_image_layer],
                        outputs=[processed_output, whole_image_info]
                    )
                    
                    whole_search_button.click(
                        interface.search_whole_image,
                        inputs=[whole_similarity_slider, whole_max_results_dropdown, whole_image_layer],
                        outputs=[search_results_output, search_info, button_section, result_selector]
                    )
                    
                    # Connect result selector for both modes
                    result_selector.change(
                        interface.display_enlarged_result,
                        inputs=[result_selector],
                        outputs=[enlarged_result]
                    )
                
            # Tab 4: About
            with gr.TabItem("About"):
                gr.Markdown("""
                    # Revers-o: A  Guide

                    ## Table of Contents
                    1. [What is Revers-o?](#what-is-revers-o)
                    2. [Understanding the Technology](#understanding-the-technology)
                    3. [Getting Started](#getting-started)
                    4. [Video Processing Workflow](#video-processing-workflow)
                    5. [Creating Image Databases](#creating-image-databases)
                    6. [Searching for Similar Content](#searching-for-similar-content)
                    7. [Parameter Guide](#parameter-guide)
                    8. [Tips and Best Practices](#tips-and-best-practices)
                    10. [Troubleshooting](#troubleshooting)

                    ---

                    ## What is Revers-o?

                    Revers-o is a powerful visual investigation tool that helps journalists analyze large collections of images and videos to find similar content, objects, or scenes. Think of it as a "reverse image search" on steroids, specifically designed for investigative work.

                    **Key Capabilities:**
                    - Extract frames from videos automatically
                    - Find similar objects, people, or scenes across thousands of images
                    - Search by describing what you're looking for in plain English
                    - Build searchable databases of visual evidence
                    - Identify patterns and connections in visual content

                    ---

                    ## Understanding the Technology

                    ### Why Combine GroundedSAM and Perception Encoders?

                    Revers-o combines two complementary AI systems to provide comprehensive visual analysis capabilities that neither could achieve alone:

                    - **GroundedSAM** excels at precise object detection and segmentation based on natural language descriptions, but operates primarily on explicit, describable objects
                    - **Perception Encoders** capture rich semantic relationships and abstract visual concepts, but lacks the ability to follow complex user prompts.

                    By combining them, Revers-o allows you to describe what objects you want your database to be made of  AND create high quality embeddings

                    ### GroundedSAM: The Object Detective

                    GroundedSAM combines two powerful AI systems:

                    1. **Grounding DINO**: Understands natural language descriptions and finds objects in images
                    2. **Segment Anything Model (SAM)**: Precisely outlines the boundaries of detected objects

                    **How it works for journalists:**
                    - You type: "person wearing red jacket"
                    - The system finds all people wearing red jackets in your images
                    - It creates precise outlines around each person
                    - These outlines become searchable "fingerprints"

                    ### Perception Encoders: The Pattern Recognizer

                    Meta's Perception Encoders are AI systems that understand the visual "meaning" of image regions. They convert visual information into mathematical representations that capture:
                    - Object appearance and context
                    - Spatial relationships
                    - Visual similarities that humans would recognize

                    **Real-world example:**
                    If you have footage of a protest and want to find all images showing the same building in the background, Perception Encoders may be able to identify that building across different angles, lighting conditions, and camera positions.

                    ### The Hierarchical Nature of Visual Understanding

                    Recent research by Meta has revealed a crucial insight: different layers of Perception Encoders capture different aspects of visual understanding:

                    **Early Layers (Low-level features):**
                    - Edges, textures, and basic shapes
                    - Color patterns and gradients
                    - Simple geometric structures

                    **Middle Layers (Mid-level features):**
                    - Object parts and components
                    - Spatial arrangements
                    - Local patterns and motifs

                    **Later Layers (High-level features):**
                    - Complete objects and scenes
                    - Abstract concepts and relationships
                    - Semantic understanding

                    **How Revers-o Leverages This Hierarchy:**

                    This multi-layered understanding theoretically enables Revers-o to search for both concrete and abstract visual concepts:

                    **Concrete Searches:**
                    - "Person wearing red jacket" (specific object detection)
                    - "Blue car with license plate" (detailed object features)
                    - "Building with glass facade" (architectural elements)

                    **Abstract Concept Searches:**
                    - Visual similarity based on "mood" or "atmosphere"
                    - Scenes with similar "energy" or "tension"
                    - Images that convey similar "emotions" or "contexts"
                    - Compositional similarities (similar layouts, even with different objects)


                    This hierarchical approach means Revers-o can help investigators to explore their visual databases in multiple ways by creating and searching embeddings from different layers.

                    ---

                    ## Getting Started

                    ### 1. Launch the Application
                    Run the application using the provided scripts:
                    - **Windows**: Double-click `run.bat`
                    - **Mac/Linux**: Run `./run.sh` in terminal

                    The interface will open in your web browser.

                    ### 2. Interface Overview
                    The main interface has three sections:
                    1. **Extract Images from Video** - Process video files
                    2. **Create New Database** - Build searchable collections
                    3. **Search Existing Database** - Find similar content

                    ---

                    ## Video Processing Workflow

                    ### Step-by-Step Process

                    1. **Prepare Your Videos**
                    - Supported formats: MP4, AVI, MOV, MKV, WMV, FLV, WebM, M4V
                    - Place videos in a dedicated folder

                    2. **Set Input and Output Folders**
                    - **Video Folder Path**: Where your videos are stored
                    - **Output Folder Path**: Where extracted frames will be saved

                    3. **Configure Extraction Settings**
                    - **Max Frames per Video**: How many frames to extract (default: 30)
                    - **Scene Detection Threshold**: Sensitivity for detecting scene changes (default: 30.0)

                    4. **Process Videos**
                    - Click "Process Videos"
                    - The system will extract keyframes and save them as images
                    - Progress will be shown in real-time

                    ### Understanding Scene Detection
                    - **Lower threshold (10-20)**: More sensitive, extracts more frames during subtle changes
                    - **Higher threshold (40-50)**: Less sensitive, only extracts frames during major scene changes
                    - **Default (30)**: Balanced approach suitable for most content

                    ---

                    ## Creating Image Databases

                    ### Planning Your Database

                    **Before you start:**
                    - Organize images by investigation topic
                    - Use descriptive database names (e.g., "protest_march_2024", "building_surveillance")
                    - Consider what objects/people you'll be searching for

                    ### Step-by-Step Database Creation

                    1. **Choose Your Images**
                    - Select a single folder containing your images OR multiple folders separated by commas
                    - Example single folder: `/path/to/images`
                    - Example multiple folders: `/path/to/images1, /path/to/images2, /path/to/images3`
                    - Can include extracted video frames from different sources
                    - All images from all folders will be processed into the same database

                    2. **Define Search Prompts**
                    - Enter descriptions of what you may want to reverse-image search in your database. For example, if your main use case is geolocation, you could ask reverso to create a database of buildings (or mountains) so you can later compare new images of buildings with what you already have.
                    - Use clear, specific language
                    - Examples: "person in uniform", "red car", "protest sign", "building entrance"

                    3. **Configure Detection Settings**
                    - **Box Threshold**: How confident the system should be (0.1-0.9)
                    - **Text Threshold**: Confidence for text-based detection (0.1-0.9)

                    4. **Name Your Database**
                    - Use descriptive names
                    - Avoid spaces (use underscores: "investigation_name")

                    5. **Start Processing**
                    - Click "Create Database"
                    - Processing time depends on image count and complexity

                    ### Understanding Prompts

                    **Good prompts:**
                    - "person wearing police uniform"
                    - "vehicle with license plate"
                    - "building with red door"
                    - "crowd of people"

                    **Avoid vague prompts:**
                    - "thing" or "object"
                    - "something suspicious"
                    - "important item"

                    ---

                    ## Searching for Similar Content

                    ### Upload and Detect
                    1. **Upload Query Image**: The image containing what you want to find
                    2. **Enter Detection Prompt**: Describe what to select in the uploaded image, these are the things that you later will be able to search within your database.
                    3. **Adjust Detection Settings**: Fine-tune confidence levels
                    4. **Click "Detect Regions"**: System identifies matching areas

                    ### Search Parameters
                    - **Top K Results**: How many similar images to return (default: 10)
                    - **Similarity Threshold**: Minimum similarity score (0.0-1.0)

                    ### Interpreting Results
                    - **Similarity Score**: Higher numbers = more similar (0.0-1.0 scale)
                    - **Bounding Boxes**: Show exactly what was matched
                    - **File Paths**: Help locate original images

                    ---

                    ## Parameter Guide

                    ### Box Threshold (0.1-0.9)
                    **What it does:** Controls how confident the system must be to detect an object.

                    - **0.1-0.3**: Very sensitive, finds more objects but may include false positives
                    - **0.4-0.6**: Balanced, good for most investigations
                    - **0.7-0.9**: Very strict, only finds objects the system is very confident about

                    **Use cases:**
                    - **Low (0.2)**: When searching for partially obscured objects
                    - **High (0.8)**: When you need very precise matches

                    ### Text Threshold (0.1-0.9)
                    **What it does:** Controls confidence for text-based object detection.

                    - **Lower values**: More liberal interpretation of your text prompts
                    - **Higher values**: Stricter matching to your exact description

                    ### Scene Detection Threshold (10.0-50.0)
                    **What it does:** Determines when video scenes change enough to extract a new frame.

                    - **10-20**: Extracts frames frequently, good for detailed analysis
                    - **30-40**: Balanced extraction, suitable for most content
                    - **40-50**: Only major scene changes, good for long videos with stable scenes

                    ### Max Frames per Video
                    **What it does:** Limits how many frames are extracted from each video.

                    - **10-20**: Quick overview of video content
                    - **30-50**: Detailed analysis while managing storage
                    - **100+**: Comprehensive frame-by-frame analysis


                    ---

                    ## Tips and Best Practices

                    ### Database Management
                    - **Use descriptive names**: Include date, location, or case identifier
                    - **Organize by topic**: Separate databases for different investigations
                    - **Regular backups**: Databases are stored locally and should be backed up

                    ### Prompt Writing
                    - **Be specific**: "red sedan" instead of "car"
                    - **Include context**: "person wearing medical mask" vs "person"
                    - **Test variations**: Try different phrasings for better results

                    ### Performance Optimization
                    - **Batch processing**: Process large collections overnight
                    - **Storage management**: Monitor disk space for large video collections
                    - **Quality over quantity**: Higher resolution images give better results

                    ### Investigation Workflow
                    1. **Plan your search strategy** before processing
                    2. **Start with small test batches** to refine parameters
                    3. **Document your methodology** for reproducible results
                    4. **Cross-reference findings** with other evidence sources

                    ---

                    ## Troubleshooting

                    ### Common Issues

                    **"No objects detected"**
                    - Lower the box threshold
                    - Try different prompt wording
                    - Check image quality and resolution

                    **"Too many false positives"**
                    - Raise the box threshold
                    - Use more specific prompts
                    - Adjust text threshold

                    **"Video processing fails"**
                    - Check video format compatibility
                    - Ensure sufficient disk space
                    - Try smaller video files first

                    **"Slow processing"**
                    - Reduce max frames per video
                    - Process smaller batches
                    - Close other applications to free memory

                    ### Getting Help
                    - Check the console output for error messages
                    - Verify all file paths are correct
                    - Ensure sufficient disk space for processing
                    - Restart the application if it becomes unresponsive
                    """)
    
    # Set the default active database if available
    if app_state.available_databases:
        interface.set_active_collection(app_state.available_databases[0])
    
    return demo

# =============================================================================
# MULTI-LAYER FEATURE HELPERS (now at module level)
# =============================================================================

def get_preset_configuration(preset_name):
    presets = {
        "object_focused": {
            "layers": [30, 40, 47],
            "weights": [0.3, 0.4, 0.3],
            "pooling": "pe_attention"
        },
        "spatial_focused": {
            "layers": [25, 30, 35], 
            "weights": [0.5, 0.3, 0.2],
            "pooling": "pe_spatial"
        },
        "semantic_focused": {
            "layers": [40, 45, 47],
            "weights": [0.2, 0.3, 0.5], 
            "pooling": "pe_semantic"
        }
    }
    return presets.get(preset_name, presets["object_focused"]) # Default to object_focused

def extract_multi_layer_features(pil_image, pe_vit_model, preprocess_fn, layers, device):
    """
    Extract features from multiple PE layers.
    Assumes pe_vit_model is the VisionTransformer model.
    Assumes pil_image is a PIL Image.
    preprocess_fn is the preprocessing function.
    """
    if pe_vit_model is None:
        print("[WARN] Vision Transformer model not available for multi-layer feature extraction.")
        return None
        
    layer_features = {}
    # Ensure preprocess_fn and pe_input are correctly handled if this function is truly global
    # For now, assuming preprocess_fn is passed in and works.
    pe_input = preprocess_fn(pil_image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        for layer_idx in layers:
            try:
                features = pe_vit_model.forward_features(pe_input, layer_idx=max(1, layer_idx))
                layer_features[layer_idx] = features
                print(f"[INFO] Extracted features from layer {layer_idx}, shape: {features.shape}")
            except Exception as e:
                print(f"[ERROR] Failed to extract features from layer {layer_idx}: {e}")
                layer_features[layer_idx] = None
    return layer_features

def combine_layer_features(layer_features_dict, weights_list, temperature=0.07):
    """
    Combine features from multiple layers using weighted averaging.
    layer_features_dict: Dictionary of features keyed by layer number {layer_idx: tensor}.
    weights_list: List of weights corresponding to the layers.
    temperature: Not used in current simple weighted average.
    """
    if not layer_features_dict or not weights_list:
        print("[WARN] Layer features or weights are empty. Cannot combine.")
        return None
    
    # Ensure correct alignment of features and weights.
    # This implementation assumes that layer_features_dict.values() will provide features
    # in an order that corresponds to weights_list if the dict was populated in layers_to_use order.
    # A more robust method would be to pass layers_to_use and iterate through it.
    feature_tensors = [feat for feat in layer_features_dict.values() if feat is not None] # Filter out None features
    
    if not feature_tensors:
        print("[WARN] No valid feature tensors to combine after filtering Nones.")
        return None

    # Adjust weights if some features were None and filtered out.
    # This part is tricky if weights_list was for the original set of layers.
    # For simplicity, this version assumes weights_list corresponds to the non-None features.
    # A truly robust implementation would require layer_keys with weights or careful filtering.
    if len(feature_tensors) != len(weights_list):
        print(f"[WARN] Number of non-None feature tensors ({len(feature_tensors)}) "
              f"does not match number of weights ({len(weights_list)}). Combination might be skewed or fail.")
        # Fallback: if only one feature tensor, return it. Otherwise, cannot proceed if mismatch.
        if len(feature_tensors) == 1 and len(weights_list) >=1 : # Try to use first weight if mismatch
             return feature_tensors[0] * (weights_list[0] / sum(weights_list)) if sum(weights_list) !=0 else feature_tensors[0]
        elif len(feature_tensors) == 1: # if weights_list is empty but one feature
             return feature_tensors[0]
        return None # Cannot reliably combine

    # Weighted sum
    combined_features = None
    # Normalize weights if they don't sum to 1 (or handle as proportions)
    total_weight = sum(weights_list)
    if total_weight == 0: # Avoid division by zero if all weights are zero
        print("[WARN] Total weight is zero. Averaging features instead.")
        # Fallback to simple average if total_weight is zero
        if feature_tensors:
            sum_features = None
            for features in feature_tensors:
                if sum_features is None: sum_features = features
                else: sum_features += features
            return sum_features / len(feature_tensors) if len(feature_tensors) > 0 else None
        return None

    for i, features in enumerate(feature_tensors):
        weight = weights_list[i] / total_weight # Normalize weight
        if combined_features is None:
            combined_features = features * weight
        else:
            if combined_features.shape != features.shape:
                 print(f"[WARN] Shape mismatch during combination: combined_features ({combined_features.shape}) vs features ({features.shape}).")
                 # Attempting to sum might fail. Add more sophisticated handling if needed.
            combined_features += features * weight
            
    print(f"[INFO] Combined {len(feature_tensors)} layer features using normalized weights.")
    return combined_features

# =============================================================================
# MAIN FUNCTION
# =============================================================================

def cleanup_qdrant_connections(force=False):
    """Attempt to forcefully clean up any existing Qdrant connections"""
    try:
        # Path to the lock file and data directories
        qdrant_data_dir = os.path.join("image_retrieval_project", "qdrant_data")
        lock_file = os.path.join(qdrant_data_dir, ".lock")
        collection_dir = os.path.join(qdrant_data_dir, "collection")
        collections_dir = os.path.join(qdrant_data_dir, "collections")
        
        # Create directories if they don't exist
        os.makedirs(qdrant_data_dir, exist_ok=True)
        
        # Force kill any processes using the qdrant files on macOS/Linux
        if sys.platform in ["darwin", "linux"]:
            try:
                import subprocess
                print(f"[STATUS] Checking for processes using Qdrant database...")
                # Find processes using the qdrant_data directory
                result = subprocess.run(
                    f"lsof -t +D {qdrant_data_dir} 2>/dev/null || echo ''", 
                    shell=True, 
                    text=True, 
                    capture_output=True
                )
                
                if result.stdout.strip():
                    pids = result.stdout.strip().split("\n")
                    print(f"[STATUS] Found {len(pids)} process(es) using Qdrant data directory")
                    
                    # Try to kill each process
                    for pid in pids:
                        if pid and pid.strip():
                            try:
                                # Don't kill our own process
                                if int(pid) != os.getpid():
                                    print(f"[STATUS] Terminating process {pid}...")
                                    subprocess.run(f"kill -9 {pid}", shell=True)
                                    print(f"[STATUS] Terminated process {pid}")
                            except Exception as e:
                                print(f"[WARNING] Could not terminate process {pid}: {e}")
                else:
                    print(f"[STATUS] No processes found using Qdrant database")
            except Exception as e:
                print(f"[WARNING] Error checking for processes using Qdrant: {e}")
        
        # Check if the lock file exists
        lock_removed = False
        if os.path.exists(lock_file):
            print(f"[STATUS] Found existing Qdrant lock file. Attempting to clean up...")
            try:
                # Try to remove the lock file
                os.remove(lock_file)
                print(f"[STATUS] Successfully removed lock file")
                lock_removed = True
                # Sleep briefly to let the filesystem settle
                time.sleep(1)
            except Exception as e:
                print(f"[WARNING] Could not remove lock file: {e}")
                
                # If force is True, try more aggressive methods
                if force:
                    print(f"[STATUS] Attempting to force remove lock...")
                    try:
                        # On Unix-like systems, try using system commands
                        if sys.platform in ["darwin", "linux"]:
                            import subprocess
                            subprocess.run(f"rm -f {lock_file}", shell=True)
                            if not os.path.exists(lock_file):
                                print(f"[STATUS] Successfully force-removed lock file")
                                lock_removed = True
                                time.sleep(1)
                    except Exception as force_e:
                        print(f"[WARNING] Force removal failed: {force_e}")
        else:
            print(f"[STATUS] No Qdrant lock file found")
            lock_removed = True  # No lock to remove
            
        # If force is True and we couldn't remove the lock, try rebuilding the database structure
        if force and not lock_removed:
            print(f"[STATUS] Attempting to rebuild Qdrant database structure...")
            try:
                # Backup the collections
                timestamp = int(time.time())
                backup_dir = f"./image_retrieval_project/qdrant_backup_{timestamp}"
                os.makedirs(backup_dir, exist_ok=True)
                
                # Copy collection data if directories exist
                if os.path.exists(collection_dir):
                    import shutil
                    shutil.copytree(collection_dir, os.path.join(backup_dir, "collection"))
                    print(f"[STATUS] Backed up collection directory to {backup_dir}/collection")
                
                if os.path.exists(collections_dir):
                    import shutil
                    shutil.copytree(collections_dir, os.path.join(backup_dir, "collections"))
                    print(f"[STATUS] Backed up collections directory to {backup_dir}/collections")
                
                # Remove the entire qdrant_data directory
                import shutil
                shutil.rmtree(qdrant_data_dir, ignore_errors=True)
                time.sleep(1)
                
                # Recreate the directory structure
                os.makedirs(qdrant_data_dir, exist_ok=True)
                
                # Restore collections
                if os.path.exists(os.path.join(backup_dir, "collection")):
                    shutil.copytree(os.path.join(backup_dir, "collection"), collection_dir)
                    print(f"[STATUS] Restored collection directory")
                
                if os.path.exists(os.path.join(backup_dir, "collections")):
                    shutil.copytree(os.path.join(backup_dir, "collections"), collections_dir)
                    print(f"[STATUS] Restored collections directory")
                
                print(f"[STATUS] Qdrant database structure rebuilt")
                return True
            except Exception as rebuild_e:
                print(f"[ERROR] Failed to rebuild database: {rebuild_e}")
                return False
        
        return lock_removed
    except Exception as e:
        print(f"[ERROR] Error during Qdrant cleanup: {e}")
        return False
        
def main():
    # Parse command line arguments for backward compatibility
    parser = argparse.ArgumentParser(description="Grounded SAM Region Search Application")
    parser.add_argument("--build-database", action="store_true", help="Build database from images folder")
    parser.add_argument("--interface", action="store_true", help="Launch Gradio interface")
    parser.add_argument("--all", action="store_true", help="Build database and launch interface")
    parser.add_argument("--folder", default="./my_images", help="Folder containing images to process")
    parser.add_argument("--prompts", default="person . building . car . text . object .", help="Text prompts for detection (period-separated format)")
    parser.add_argument("--collection", default="grounded_image_regions", help="Qdrant collection name")
    # Removed legacy interface options
    # Add mode option
    parser.add_argument("--mode", choices=["region", "whole_image"], default="region", 
                        help="Processing mode: 'region' for GroundedSAM+PE or 'whole_image' for PE only")
    parser.add_argument("--optimal-layer", type=int, default=40, help="Optimal layer for PE model (default: 40)")
    # Add port option
    parser.add_argument("--port", type=int, default=7860, help="Port to use for the Gradio interface")

    args = parser.parse_args()

    # Clean up any existing database connections with force option
    print(f"[STATUS] Running database cleanup and integrity checks...")
    # Try to clean up with a timeout to prevent blocking forever
    import threading
    cleanup_thread = threading.Thread(target=cleanup_qdrant_connections, args=(True,))
    cleanup_thread.start()
    # Wait for 10 seconds max
    cleanup_thread.join(timeout=10)
    
    # If the thread is still alive, it's taking too long
    if cleanup_thread.is_alive():
        print(f"[WARNING] Database cleanup is taking too long - continuing with application startup")
        # We don't want to block forever, just continue with application startup
    else:
        print(f"[STATUS] Database cleanup completed")

    # Setup device and load models
    print("🚀 Initializing Grounded SAM Region Search Application")
    print("=" * 50)

    # Setup device and declare globals
    global device, pe_model, pe_vit_model, preprocess
    device = setup_device()

    # Load PE models
    print("\n🧠 Loading Perception Encoder models...")
    try:
        pe_model, pe_vit_model, preprocess = load_pe_model(device)
        print("✅ Models loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load models: {e}")
        print("Make sure you've run setup.py first!")
        return

    # Check if we should show the interface (either standalone or after database build)
    launch_interface = args.interface or args.all or not args.build_database
    
    # Build interface if needed
    if launch_interface:
        print("\n🌐 Launching multi-mode interface with both build and search functionality...")
        demo = create_multi_mode_interface(pe_model, pe_vit_model, preprocess, device)
        print("🚀 Interface ready! Opening in browser...")
        
        # Try to launch with different ports to avoid port conflicts
        import socket
        import os
        
        # First attempt to kill any existing process on the port
        os.system(f"lsof -ti:{args.port} | xargs kill -9 2>/dev/null || true")
        
        # Function to check if a port is in use
        def is_port_in_use(port):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                return s.connect_ex(('localhost', port)) == 0
        
        # Try ports sequentially
        max_port = args.port + 10
        current_port = args.port
        
        while current_port <= max_port:
            if not is_port_in_use(current_port):
                try:
                    print(f"Trying to start server on port {current_port}...")
                    demo.launch(share=False, server_name="127.0.0.1", server_port=current_port)
                    return  # If successful, exit the function
                except OSError:
                    print(f"Failed to start on port {current_port}")
            current_port += 1
        
        # If all ports failed, try without specifying a port
        print("All specified ports are in use. Letting Gradio choose an available port...")
        demo.launch(share=False)
        return

    # Build database if requested (legacy mode)
    if args.build_database or args.all:
        print(f"\n📁 Building database from folder: {args.folder}")
        
        if not os.path.exists(args.folder):
            print(f"❌ Error: Folder not found at {args.folder}")
            print(f"Please create the folder and add images, or specify a different folder with --folder")
            return

        # Choose processing mode based on argument
        if args.mode == "whole_image":
            print(f"📊 Processing Mode: Whole Image (PE Only)")
            client = process_folder_for_whole_image_database(
                folder_path=args.folder,
                collection_name=args.collection,
                pe_model=pe_model,
                pe_vit_model=pe_vit_model,
                preprocess=preprocess,
                device=device,
                optimal_layer=args.optimal_layer,
                checkpoint_interval=3,
                resume_from_checkpoint=True
            )
        else:  # Default to region mode
            print(f"📊 Processing Mode: Region Detection (GroundedSAM + PE)")
            client = process_folder_for_region_database(
                folder_path=args.folder,
                collection_name=args.collection,
                text_prompts=args.prompts,
                pe_model=pe_model,
                pe_vit_model=pe_vit_model,
                preprocess=preprocess,
                device=device,
                optimal_layer=args.optimal_layer,
                min_area_ratio=0.005,
                max_regions=5,
                visualize_samples=False,  # Disable for non-interactive mode
                sample_count=2,
                checkpoint_interval=3,
                resume_from_checkpoint=True
            )

        if client is not None:
            print("✅ Database built successfully!")
            client.close()
        else:
            print("❌ Failed to build database")
            if not args.interface:
                return

    # The interface launch is now handled at the beginning of the function
    # This legacy code has been removed

def verify_repair_database(collection_name, force_rebuild=False):
    """
    Verifies and optionally repairs a database collection
    
    Args:
        collection_name: Name of the collection to verify
        force_rebuild: If True, forces a complete rebuild of the collection
                     This is useful if the collection is corrupted
    
    Returns:
        tuple: (success, message)
    """
    try:
        print(f"[STATUS] Verifying database collection: {collection_name}")
        
        # Check if the collection exists
        qdrant_data_dir = os.path.join("image_retrieval_project", "qdrant_data")
        if not os.path.exists(qdrant_data_dir):
            return False, f"Qdrant data directory not found at {qdrant_data_dir}"
        
        # Try to connect to the database
        try:
            print(f"[STATUS] Connecting to Qdrant...")
            client = QdrantClient(path=qdrant_data_dir)
            print(f"[STATUS] Successfully connected to Qdrant database")
        except Exception as e:
            print(f"[ERROR] Failed to connect to Qdrant: {e}")
            
            # Try to fix connection issues
            print(f"[STATUS] Attempting to fix database connection...")
            cleanup_success = cleanup_qdrant_connections(force=True)
            
            if cleanup_success:
                try:
                    client = QdrantClient(path=qdrant_data_dir)
                    print(f"[STATUS] Successfully reconnected to database after cleanup")
                except Exception as e2:
                    return False, f"Failed to connect to database even after cleanup: {e2}"
            else:
                return False, f"Failed to clean up database connections: {e}"
        
        # Check if the collection exists
        try:
            collections = client.get_collections().collections
            collection_names = [collection.name for collection in collections]
            print(f"[STATUS] Found collections: {collection_names}")
            
            if collection_name not in collection_names:
                return False, f"Collection '{collection_name}' not found"
        except Exception as e:
            print(f"[ERROR] Failed to get collections list: {e}")
            return False, f"Failed to list collections: {e}"
        
        # Get collection info
        try:
            collection_info = client.get_collection(collection_name=collection_name)
            vector_size = collection_info.config.params.vectors.size
            print(f"[STATUS] Collection '{collection_name}' has vector size {vector_size}")
            
            # Count points
            count_result = client.count(collection_name=collection_name)
            point_count = count_result.count
            print(f"[STATUS] Collection has {point_count} points")
            
            if point_count == 0:
                return False, f"Collection '{collection_name}' is empty (0 points)"
            
            # If we're not forcing a rebuild, check a sample to ensure quality
            if not force_rebuild and point_count > 0:
                print(f"[STATUS] Checking sample points for quality...")
                
                # Get a small random sample
                sample_size = min(10, point_count)
                try:
                    # Search with random vector to get points
                    random_vector = np.random.rand(vector_size).astype(np.float32)
                    random_vector = random_vector / np.linalg.norm(random_vector)
                    
                    # Get sample points
                    sample = client.search(
                        collection_name=collection_name,
                        query_vector=random_vector.tolist(),
                        limit=sample_size,
                        score_threshold=0.0  # No threshold to get diverse samples
                    )
                    
                    print(f"[STATUS] Retrieved {len(sample)} sample points")
                    
                    # Check if the points have the necessary metadata
                    valid_points = 0
                    required_keys = ["bbox", "region_id", "image_source", "phrase"]
                    for point in sample:
                        if all(key in point.payload for key in required_keys):
                            valid_points += 1
                        else:
                            missing = [key for key in required_keys if key not in point.payload]
                            print(f"[WARNING] Point {point.id} is missing required metadata: {missing}")
                    
                    validity_percentage = (valid_points / len(sample)) * 100 if sample else 0
                    print(f"[STATUS] {validity_percentage:.1f}% of sampled points have valid metadata")
                    
                    if validity_percentage < 50:
                        print(f"[WARNING] Less than 50% of points have valid metadata")
                        if force_rebuild:
                            print(f"[STATUS] Will rebuild collection due to force_rebuild flag")
                        else:
                            return False, f"Database quality check failed: only {validity_percentage:.1f}% of points have valid metadata"
                    
                except Exception as e:
                    print(f"[ERROR] Error sampling points: {e}")
                    return False, f"Failed to sample points: {e}"
            
            # If force_rebuild is True, rebuild the collection
            if force_rebuild:
                print(f"[STATUS] Rebuilding collection '{collection_name}'...")
                
                # Create a backup first
                backup_name = f"{collection_name}_backup_{int(time.time())}"
                try:
                    print(f"[STATUS] Creating backup collection '{backup_name}'...")
                    client.create_collection(
                        collection_name=backup_name,
                        vectors_config=models.VectorParams(
                            size=vector_size,
                            distance=models.Distance.COSINE
                        )
                    )
                    
                    # Copy all points to the backup collection
                    batch_size = 100
                    offset = 0
                    
                    while offset < point_count:
                        # Scroll to get a batch of points
                        scroll_result = client.scroll(
                            collection_name=collection_name,
                            limit=batch_size,
                            offset=offset
                        )
                        
                        points = scroll_result[0]
                        if not points:
                            break
                            
                        # Convert to point structs
                        point_structs = []
                        for point in points:
                            point_structs.append(
                                models.PointStruct(
                                    id=point.id,
                                    vector=point.vector,
                                    payload=point.payload
                                )
                            )
                        
                        # Insert into backup
                        client.upsert(
                            collection_name=backup_name,
                            points=point_structs
                        )
                        
                        offset += len(points)
                        print(f"[STATUS] Backed up {offset}/{point_count} points...")
                    
                    print(f"[STATUS] Successfully created backup in collection '{backup_name}'")
                    
                    # Now rebuild the original collection
                    print(f"[STATUS] Deleting original collection '{collection_name}'...")
                    client.delete_collection(collection_name=collection_name)
                    
                    print(f"[STATUS] Recreating collection '{collection_name}'...")
                    client.create_collection(
                        collection_name=collection_name,
                        vectors_config=models.VectorParams(
                            size=vector_size,
                            distance=models.Distance.COSINE
                        )
                    )
                    
                    # Restore points from backup with better error handling
                    batch_size = 100
                    offset = 0
                    restored_count = 0
                    
                    while True:
                        try:
                            # Scroll to get a batch of points
                            scroll_result = client.scroll(
                                collection_name=backup_name,
                                limit=batch_size,
                                offset=offset
                            )
                            
                            points = scroll_result[0]
                            if not points:
                                break
                                
                            # Convert to point structs
                            point_structs = []
                            for point in points:
                                point_structs.append(
                                    models.PointStruct(
                                        id=point.id,
                                        vector=point.vector,
                                        payload=point.payload
                                    )
                                )
                            
                            # Insert into original
                            client.upsert(
                                collection_name=collection_name,
                                points=point_structs
                            )
                            
                            restored_count += len(points)
                            offset += len(points)
                            print(f"[STATUS] Restored {restored_count}/{point_count} points...")
                            
                        except Exception as e:
                            print(f"[ERROR] Error restoring batch at offset {offset}: {e}")
                            offset += batch_size  # Skip problematic batch
                    
                    print(f"[STATUS] Rebuild complete. Restored {restored_count}/{point_count} points.")
                    
                    if restored_count < point_count:
                        print(f"[WARNING] Some points ({point_count - restored_count}) could not be restored.")
                        return True, f"Collection rebuilt with partial success. Only {restored_count}/{point_count} points restored. Original data is preserved in '{backup_name}'."
                    
                    return True, f"Collection '{collection_name}' successfully rebuilt. Original data is preserved in '{backup_name}'."
                    
                except Exception as e:
                    print(f"[ERROR] Error during rebuild: {e}")
                    return False, f"Failed to rebuild collection: {e}"
            
            # If we got here without force_rebuild, the collection is valid
            return True, f"Collection '{collection_name}' verified successfully. {point_count} points with vector size {vector_size}."
            
        except Exception as e:
            print(f"[ERROR] Error getting collection info: {e}")
            return False, f"Failed to get collection info: {e}"
            
    except Exception as e:
        print(f"[ERROR] Verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False, f"Verification failed: {e}"

def collect_images_from_multiple_folders(folder_paths_string):
    """
    Collect all image paths from multiple comma-separated folder paths.
    Returns a tuple of (all_image_paths, folder_stats, error_messages)
    """
    # Parse and validate folder paths
    folder_paths = [path.strip() for path in folder_paths_string.split(',') if path.strip()]
    
    if not folder_paths:
        return [], {}, ["No folder paths provided"]
    
    all_image_paths = []
    folder_stats = {}
    error_messages = []
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    
    print(f"[STATUS] Processing {len(folder_paths)} folder(s)")
    
    for folder_path in folder_paths:
        folder_path = folder_path.strip()
        
        if not os.path.exists(folder_path):
            error_msg = f"❌ Folder not found: {folder_path}"
            error_messages.append(error_msg)
            print(f"[ERROR] {error_msg}")
            continue
            
        if not os.path.isdir(folder_path):
            error_msg = f"❌ Path is not a directory: {folder_path}"
            error_messages.append(error_msg)
            print(f"[ERROR] {error_msg}")
            continue
        
        # Get image files from this folder
        try:
            folder_images = []
            for file in os.listdir(folder_path):
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    full_path = os.path.join(folder_path, file)
                    folder_images.append(full_path)
            
            folder_stats[folder_path] = len(folder_images)
            all_image_paths.extend(folder_images)
            
            print(f"[STATUS] Found {len(folder_images)} images in {folder_path}")
            
        except Exception as e:
            error_msg = f"❌ Error reading folder {folder_path}: {str(e)}"
            error_messages.append(error_msg)
            print(f"[ERROR] {error_msg}")
    
    # Sort all paths for consistent processing order
    all_image_paths.sort()
    
    print(f"[STATUS] Total images collected: {len(all_image_paths)} from {len([f for f in folder_stats if folder_stats[f] > 0])} valid folders")
    
    return all_image_paths, folder_stats, error_messages


def process_multiple_folders_with_progress_advanced(folder_paths_string, prompts, collection_name, 
                                                  optimal_layer=40, min_area_ratio=0.01, max_regions=5, 
                                                  resume_from_checkpoint=True, pooling_strategy="top_k"):
    """Process multiple folders with progress updates for the UI with advanced parameters"""
    try:
        print(f"[STATUS] Starting multiple folder processing: {folder_paths_string}")
        print(f"[STATUS] Using collection: {collection_name}")
        print(f"[STATUS] Using prompts: {prompts}")
        print(f"[STATUS] Advanced parameters: optimal_layer={optimal_layer}, min_area_ratio={min_area_ratio}, " 
              f"max_regions={max_regions}, resume_from_checkpoint={resume_from_checkpoint}, pooling_strategy={pooling_strategy}")
        
        # Collect all image paths from multiple folders
        all_image_paths, folder_stats, error_messages = collect_images_from_multiple_folders(folder_paths_string)
        
        # Report folder validation results
        if error_messages:
            error_summary = "\n".join(error_messages)
            yield f"⚠️ Folder validation warnings:\n{error_summary}\n", gr.update(visible=False)
        
        if not all_image_paths:
            error_msg = "❌ No valid images found in any of the specified folders"
            print(f"[ERROR] {error_msg}")
            yield error_msg, gr.update(visible=False)
            return
        
        # Show folder summary
        folder_summary = "📁 Folder Summary:\n"
        valid_folders = 0
        for folder_path, count in folder_stats.items():
            if count > 0:
                folder_summary += f"  ✅ {folder_path}: {count} images\n"
                valid_folders += 1
            else:
                folder_summary += f"  ⚠️ {folder_path}: 0 images\n"
        
        folder_summary += f"\n📊 Total: {len(all_image_paths)} images from {valid_folders} folders"
        
        yield folder_summary, gr.update(visible=False)
        
        # Input validation for other parameters
        if not collection_name or not isinstance(collection_name, str):
            collection_name = f"image_database_{int(time.time())}"
            print(f"[WARNING] Invalid collection name provided, using: {collection_name}")
        
        # Ensure parameters are in valid ranges
        optimal_layer = max(1, int(optimal_layer))
        min_area_ratio = max(0.001, min(float(min_area_ratio), 0.5))
        max_regions = max(1, min(int(max_regions), 20))
        
        # Get global model variables
        global pe_model, pe_vit_model, preprocess, device
        
        # Process and validate prompts
        if not prompts or not isinstance(prompts, str) or prompts.strip() == "":
            prompts = "person . building . car . text . object ."
            print(f"[WARNING] No valid prompts provided, using defaults: {prompts}")
            
        # Split prompts if provided as period-separated string
        prompt_list = [p.strip().lower() for p in prompts.split(".") if p.strip()]
        
        # Ensure we have valid prompts
        if not prompt_list:
            prompt_list = ["person", "building", "car", "text", "object"]
        
        print(f"[STATUS] Parsed prompts: {prompt_list}")
        
        total = len(all_image_paths)
        print(f"[STATUS] Processing {total} images total")
        
        # Process images with progress updates
        client = None
        processed = 0
        skipped = 0
        errors = 0
        total_regions = 0
        vector_dimension = None
        
        # Force database cleanup before starting to avoid connection issues
        from threading import Thread
        cleanup_thread = Thread(target=cleanup_qdrant_connections, args=(True,))
        cleanup_thread.start()
        cleanup_thread.join(timeout=10)
        
        # Use collection name provided (should already include layer info from caller)
        collection_with_layer = collection_name
        
        # Update Gradio with initial status
        yield (f"🔍 Starting to process {total} images from {valid_folders} folders\n"
              f"📊 Parameters:\n"
              f"  - Semantic Layer: {optimal_layer}\n"
              f"  - Min Region Size: {min_area_ratio}\n"
              f"  - Max Regions: {max_regions}\n"
              f"  - Resume from checkpoint: {resume_from_checkpoint}\n"
              f"  - Collection name: {collection_with_layer}"), gr.update(visible=False)
        
        print(f"[STATUS] Beginning image processing loop")
        for i, img_path in enumerate(all_image_paths):
            # Get just the filename for display
            img_file = os.path.basename(img_path)
            # Get the parent folder for context
            parent_folder = os.path.basename(os.path.dirname(img_path))
            
            # Yield progress update with percentage
            progress_pct = ((i+1) / total) * 100
            yield f"🔄 Processing: {i+1}/{total} images ({progress_pct:.1f}%)\n📄 Current: {img_file} (from {parent_folder})", gr.update(visible=False)
            
            print(f"[STATUS] Processing image {i+1}/{total}: {img_path}")
            
            # Skip files that are too small or potentially corrupt
            try:
                img_size = os.path.getsize(img_path)
                if img_size < 1000:  # Files smaller than 1KB are likely invalid
                    print(f"[WARNING] Skipping very small file ({img_size} bytes): {img_path}")
                    skipped += 1
                    continue
            except Exception as e:
                print(f"[WARNING] Error checking file size: {e}")
            
            try:
                # Add memory management to avoid memory leaks
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                
                # Extract region embeddings with custom parameters
                image, masks, embeddings, metadata, labels, error_message = extract_region_embeddings_autodistill(
                    img_path,
                    " . ".join(prompt_list) + " .",
                    pe_model_param=pe_model,
                    pe_vit_model_param=pe_vit_model,
                    preprocess_param=preprocess,
                    device_param=device,
                    min_area_ratio=min_area_ratio,
                    max_regions=max_regions,
                    optimal_layer=optimal_layer,
                    pooling_strategy=pooling_strategy
                )
                
                # Check if there was an error during processing
                if error_message is not None:
                    print(f"[ERROR] Error processing {img_path}: {error_message}")
                    errors += 1
                    # Show error in UI progress updates
                    if (i + 1) % 1 == 0 or i == total - 1:  # Update every image when there are errors
                        error_stats = (
                            f"❌ Error processing {img_file}:\n{error_message}\n\n"
                            f"📊 Progress: {i+1}/{total} images ({(i+1)/total*100:.1f}%)\n"
                            f"✅ Processed: {processed} images\n"
                            f"⏭️ Skipped: {skipped} images (no regions found)\n"
                            f"❌ Errors: {errors} images\n"
                            f"🔍 Total regions found: {total_regions}"
                        )
                        yield error_stats, gr.update(visible=False)
                    continue
                
                if image is not None and len(embeddings) > 0:
                    print(f"[STATUS] Found {len(embeddings)} regions in {img_file}")
                    
                    # Additional validation for database creation
                    valid_embeddings = []
                    valid_metadata = []
                    invalid_count = 0
                    
                    for idx, (emb, meta) in enumerate(zip(embeddings, metadata)):
                        # Skip regions with missing or invalid data
                        if emb is None or meta is None:
                            print(f"[WARNING] Region {idx} has None embedding or metadata, skipping")
                            invalid_count += 1
                            continue
                            
                        # Validate embedding dimensions
                        if vector_dimension is not None:
                            # Use the same logic as dimension detection to extract actual feature dimension
                            if len(emb.shape) == 2 and emb.shape[0] == 1:
                                # Shape is [1, D], extract D
                                actual_dimension = emb.shape[1]
                            elif len(emb.shape) == 1:
                                # Shape is [D], use D
                                actual_dimension = emb.shape[0]
                            else:
                                # Flatten for safety
                                actual_dimension = emb.numel()
                            
                            if actual_dimension != vector_dimension:
                                print(f"[WARNING] Region {idx} has inconsistent embedding dimension: expected {vector_dimension}, got {actual_dimension}")
                                invalid_count += 1
                                continue
                        
                        # Validate the region metadata
                        if "bbox" not in meta or len(meta["bbox"]) != 4:
                            print(f"[WARNING] Region {idx} has invalid bbox, skipping")
                            invalid_count += 1
                            continue
                            
                        # Skip regions with suspiciously high confidence (1.0 exactly)
                        phrase = meta.get("phrase", "")
                        if ": 1.00" in phrase:
                            print(f"[WARNING] Region {idx} has perfect 1.00 confidence, likely false positive: {phrase}")
                            invalid_count += 1
                            continue
                        
                        # Add timestamp and source info to metadata
                        meta["processed_timestamp"] = time.time()
                        meta["source_filename"] = img_file
                        meta["source_folder"] = os.path.dirname(img_path)
                        meta["full_path"] = img_path
                            
                        # Include only valid embeddings
                        valid_embeddings.append(emb)
                        valid_metadata.append(meta)
                    
                    # If we filtered out bad regions, update counts
                    if invalid_count > 0:
                        print(f"[WARNING] Filtered out {invalid_count} invalid regions from {img_file}")
                    
                    if len(valid_embeddings) > 0:
                        # Initialize database if not done yet
                        if not client:
                            # Extract vector dimension from first valid embedding
                            first_embedding = valid_embeddings[0]
                            if len(first_embedding.shape) == 2 and first_embedding.shape[0] == 1:
                                vector_dimension = first_embedding.shape[1]
                            elif len(first_embedding.shape) == 1:
                                vector_dimension = first_embedding.shape[0]
                            else:
                                vector_dimension = first_embedding.numel()
                            
                            print(f"[STATUS] Initializing database with vector dimension: {vector_dimension}")
                            client = setup_qdrant(collection_with_layer, vector_dimension)
                            if client is None:
                                error_msg = "❌ Failed to initialize Qdrant client"
                                print(f"[ERROR] {error_msg}")
                                yield error_msg, gr.update(visible=False)
                                return
                        
                        # Store embeddings in database
                        store_embeddings_in_qdrant(client, collection_with_layer, valid_embeddings, valid_metadata)
                        total_regions += len(valid_embeddings)
                        processed += 1
                    else:
                        print(f"[WARNING] No valid regions found in {img_file} after filtering")
                        skipped += 1
                else:
                    print(f"[WARNING] No regions found in {img_file}")
                    skipped += 1
                
            except Exception as e:
                print(f"[ERROR] Exception processing {img_path}: {e}")
                import traceback
                traceback.print_exc()
                errors += 1
                continue
            
            # Periodic progress updates
            if (i + 1) % 5 == 0 or i == total - 1:
                progress_stats = (
                    f"📊 Progress: {i+1}/{total} images ({(i+1)/total*100:.1f}%)\n"
                    f"✅ Processed: {processed} images\n"
                    f"⏭️ Skipped: {skipped} images (no regions found)\n"
                    f"❌ Errors: {errors} images\n"
                    f"🔍 Total regions found: {total_regions}"
                )
                yield progress_stats, gr.update(visible=False)
        
        # Final completion message
        completion_message = (
            f"✅ Processing Complete!\n\n"
            f"📊 Final Statistics:\n"
            f"  - Total images processed: {processed}/{total}\n"
            f"  - Images skipped (no regions): {skipped}\n"
            f"  - Images with errors: {errors}\n"
            f"  - Total regions stored: {total_regions}\n"
            f"  - Folders processed: {valid_folders}\n"
            f"  - Database collection: {collection_with_layer}\n"
            f"  - Embedding layer used: {optimal_layer}"
        )
        
        print(f"[STATUS] {completion_message}")
        yield completion_message, gr.update(visible=True)
        
        # Close database connection
        if client:
            client.close()
            
    except Exception as e:
        error_msg = f"❌ Fatal error during processing: {str(e)}"
        print(f"[ERROR] {error_msg}")
        import traceback
        traceback.print_exc()
        yield error_msg, gr.update(visible=False)

def process_multiple_folders_with_progress_whole_images(folder_paths_string, collection_name, 
                                                     optimal_layer=40, resume_from_checkpoint=True):
    """Process multiple folders for whole image embeddings with progress updates"""
    try:
        print(f"[STATUS] Starting multiple folder processing for whole images: {folder_paths_string}")
        print(f"[STATUS] Using collection: {collection_name}")
        print(f"[STATUS] Parameters: optimal_layer={optimal_layer}, resume_from_checkpoint={resume_from_checkpoint}")
        
        # Collect all image paths from multiple folders
        all_image_paths, folder_stats, error_messages = collect_images_from_multiple_folders(folder_paths_string)
        
        # Report folder validation results
        if error_messages:
            error_summary = "\n".join(error_messages)
            yield f"⚠️ Folder validation warnings:\n{error_summary}\n", gr.update(visible=False)
        
        if not all_image_paths:
            error_msg = "❌ No valid images found in any of the specified folders"
            print(f"[ERROR] {error_msg}")
            yield error_msg, gr.update(visible=False)
            return
        
        # Show folder summary
        folder_summary = "📁 Folder Summary:\n"
        valid_folders = 0
        for folder_path, count in folder_stats.items():
            if count > 0:
                folder_summary += f"  ✅ {folder_path}: {count} images\n"
                valid_folders += 1
            else:
                folder_summary += f"  ⚠️ {folder_path}: 0 images\n"
        
        folder_summary += f"\n📊 Total: {len(all_image_paths)} images from {valid_folders} folders"
        
        yield folder_summary, gr.update(visible=False)
        
        # Input validation
        if not collection_name or not isinstance(collection_name, str):
            collection_name = f"whole_image_database_{int(time.time())}"
            print(f"[WARNING] Invalid collection name provided, using: {collection_name}")
        
        # Ensure parameters are in valid ranges
        optimal_layer = max(1, int(optimal_layer))
        
        # Get global model variables
        global pe_model, pe_vit_model, preprocess, device
        
        total = len(all_image_paths)
        print(f"[STATUS] Processing {total} images total for whole image embeddings")
        
        # Process images with progress updates
        client = None
        processed = 0
        failed = 0
        vector_dimension = None
        
        # Force database cleanup before starting
        from threading import Thread
        cleanup_thread = Thread(target=cleanup_qdrant_connections, args=(True,))
        cleanup_thread.start()
        cleanup_thread.join(timeout=10)
        
        # Use collection name provided (should already include layer info from caller)
        collection_with_layer = collection_name
        
        # Update Gradio with initial status
        yield (f"🔍 Starting to process {total} whole images from {valid_folders} folders\n"
               f"📊 Parameters:\n"
               f"  - Semantic Layer: {optimal_layer}\n"
               f"  - Resume from checkpoint: {resume_from_checkpoint}\n"
               f"  - Collection name: {collection_with_layer}"), gr.update(visible=False)
        
        print(f"[STATUS] Beginning whole image processing loop")
        for i, img_path in enumerate(all_image_paths):
            # Get just the filename for display
            img_file = os.path.basename(img_path)
            # Get the parent folder for context
            parent_folder = os.path.basename(os.path.dirname(img_path))
            
            # Yield progress update with percentage
            progress_pct = ((i+1) / total) * 100
            yield f"🔄 Processing: {i+1}/{total} images ({progress_pct:.1f}%)\n📄 Current: {img_file} (from {parent_folder})", gr.update(visible=False)
            
            print(f"[STATUS] Processing whole image {i+1}/{total}: {img_path}")
            
            try:
                # Add memory management
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                
                # Extract whole image embedding
                image, embedding, metadata, label = extract_whole_image_embeddings(
                    img_path,
                    pe_model_param=pe_model,
                    pe_vit_model_param=pe_vit_model,
                    preprocess_param=preprocess,
                    device_param=device,
                    optimal_layer=optimal_layer
                )
                
                if image is not None and embedding is not None:
                    print(f"[STATUS] Successfully extracted whole image embedding for {img_file}")
                    
                    # Initialize database if not done yet
                    if not client:
                        # Extract vector dimension from first embedding
                        if len(embedding.shape) == 2 and embedding.shape[0] == 1:
                            vector_dimension = embedding.shape[1]
                        elif len(embedding.shape) == 1:
                            vector_dimension = embedding.shape[0]
                        else:
                            vector_dimension = embedding.numel()
                        
                        print(f"[STATUS] Initializing whole image database with vector dimension: {vector_dimension}")
                        client = setup_qdrant(collection_with_layer, vector_dimension)
                        if client is None:
                            error_msg = "❌ Failed to initialize Qdrant client for whole images"
                            print(f"[ERROR] {error_msg}")
                            yield error_msg, gr.update(visible=False)
                            return
                    
                    # Add source info to metadata
                    metadata["processed_timestamp"] = time.time()
                    metadata["source_filename"] = img_file
                    metadata["source_folder"] = os.path.dirname(img_path)
                    metadata["full_path"] = img_path
                    
                    # Store embedding in database
                    store_embeddings_in_qdrant(client, collection_with_layer, [embedding], [metadata])
                    processed += 1
                else:
                    print(f"[WARNING] Failed to extract embedding for {img_file}")
                    failed += 1
                
            except Exception as e:
                print(f"[ERROR] Exception processing {img_path}: {e}")
                import traceback
                traceback.print_exc()
                failed += 1
                continue
            
            # Periodic progress updates
            if (i + 1) % 5 == 0 or i == total - 1:
                progress_stats = (
                    f"📊 Progress: {i+1}/{total} images ({(i+1)/total*100:.1f}%)\n"
                    f"✅ Processed: {processed} images\n"
                    f"❌ Failed: {failed} images"
                )
                yield progress_stats, gr.update(visible=False)
        
        # Final completion message
        completion_message = (
            f"✅ Whole Image Processing Complete!\n\n"
            f"📊 Final Statistics:\n"
            f"  - Total images processed: {processed}/{total}\n"
            f"  - Images failed: {failed}\n"
            f"  - Folders processed: {valid_folders}\n"
            f"  - Database collection: {collection_with_layer}\n"
            f"  - Embedding layer used: {optimal_layer}"
        )
        
        print(f"[STATUS] {completion_message}")
        yield completion_message, gr.update(visible=True)
        
        # Close database connection
        if client:
            client.close()
            
    except Exception as e:
        error_msg = f"❌ Fatal error during whole image processing: {str(e)}"
        print(f"[ERROR] {error_msg}")
        import traceback
        traceback.print_exc()
        yield error_msg, gr.update(visible=False)

if __name__ == "__main__":
    main()