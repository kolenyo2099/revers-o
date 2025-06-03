import os
import tempfile
import shutil
import urllib.parse
import hashlib
import time

# Attempt to import yt_dlp
try:
    import yt_dlp
    YT_DLP_AVAILABLE = True
except ImportError:
    YT_DLP_AVAILABLE = False
    print("[WARNING] yt-dlp not available. URL video downloads will not work.")

# Import from current project structure
from video_processing.extraction import extract_keyframes_from_video, VIDEO_PROCESSING_AVAILABLE # Import VIDEO_PROCESSING_AVAILABLE too


def is_supported_video_url(url):
    """
    Check if the URL is from a supported video platform.
    """
    if not url or not isinstance(url, str):
        return False

    try:
        parsed = urllib.parse.urlparse(url.strip())
        domain = parsed.netloc.lower()

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
    """
    if not YT_DLP_AVAILABLE:
        return False, "yt-dlp not available. Please install it: pip install yt-dlp", None

    if not is_supported_video_url(url):
        return False, f"Unsupported URL or invalid format: {url}", None

    try:
        os.makedirs(output_dir, exist_ok=True)
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        timestamp = int(time.time())
        safe_filename = f"video_{timestamp}_{url_hash}.%(ext)s"

        ydl_opts = {
            'outtmpl': os.path.join(output_dir, safe_filename),
            'format': f'best[height<={max_quality[:-1]}]/best',
            'merge_output_format': 'mp4',
            'writeinfojson': False, 'writethumbnail': False, 'quiet': True,
            'no_warnings': True, 'restrictfilenames': True, 'windowsfilenames': True,
        }

        downloaded_files = []
        def progress_hook(d):
            if d['status'] == 'finished':
                file_path = d['filename']
                if os.path.exists(file_path) and not any(temp in os.path.basename(file_path) for temp in ['.f', '.part', '.temp']):
                    downloaded_files.append(file_path)

        ydl_opts['progress_hooks'] = [progress_hook]

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        time.sleep(2) # Allow fs to catch up

        if not downloaded_files: # Fallback scan
            video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v'}
            for file in os.listdir(output_dir):
                file_path = os.path.join(output_dir, file)
                if (os.path.splitext(file.lower())[1] in video_extensions and
                    os.path.isfile(file_path) and
                    not any(temp in file for temp in ['.f', '.part', '.temp']) and
                    os.path.getmtime(file_path) > time.time() - 120):
                    downloaded_files.append(file_path)

        if downloaded_files:
            downloaded_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            final_path = downloaded_files[0]
            file_size = os.path.getsize(final_path)
            return True, f"Successfully downloaded video ({file_size} bytes)", final_path
        else:
            return False, "Download completed but no valid video file was created", None

    except Exception as e:
        error_msg = str(e)
        if "Private video" in error_msg: return False, "Video is private or requires authentication", None
        elif "Video unavailable" in error_msg: return False, "Video is unavailable or has been removed", None
        elif "Unsupported URL" in error_msg: return False, f"URL not supported by yt-dlp: {url}", None
        else: return False, f"Download failed: {error_msg}", None

def process_video_urls(urls_text, output_folder, max_frames_per_video=30, scene_threshold=30.0, max_quality='720p'):
    """
    Process videos from URLs by downloading them first, then extracting keyframes.
    """
    if not VIDEO_PROCESSING_AVAILABLE: # This constant is now imported
        yield "❌ Video processing libraries not available"
        return

    if not YT_DLP_AVAILABLE:
        yield "❌ yt-dlp not available. Please install it: pip install yt-dlp"
        return

    urls = [url.strip() for url in urls_text.split('\n') if url.strip()]
    if not urls: yield "❌ No URLs provided"; return

    valid_urls = [url for url in urls if is_supported_video_url(url)]
    invalid_urls_count = len(urls) - len(valid_urls)
    if invalid_urls_count > 0: yield f"⚠️ Skipping {invalid_urls_count} invalid/unsupported URLs"
    if not valid_urls: yield "❌ No valid video URLs found"; return

    yield f"🔗 Found {len(valid_urls)} valid video URLs to process"

    temp_download_dir = tempfile.mkdtemp(prefix="revers_o_downloads_")

    try:
        total_extracted, successful_videos, downloaded_video_paths = 0, 0, [] # Renamed from downloaded_files

        yield f"📥 Starting downloads to temporary directory..."
        for i, url in enumerate(valid_urls):
            yield f"📥 Downloading video {i+1}/{len(valid_urls)}: {url}"
            try:
                success, message, downloaded_file_path = download_video_from_url(url, temp_download_dir, max_quality)
                if success and downloaded_file_path:
                    downloaded_video_paths.append(downloaded_file_path)
                    yield f"✅ Downloaded: {os.path.basename(downloaded_file_path)}"
                else: yield f"❌ Failed to download {url}: {message}"
            except Exception as e: yield f"❌ Error downloading {url}: {str(e)}"

        if not downloaded_video_paths: yield "❌ No videos were successfully downloaded"; return

        yield f"🎬 Successfully downloaded {len(downloaded_video_paths)} videos. Starting frame extraction..."

        for i, video_path in enumerate(downloaded_video_paths):
            video_name = os.path.basename(video_path)
            yield f"🎬 Processing video {i+1}/{len(downloaded_video_paths)}: {video_name}"
            video_frame_output_folder = os.path.join(output_folder, os.path.splitext(video_name)[0]) # Renamed
            try:
                success, message, extracted_frames = extract_keyframes_from_video(
                    video_path, video_frame_output_folder, max_frames_per_video, scene_threshold
                )
                if success:
                    total_extracted += len(extracted_frames)
                    successful_videos += 1
                    yield f"✅ {video_name}: {message}"
                else: yield f"❌ {video_name}: {message}"
            except Exception as e: yield f"❌ {video_name}: Error - {str(e)}"

        yield f"🎉 Processing complete! Successfully processed {successful_videos}/{len(downloaded_video_paths)} videos"
        yield f"📊 Total frames extracted: {total_extracted}"
        yield f"💾 Frames saved to: {output_folder}"

    finally:
        try:
            shutil.rmtree(temp_download_dir)
            yield f"🧹 Cleaned up temporary downloads"
        except Exception as e:
            yield f"⚠️ Warning: Could not clean up temporary directory: {str(e)}"
