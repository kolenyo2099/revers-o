import os
import cv2
from PIL import Image
import traceback # Added for error printing

# Attempt to import scenedetect and imageio
try:
    from scenedetect import VideoManager, SceneManager
    from scenedetect.detectors import ContentDetector
    import imageio # Often used with scenedetect or for direct video read/write
    VIDEO_PROCESSING_AVAILABLE = True
except ImportError as e:
    print(f"[WARNING] Video processing libraries (scenedetect or imageio) not available: {e}")
    VIDEO_PROCESSING_AVAILABLE = False

def extract_keyframes_from_video(video_path, output_folder, max_frames=30, scene_threshold=30.0):
    """
    Extract keyframes from a video using scene detection and uniform sampling.
    """
    if not VIDEO_PROCESSING_AVAILABLE:
        return False, "Video processing libraries not available. Please install scenedetect and imageio.", []

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
        os.makedirs(output_folder, exist_ok=True)
        print(f"[DEBUG] Initializing video manager for: {video_path}")

        video_manager = VideoManager([video_path])
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=scene_threshold))

        video_manager.set_duration()
        video_manager.start()
        scene_manager.detect_scenes(frame_source=video_manager)
        scene_list = scene_manager.get_scene_list()
        video_manager.release()

        print(f"[VIDEO] Detected {len(scene_list)} scenes in {video_path}")

        if not scene_list:
            return extract_uniform_frames(video_path, output_folder, max_frames)

        extracted_frames = []
        frames_per_scene = max(1, max_frames // len(scene_list))

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[ERROR] OpenCV could not open video: {video_path}")
            return False, f"Could not open video file with OpenCV: {video_path}", []

        total_frames_cv = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # Renamed to avoid conflict
        fps = cap.get(cv2.CAP_PROP_FPS)

        print(f"[DEBUG] Video properties - Total frames: {total_frames_cv}, FPS: {fps}")

        frame_count = 0
        for i, scene in enumerate(scene_list):
            if frame_count >= max_frames:
                break
            start_time = scene[0].get_seconds()
            end_time = scene[1].get_seconds()
            scene_duration = end_time - start_time
            if scene_duration > 0:
                for j in range(frames_per_scene):
                    if frame_count >= max_frames:
                        break
                    time_offset = (j + 0.5) * scene_duration / frames_per_scene
                    frame_time = start_time + time_offset
                    frame_number = int(frame_time * fps)

                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                    ret, frame = cap.read()

                    if ret:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        video_name_base = os.path.splitext(os.path.basename(video_path))[0] # Renamed
                        frame_filename = f"{video_name_base}_scene{i:03d}_frame{j:03d}.jpg"
                        frame_path = os.path.join(output_folder, frame_filename)

                        pil_image = Image.fromarray(frame_rgb)
                        pil_image.save(frame_path, 'JPEG', quality=95)

                        extracted_frames.append(frame_path)
                        frame_count += 1
                        print(f"[VIDEO] Extracted frame {frame_count}/{max_frames}: {frame_filename}")

        cap.release()
        return True, f"Successfully extracted {len(extracted_frames)} keyframes from {len(scene_list)} scenes", extracted_frames

    except Exception as e:
        print(f"[ERROR] Video processing failed: {e}")
        traceback.print_exc()
        return False, f"Error processing video: {str(e)}", []

def extract_uniform_frames(video_path, output_folder, max_frames=30):
    """
    Extract frames uniformly distributed across the video duration.
    """
    if not VIDEO_PROCESSING_AVAILABLE: # Check again in case this function is called directly
        return False, "Video processing libraries not available for uniform extraction.", []
    try:
        os.makedirs(output_folder, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        total_frames_cv = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # Renamed
        fps = cap.get(cv2.CAP_PROP_FPS)
        # duration = total_frames_cv / fps # This was not used, removed

        print(f"[VIDEO] Video has {total_frames_cv} frames, {total_frames_cv / fps:.2f} seconds duration")

        extracted_frames = []
        frame_interval = max(1, total_frames_cv // max_frames)

        for i in range(0, total_frames_cv, frame_interval):
            if len(extracted_frames) >= max_frames:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()

            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_name_base = os.path.splitext(os.path.basename(video_path))[0] # Renamed
                frame_filename = f"{video_name_base}_uniform_{len(extracted_frames):03d}.jpg"
                frame_path = os.path.join(output_folder, frame_filename)

                pil_image = Image.fromarray(frame_rgb)
                pil_image.save(frame_path, 'JPEG', quality=95)

                extracted_frames.append(frame_path)
                print(f"[VIDEO] Extracted uniform frame {len(extracted_frames)}/{max_frames}: {frame_filename}")

        cap.release()
        return True, f"Successfully extracted {len(extracted_frames)} frames uniformly", extracted_frames

    except Exception as e:
        print(f"[ERROR] Uniform frame extraction failed: {e}")
        traceback.print_exc()
        return False, f"Error extracting uniform frames: {str(e)}", []

def process_video_folder(input_folder, output_folder, max_frames_per_video=30, scene_threshold=30.0):
    """
    Process all videos in a folder to extract keyframes.
    """
    if not VIDEO_PROCESSING_AVAILABLE:
        yield "❌ Video processing libraries not available"
        return

    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v'}
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
