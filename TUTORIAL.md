# Revers-o: A  Guide for Investigative Journalists

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

### Two Ways to Process Videos

Revers-o offers two methods for processing videos into searchable image databases:

#### Method 1: Local Video Files

**Step-by-Step Process**

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

#### Method 2: URL Video Download

**Supported Platforms:**
- YouTube (youtube.com, youtu.be)
- Twitter/X (twitter.com, x.com)
- Facebook (facebook.com, fb.watch)
- Instagram (instagram.com)
- TikTok (tiktok.com)
- Reddit (reddit.com)
- Vimeo (vimeo.com)
- And 1000+ other sites supported by yt-dlp

**Step-by-Step Process**

1. **Navigate to URL Download Tab**
   - In the "Extract Images from Video" section
   - Select the "🌐 From URL" tab

2. **Enter Video URL**
   - Paste the complete URL of the video
   - Examples:
     - `https://www.youtube.com/watch?v=VIDEO_ID`
     - `https://twitter.com/user/status/TWEET_ID`
     - `https://www.facebook.com/watch/?v=VIDEO_ID`

3. **Set Output Folder**
   - **Output Folder Path**: Where extracted frames will be saved
   - System will create subfolders automatically

4. **Configure Extraction Settings**
   - **Max Frames per Video**: How many frames to extract (default: 30)
   - **Scene Detection Threshold**: Sensitivity for detecting scene changes (default: 30.0)

5. **Download and Process**
   - Click "Download and Process Video"
   - The system will:
     - Download the video automatically
     - Extract keyframes using scene detection
     - Save frames to your specified folder
     - Show progress in real-time

**Important Notes for URL Downloads:**
- Video must be publicly accessible
- Some platforms may have regional restrictions
- Large videos may take time to download
- Downloaded videos are temporarily stored and then cleaned up
- **ffmpeg is required** for video processing (installed automatically by setup scripts)

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
