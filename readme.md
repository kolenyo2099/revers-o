# Revers-o: GroundedSAM + Perception Encoders Image Similarity Search

A streamlined visual investigation tool that combines GroundedSAM for zero-shot object detection with Meta's Perception Encoder for powerful visual similarity search.

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/revers-o.git
cd revers-o
```

### 2. Run the Setup Script
```bash
chmod +x setup.sh
./setup.sh
```

The setup script will:
- Install UV package manager if not present
- Create a Python virtual environment
- Install all required dependencies
- Set up the perception models repository
- Configure the system for optimal performance

### 3. Start the Application
```bash
./run.sh
```

## 📁 Project Structure
```
revers-o/
├── main.py              # Main application file
├── setup.sh             # Setup script
├── run.sh              # Run script
├── requirements.txt     # Python dependencies
├── config.py           # Configuration file
├── my_images/          # Your images folder
├── perception_models/  # Facebook perception models (auto-cloned)
├── models/             # Downloaded models
└── checkpoints/        # Model checkpoints
```

## ⚙️ Configuration

Edit `config.py` to customize:
- Model settings
- Processing parameters
- Hardware preferences
- Interface options

## 🎯 Features

- **Zero-shot Object Detection**: Detect objects using natural language prompts
- **Visual Similarity Search**: Find similar regions across your image collection
- **Video Frame Extraction**: Extract keyframes from videos for analysis
- **Apple Silicon Support**: Optimized for M1/M2 Macs
- **Simple Interface**: Clean and intuitive Gradio UI

## 🔧 Troubleshooting

1. **Setup fails**
   - Make sure you have Python 3.8+ installed
   - Check if you have git installed
   - Ensure you have write permissions in the directory

2. **Dependencies fail to install**
   - Try running: `uv pip install -r requirements.txt`
   - Check your internet connection
   - Make sure you have enough disk space

3. **Models fail to download**
   - Check your internet connection
   - Ensure you have enough disk space
   - Try running the setup script again

4. **Perception models clone fails**
   - Check your internet connection
   - Make sure git is installed
   - Try cloning manually:
     ```bash
     git clone https://github.com/facebookresearch/perception_models.git
     ```

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Facebook Research for the Perception Encoder
- GroundedSAM team for the zero-shot detection model
- Meta AI Research for their groundbreaking work in computer vision