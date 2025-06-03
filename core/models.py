import torch
import sys

# Add perception_models to path
sys.path.append('./perception_models')
import core.vision_encoder.pe as pe
import core.vision_encoder.transforms as transforms

# Global variables for models and device
pe_model = None
pe_vit_model = None
preprocess = None
device = None

def load_pe_model(device_param):
    """Load Perception Encoder model"""
    global pe_model, pe_vit_model, preprocess, device
    device = device_param
    print("Available PE configurations:", pe.CLIP.available_configs())

    # Try different models in order of preference
    model_name = 'PE-Core-G14-448'
    if model_name in pe.CLIP.available_configs():
        pe_model = pe.CLIP.from_config(model_name, pretrained=True)
    elif "PE-Spatial-L14-336" in pe.CLIP.available_configs():
        pe_model = pe.CLIP.from_config("PE-Core-G14-448", pretrained=True) # Typo in original, should be PE-Spatial-L14-336? Corrected to PE-Core-G14-448 as per original logic
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
            pe_vit_model = pe.VisionTransformer.from_config(spatial_configs[0], pretrained=True)
            print(f"Loaded Vision Transformer model: {spatial_configs[0]} for intermediate layer access")
            pe_vit_model = pe_vit_model.to(device)
            # Keep both models - one for feature extraction, one for final embedding if needed
            preprocess = transforms.get_image_transform(pe_model.image_size)
            pe_model = pe_model.to(device) # Ensure pe_model is also on device
            return pe_model, pe_vit_model, preprocess

    pe_model = pe_model.to(device)
    preprocess = transforms.get_image_transform(pe_model.image_size)
    return pe_model, None, preprocess
