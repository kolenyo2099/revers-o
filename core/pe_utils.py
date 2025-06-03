import torch
import numpy as np # Required by tokens_to_grid if it were here, but it's in image_processing.utils

# PE Enhancement Utilities - Multi-Layer Feature Extraction and Advanced Pooling
def get_preset_configuration(preset_name):
    """Return layer and weight configurations for PE-optimized presets based on Meta's research"""
    presets = {
        "object_focused": {
            "layers": [25, 30, 35], "weights": [0.3, 0.4, 0.3], "pooling": "pe_attention"
        },
        "spatial_focused": {
            "layers": [15, 20, 25], "weights": [0.4, 0.4, 0.2], "pooling": "pe_spatial"
        },
        "semantic_focused": {
            "layers": [40, 45, 47], "weights": [0.2, 0.3, 0.5], "pooling": "pe_semantic"
        },
        "texture_focused": {
            "layers": [10, 15, 20], "weights": [0.5, 0.3, 0.2], "pooling": "pe_spatial"
        }
    }
    return presets.get(preset_name, presets["object_focused"])

def extract_multi_layer_features(image_input, pe_vit_model, layers, device): # Renamed 'image' to 'image_input' to avoid conflict
    """Extract features from multiple PE layers simultaneously"""
    if pe_vit_model is None:
        raise ValueError("PE ViT model is required for multi-layer extraction")

    layer_features = []
    with torch.no_grad():
        for layer_idx in layers: # Renamed 'layer' to 'layer_idx'
            try:
                safe_layer = max(1, layer_idx)
                features = pe_vit_model.forward_features(image_input, layer_idx=safe_layer)
                layer_features.append(features)
                print(f"[STATUS] Extracted features from layer {safe_layer}")
            except Exception as e:
                print(f"[WARNING] Failed to extract from layer {layer_idx}: {e}")
                try:
                    fallback_features = pe_vit_model.forward_features(image_input, layer_idx=40) # Default fallback
                    layer_features.append(fallback_features)
                    print(f"[STATUS] Used fallback layer 40 for failed layer {layer_idx}")
                except Exception as e_fallback:
                    print(f"[ERROR] Fallback layer also failed for layer {layer_idx}: {e_fallback}")
                    continue # Skip this layer if fallback fails

    if not layer_features: # If no features were extracted from any layer (even fallbacks)
        # Attempt to get at least one feature vector from a default layer as a last resort
        print(f"[ERROR] No features extracted from specified layers. Attempting default layer 40.")
        try:
            default_features = pe_vit_model.forward_features(image_input, layer_idx=40)
            if default_features is not None:
                 layer_features.append(default_features)
                 print(f"[STATUS] Successfully used default layer 40 as a final fallback.")
            else: # This case should ideally not be reached if model is valid
                 raise RuntimeError("Failed to extract any features even from default layer.")
        except Exception as e_default_final:
             print(f"[CRITICAL ERROR] Final fallback to default layer also failed: {e_default_final}")
             raise RuntimeError("No valid features extracted from any layer, including default.")

    return layer_features


def combine_layer_features(layer_features, weights, temperature=0.07):
    """Combine multi-layer features with learned weights and temperature scaling"""
    if not layer_features: # Handle case where no features could be extracted
        raise ValueError("Cannot combine features: layer_features list is empty.")
    if len(layer_features) != len(weights):
        # If layer extraction failed for some, weights might not match.
        # Attempt to use only as many weights as there are successfully extracted features.
        print(f"[WARNING] Mismatch between layer_features ({len(layer_features)}) and weights ({len(weights)}). Using subset of weights.")
        weights = weights[:len(layer_features)]
        # Ensure weights are re-normalized if subset is used
        if not weights: # If for some reason weights list becomes empty
             raise ValueError("Weights list became empty after adjusting for feature mismatch.")

    weights_tensor = torch.tensor(weights, dtype=torch.float32, device=layer_features[0].device)
    weights_tensor = weights_tensor / weights_tensor.sum() # Normalize

    if temperature > 0:
        weights_tensor = torch.softmax(weights_tensor / temperature, dim=0)

    combined_features = sum(w * feat for w, feat in zip(weights_tensor, layer_features))
    print(f"[STATUS] Combined {len(layer_features)} layers with weights {weights_tensor.cpu().numpy()} and temperature {temperature}")
    return combined_features

def _pe_pooling_base(masked_features, temperature, strategy_name, attention_calculator):
    if masked_features.shape[0] == 0:
        print(f"[WARNING] {strategy_name} pooling received empty masked_features. Returning mean if possible or zero vector.")
        # Attempt to return mean, but if feature_dim is 0, this will fail.
        # It's better to return a zero vector of expected dimension if known, or handle upstream.
        # For now, let's assume masked_features has a second dimension (feature_dim)
        if masked_features.shape[1] > 0:
            return masked_features.mean(dim=0, keepdim=True)
        else: # This case should ideally not happen if inputs are validated
            return torch.zeros((1,0), device=masked_features.device) # Placeholder for zero-dim features

    attention_scores = attention_calculator(masked_features)
    attention_weights = torch.softmax(attention_scores / temperature, dim=0)
    pooled = (masked_features * attention_weights.unsqueeze(1)).sum(dim=0, keepdim=True)
    return pooled

def pe_attention_pooling(masked_features, temperature=0.07, attention_heads=8):
    """PE-optimized attention-weighted pooling with temperature scaling"""
    def calculate_attention(features):
        feature_dim = features.shape[1]
        head_dim = feature_dim // attention_heads
        if head_dim * attention_heads != feature_dim:
            current_attention_heads = 1
            current_head_dim = feature_dim
        else:
            current_attention_heads = attention_heads
            current_head_dim = head_dim

        reshaped = features.view(features.shape[0], current_attention_heads, current_head_dim)
        scores = torch.matmul(reshaped, reshaped.transpose(-2,-1))
        return scores.mean(dim=1).sum(dim=-1) # Sum over keys for each query, then mean over heads

    return _pe_pooling_base(masked_features, temperature, "PE Attention", calculate_attention)

def pe_spatial_pooling(masked_features, temperature=0.07):
    """PE-optimized spatial pooling focusing on boundary information"""
    def calculate_spatial_scores(features):
        norms = features.norm(dim=1)
        if features.shape[0] > 1:
            variance = features.var(dim=0, keepdim=True)
            importance = variance.norm(dim=1, keepdim=True)
            return norms + importance.squeeze()
        return norms
    return _pe_pooling_base(masked_features, temperature, "PE Spatial", calculate_spatial_scores)

def pe_semantic_pooling(masked_features, temperature=0.07):
    """PE-optimized semantic pooling for high-level concept extraction"""
    def calculate_semantic_scores(features):
        importance = features.abs().mean(dim=0)
        semantic_weights = torch.softmax(importance / temperature, dim=0) # Temp applied early here
        weighted_features = features * semantic_weights.unsqueeze(0)
        # Using sum of dot products of weighted features as attention scores
        return torch.matmul(weighted_features, weighted_features.t()).sum(dim=1)
    return _pe_pooling_base(masked_features, temperature, "PE Semantic", calculate_semantic_scores)

def pe_adaptive_pooling(masked_features, temperature=0.07):
    """PE-optimized adaptive pooling that combines multiple strategies"""
    if masked_features.shape[0] == 0: return masked_features.mean(dim=0, keepdim=True) # Fallback for empty

    attention_pooled = pe_attention_pooling(masked_features, temperature)
    spatial_pooled = pe_spatial_pooling(masked_features, temperature)
    semantic_pooled = pe_semantic_pooling(masked_features, temperature)

    num_features = masked_features.shape[0]
    weights = [0.4, 0.3, 0.3] # Default balanced weights
    if num_features > 100: weights = [0.3, 0.5, 0.2]  # More spatial
    elif num_features < 20: weights = [0.2, 0.2, 0.6]  # More semantic

    combined = (weights[0] * attention_pooled + weights[1] * spatial_pooled + weights[2] * semantic_pooled)
    return combined

def temperature_scaled_similarity(query_emb, candidate_emb, temperature=0.07):
    """PE-inspired temperature-scaled cosine similarity"""
    import torch.nn.functional as F # Keep import local if only used here
    query_emb_norm = F.normalize(query_emb, p=2, dim=-1)
    candidate_emb_norm = F.normalize(candidate_emb, p=2, dim=-1)
    cosine_sim = F.cosine_similarity(query_emb_norm, candidate_emb_norm, dim=-1)
    scaled_sim = cosine_sim / temperature
    return torch.sigmoid(scaled_sim) if scaled_sim.dim() == 0 or scaled_sim.size(0) == 1 else torch.softmax(scaled_sim, dim=0)

def apply_spatial_pooling_enhanced(masked_features, strategy="top_k", top_k_ratio=0.1,
                                 temperature=0.07, attention_heads=8):
    """Enhanced pooling with PE-optimized strategies"""
    if masked_features.shape[0] == 0: # Handle empty input
        # Return a zero tensor of appropriate shape if possible, or raise error
        # Assuming feature dimension can be inferred or is fixed. For now, returning as is.
        # This case should ideally be handled before calling pooling, by ensuring non-empty features.
        print("[WARNING] apply_spatial_pooling_enhanced received empty masked_features.")
        # Fallback to a zero vector of the same feature dimension
        if masked_features.shape[1] > 0:
             return torch.zeros((1, masked_features.shape[1]), device=masked_features.device)
        else: # Cannot determine feature dimension
             return torch.zeros((1,0), device=masked_features.device)


    if strategy == "max": return masked_features.max(dim=0, keepdim=True)[0]
    elif strategy == "top_k":
        norms = masked_features.norm(dim=1)
        k = max(1, int(top_k_ratio * len(norms)))
        top_k_indices = torch.topk(norms, k)[1]
        return masked_features[top_k_indices].mean(dim=0, keepdim=True)
    elif strategy == "attention": # Original simple attention
        norms = masked_features.norm(dim=1)
        att_weights = torch.softmax(norms, dim=0)
        return (masked_features * att_weights.unsqueeze(1)).sum(dim=0, keepdim=True)
    elif strategy == "average": return masked_features.mean(dim=0, keepdim=True)
    elif strategy == "pe_attention": return pe_attention_pooling(masked_features, temperature, attention_heads)
    elif strategy == "pe_spatial": return pe_spatial_pooling(masked_features, temperature)
    elif strategy == "pe_semantic": return pe_semantic_pooling(masked_features, temperature)
    elif strategy == "pe_adaptive": return pe_adaptive_pooling(masked_features, temperature)
    else: return masked_features.mean(dim=0, keepdim=True) # Default fallback

def get_optimal_single_layer(task_type="general"):
    """Return optimal single layer based on Meta PE research findings"""
    optimal_layers = {
        "general": 30, "object": 30, "spatial": 20,
        "semantic": 45, "texture": 15, "fine_detail": 12, "global_context": 42
    }
    return optimal_layers.get(task_type, 30)
