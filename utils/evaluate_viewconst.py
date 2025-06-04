import os
import glob
import torch
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
from torchvision.models.optical_flow import raft_large
from utils.image_utils import mse

def load_images_from_directory(image_dir, sorted=True):
    """
    Load all png images from a directory and convert them to PyTorch tensors
    """
    image_paths = glob.glob(os.path.join(image_dir, "*.png"))
    if sorted:
        image_paths.sort()
    
    images = []
    print(f"Loading {len(image_paths)} images from {image_dir}")
    
    for img_path in tqdm(image_paths, desc="Loading images"):
        try:
            img = Image.open(img_path)
            if img.mode != "RGB":
                img = img.convert("RGB")
            img_tensor = torch.from_numpy(np.array(img)).float().permute(2, 0, 1).unsqueeze(0) / 255.0
            images.append(img_tensor)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
    
    if images:
        return torch.cat(images, dim=0)
    else:
        raise ValueError(f"No valid images found in {image_dir}")

def preprocess_for_raft(images):
    """
    Preprocess images for RAFT model: normalize to [-1, 1] and ensure dimensions divisible by 8
    """
    images = images * 2.0 - 1.0
    
    _, _, h, w = images.shape
    new_h = ((h + 7) // 8) * 8
    new_w = ((w + 7) // 8) * 8
    
    if h != new_h or w != new_w:
        images = F.interpolate(images, size=(new_h, new_w), mode='bilinear', align_corners=False)
    
    return images

def forward_warp(image, flow):
    """
    Forward warp an image using optical flow
    Args:
        image: (1, C, H, W) tensor
        flow: (1, 2, H, W) tensor with flow vectors
    Returns:
        warped_image: (1, C, H, W) tensor
    """
    _, _, img_h, img_w = image.shape
    _, _, flow_h, flow_w = flow.shape
    
    if img_h != flow_h or img_w != flow_w:
        flow = F.interpolate(flow, size=(img_h, img_w), mode='bilinear', align_corners=False)
        flow[:, 0] = flow[:, 0] * (img_w / flow_w)  # x-component
        flow[:, 1] = flow[:, 1] * (img_h / flow_h)  # y-component
    
    y_coords, x_coords = torch.meshgrid(
        torch.arange(img_h, dtype=torch.float32, device=image.device),
        torch.arange(img_w, dtype=torch.float32, device=image.device),
        indexing='ij'
    )
    
    x_coords = x_coords + flow[0, 0]
    y_coords = y_coords + flow[0, 1]
    
    x_coords = 2.0 * x_coords / (img_w - 1) - 1.0
    y_coords = 2.0 * y_coords / (img_h - 1) - 1.0
    
    grid = torch.stack([x_coords, y_coords], dim=-1).unsqueeze(0)
    
    warped = F.grid_sample(image, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    
    return warped

def save_debug_image(tensor, path):
    """
    Save a tensor as an image for debugging
    """
    img_np = (tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    img_pil = Image.fromarray(img_np)
    img_pil.save(path)

def main():
    parser = argparse.ArgumentParser(description="Evaluate temporal view consistency using optical flow")
    parser.add_argument("--imgs_dir", type=str, required=True, 
                       help="Path to directory containing sequential images to be evaluated")
    parser.add_argument("--imgs_orig_dir", type=str, required=True, 
                       help="Path to directory containing original sequential images")
    parser.add_argument("--debug", action="store_true", 
                       help="Save forward warped images to temp/ directory for debugging")
    
    args = parser.parse_args()
    
    try:
        print("Loading images...")
        images = load_images_from_directory(args.imgs_dir)
        orig_images = load_images_from_directory(args.imgs_orig_dir)
        
        min_len = min(images.shape[0], orig_images.shape[0])
        if images.shape[0] != orig_images.shape[0]:
            print(f"Warning: Number of images doesn't match. Using the first {min_len} images from both directories.")
            images = images[:min_len]
            orig_images = orig_images[:min_len]
        
        if min_len < 2:
            raise ValueError("Need at least 2 images for temporal consistency evaluation")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        images = images.to(device)
        orig_images = orig_images.to(device)
        
        print("Loading RAFT model...")
        raft_model = raft_large(pretrained=True, progress=False).to(device)
        raft_model.eval()
        
        orig_images_raft = preprocess_for_raft(orig_images)
        
        if args.debug:
            os.makedirs("temp", exist_ok=True)
        
        mse_values = []
        
        print("Computing optical flow and evaluating consistency...")
        for i in tqdm(range(1, min_len), desc="Processing image pairs"):
            img1 = orig_images_raft[i-1:i]
            img2 = orig_images_raft[i:i+1]
            
            with torch.no_grad():
                flow_predictions = raft_model(img1, img2)
                flow = flow_predictions[-1]
            
            warped_image = forward_warp(images[i-1:i], flow)
            
            mse_val = mse(warped_image, images[i:i+1])
            mse_values.append(mse_val.item())
            
            if args.debug:
                save_debug_image(warped_image, f"temp/warped_{i:04d}.png")
                save_debug_image(images[i:i+1], f"temp/original_{i:04d}.png")
        
        avg_mse = np.mean(mse_values)
        
        print(f"\nResults:")
        print(f"Number of image pairs processed: {len(mse_values)}")
        print(f"Average MSE (temporal view consistency): {avg_mse:.6f}")
        
        if args.debug:
            print(f"Debug images saved to temp/ directory")
        
        return avg_mse
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()
