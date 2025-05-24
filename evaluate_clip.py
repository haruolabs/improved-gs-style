#
#
#
# CLIP-based metrics for 3D editing

import os
import glob
import torch
import argparse
from PIL import Image
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from metric.clip_metrics import ClipSimilarity

def load_images_from_directory(image_dir, sorted=True):
    """
    Load all jpg images from a directory and convert them to PyTorch tensors
    """
    image_paths = glob.glob(os.path.join(image_dir, "*.png"))
    if sorted:
        image_paths.sort()
    print(image_dir, image_paths)
    images = []
    #print(f"Loading {len(image_paths)} images from {image_dir}")
    
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
        return torch.cat(images, dim=0).cuda()
    else:
        raise ValueError(f"No valid images found in {image_dir}")

def clip_temporal_consistency(clipsim, x, x_gt):
    """
    Calculate CLIP directional consistency as described
    x, x_gt: tensors of shape [b, c, h, w]
    """
    dx_gt = clipsim.encode_image(x[1:]) - clipsim.encode_image(x_gt[1:])
    dx = clipsim.encode_image(x[:-1]) - clipsim.encode_image(x_gt[:-1])
    cos_temp = F.cosine_similarity(dx, dx_gt, dim=1)
    return cos_temp.mean().item()

def main():
    parser = argparse.ArgumentParser(description="Evaluate CLIP-based metrics between sets of images")
    parser.add_argument("--imgs_dir", type=str, required=True, help="Path to directory containing the stylized set of images")
    parser.add_argument("--imgs_orig_dir", type=str, required=True, help="Path to directory containing the original set of images")
    parser.add_argument("--prompt", type=str, default="", help="Text prompt corresponding to imgs")
    parser.add_argument("--prompt_orig", type=str, default="", help="Text prompt corresponding to imgs_orig")
    parser.add_argument("--clip_model", type=str, default="ViT-L/14", help="CLIP model to use")
    
    args = parser.parse_args()
    
    try:
        imgs = load_images_from_directory(args.imgs_dir)
        imgs_orig = load_images_from_directory(args.imgs_orig_dir)
        
        min_len = min(imgs.shape[0], imgs_orig.shape[0])
        if imgs.shape[0] != imgs_orig.shape[0]:
            print(f"Warning: Number of images doesn't match. Using the first {min_len} images from both directories.")
            imgs = imgs[:min_len]
            imgs_orig = imgs_orig[:min_len]
        
        print(f"Initializing CLIP model: {args.clip_model}")
        clipsim = ClipSimilarity(args.clip_model)
        clipsim = clipsim.cuda()
        
        if args.prompt and args.prompt_orig:
            print("Calculating CLIP text-image directional similarity...")
            sim_0, sim_1, sim_direction, sim_image = clipsim(imgs, imgs_orig, args.prompt, args.prompt_orig)
            clip_tids = sim_direction.mean().item()
            print(f"CLIP text-image directional similarity: {clip_tids:.6f}")
        else:
            print("Skipping CLIP text-image directional similarity (prompts not provided)")
            clip_tids = None
        
        print("Calculating CLIP directional consistency...")
        clip_dc = clip_temporal_consistency(clipsim, imgs, imgs_orig)
        print(f"CLIP directional consistency: {clip_dc:.6f}")
        
        print("\nResults Summary:")
        print(f"Number of images processed: {min_len}")
        if clip_tids is not None:
            print(f"CLIP text-image directional similarity: {clip_tids:.6f}")
        print(f"CLIP directional consistency: {clip_dc:.6f}")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
