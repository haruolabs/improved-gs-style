#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
import json
from random import randint
from utils.loss_utils import l1_loss, ssim
from utils.swdloss import VGG19
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, PILtoTorch
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, render_net_image
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from PIL import Image
import numpy as np
import csv
from datetime import datetime
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, extra_iterations=5000, style_json_name=None, record_gramloss=False, record_flops=False):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0

    style_data = None
    images_stylized_path = None
    masks_path = None
    
    if style_json_name is None:
        style_json_path = os.path.join(dataset.source_path, "style.json")
    else:
        style_json_path = style_json_name #os.path.join(dataset.source_path, style_json_name)
    
    print('Style JSON path:', style_json_path)
    if os.path.exists(style_json_path):
        print(f"Found style.json at {style_json_path}, loading...")
        try:
            with open(style_json_path, 'r') as f:
                style_data = json.load(f)
            print(f"Successfully loaded style data with {len(style_data)} entries")
            
            if "images_stylized_path" in style_data:
                images_stylized_path = style_data["images_stylized_path"]
                if isinstance(images_stylized_path, str):
                    images_stylized_path = [images_stylized_path]
                    print(f"Found single stylized images path: {images_stylized_path[0]}")
                elif isinstance(images_stylized_path, list):
                    print(f"Found multiple stylized images paths: {len(images_stylized_path)} paths")
                
                valid_paths = []
                for path in images_stylized_path:
                    if os.path.exists(path):
                        valid_paths.append(path)
                    else:
                        print(f"Warning: Stylized images path {path} does not exist")
                
                if not valid_paths:
                    images_stylized_path = None
                    print("No valid stylized images paths found")
                else:
                    images_stylized_path = valid_paths
                    print(f"Using {len(images_stylized_path)} valid stylized images paths")
                
                if images_stylized_path:
                    print("Preprocessing stylized images...")
                    train_cameras = scene.getTrainCameras()
                    
                    if train_cameras:
                        first_camera = train_cameras[0]
                        gt_image_shape = first_camera.original_image.shape
                        target_resolution = (gt_image_shape[2], gt_image_shape[1])  # (width, height)
                        
                        needs_resize = False
                        first_stylized_path = None
                        for path in images_stylized_path:
                            for ext in ['.jpg', '.JPG']:
                                test_stylized_path = os.path.join(path, f"{first_camera.image_name}{ext}")
                                if os.path.exists(test_stylized_path):
                                    first_stylized_path = test_stylized_path
                                    break
                            if first_stylized_path:
                                break
                        
                        if first_stylized_path:
                            try:
                                first_pil = Image.open(first_stylized_path)
                                first_stylized_size = (first_pil.width, first_pil.height)
                                needs_resize = first_stylized_size != target_resolution
                                if needs_resize:
                                    print(f"Stylized images need resizing from {first_stylized_size} to {target_resolution}")
                                else:
                                    print("Stylized images already match target resolution")
                            except Exception as e:
                                print(f"Error checking first stylized image: {e}")
                        
                        for camera in train_cameras:
                            camera_stylized_images = []
                            for path in images_stylized_path:
                                stylized_image_path = None
                                for ext in ['.jpg', '.JPG']:
                                    test_path = os.path.join(path, f"{camera.image_name}{ext}")
                                    if os.path.exists(test_path):
                                        stylized_image_path = test_path
                                        break
                                if stylized_image_path:
                                    try:
                                        pil_image = Image.open(stylized_image_path)
                                        
                                        if needs_resize:
                                            stylized_tensor = PILtoTorch(pil_image, target_resolution)
                                        else:
                                            np_image = np.array(pil_image) / 255.0
                                            stylized_tensor = torch.from_numpy(np_image.transpose(2, 0, 1)).float()
                                        
                                        camera_stylized_images.append(stylized_tensor)
                                    except Exception as e:
                                        print(f"Error preprocessing stylized image from {stylized_image_path}: {e}")
                            
                            camera.stylized_images = camera_stylized_images
                        
                        print(f"Preprocessed stylized images for {len(train_cameras)} cameras")
            
            if "masks_path" in style_data:
                masks_path = style_data["masks_path"]
                if isinstance(masks_path, str):
                    masks_path = [masks_path]
                    print(f"Found single masks path: {masks_path[0]}")
                elif isinstance(masks_path, list):
                    print(f"Found multiple masks paths: {len(masks_path)} paths")
                
                valid_paths = []
                for path in masks_path:
                    if os.path.exists(path):
                        valid_paths.append(path)
                    else:
                        print(f"Warning: Masks path {path} does not exist")
                
                if not valid_paths:
                    masks_path = None
                    print("No valid mask paths found")
                else:
                    masks_path = valid_paths
                    print(f"Using {len(masks_path)} valid mask paths")
                    
                    if train_cameras:
                        first_camera = train_cameras[0]
                        gt_image_shape = first_camera.original_image.shape
                        target_resolution = (gt_image_shape[2], gt_image_shape[1])  # (width, height)
                        
                        needs_resize = False
                        first_mask_path = None
                        for path in masks_path:
                            for ext in ['.jpg', '.JPG']:
                                test_mask_path = os.path.join(path, f"{first_camera.image_name}{ext}")
                                if os.path.exists(test_mask_path):
                                    first_mask_path = test_mask_path
                                    break
                            if first_mask_path:
                                break
                        
                        if first_mask_path:
                            try:
                                first_pil_mask = Image.open(first_mask_path)
                                first_mask_size = (first_pil_mask.width, first_pil_mask.height)
                                needs_resize = first_mask_size != target_resolution
                                if needs_resize:
                                    print(f"Mask images need resizing from {first_mask_size} to {target_resolution}")
                                else:
                                    print("Mask images already match target resolution")
                            except Exception as e:
                                print(f"Error checking first mask image: {e}")
                        
                        for camera in train_cameras:
                            camera_mask_images = []
                            for path in masks_path:
                                mask_image_path = None
                                for ext in ['.jpg', '.JPG']:
                                    test_path = os.path.join(path, f"{camera.image_name}{ext}")
                                    if os.path.exists(test_path):
                                        mask_image_path = test_path
                                        break
                                if mask_image_path:
                                    try:
                                        pil_mask = Image.open(mask_image_path)
                                        
                                        if needs_resize:
                                            mask_tensor = PILtoTorch(pil_mask, target_resolution)
                                        else:
                                            np_mask = np.array(pil_mask) / 255.0  # Normalize to [0, 1]
                                            if len(np_mask.shape) == 2:  # If grayscale, add channel dimension
                                                np_mask = np_mask[np.newaxis, :, :]
                                            elif len(np_mask.shape) == 3:  # If RGB, convert to CHW format
                                                np_mask = np_mask.transpose(2, 0, 1)
                                            mask_tensor = torch.from_numpy(np_mask).float()
                                        
                                        camera_mask_images.append(mask_tensor)
                                    except Exception as e:
                                        print(f"Error preprocessing mask image from {mask_image_path}: {e}")
                            
                            camera.mask_images = camera_mask_images
                        
                        print(f"Preprocessed mask images for {len(train_cameras)} cameras")
        except Exception as e:
            print(f"Error loading style.json: {e}")

    print('op args:', opt.iterations)
    if style_data is not None:
        opt.iterations += extra_iterations

    # SWD loss
    vgg = VGG19().to(torch.device("cuda"))
    vgg.load_state_dict(torch.load("models/vgg19.pth")) # Load VGG19 weights

    # Initialize gram loss recording if requested
    gram_loss_csv_file = None
    gram_loss_writer = None
    csv_file = None
    if record_gramloss:
        os.makedirs("temp", exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        gram_loss_csv_file = f"temp/gramloss_{timestamp}.csv"
        
        # Initialize CSV writer
        csv_file = open(gram_loss_csv_file, 'w', newline='')
        gram_loss_writer = csv.writer(csv_file)

    if record_gramloss and gram_loss_writer:
        initial_cam = scene.getTrainCameras()[0]
        with torch.no_grad():
            initial_render_pkg = render(initial_cam, gaussians, pipe, background)
            initial_image = initial_render_pkg["render"]
            initial_gt_image = initial_cam.original_image.cuda()
            
            initial_stylized_images = []
            if images_stylized_path and hasattr(initial_cam, 'stylized_images') and initial_cam.stylized_images:
                initial_stylized_images = [img.cuda() for img in initial_cam.stylized_images]
            
            initial_gram_loss = vgg.gram_loss(initial_image.unsqueeze(0), initial_gt_image.unsqueeze(0))
            
            if initial_stylized_images:
                initial_ref_image = initial_stylized_images[0]
                initial_slicing_loss = vgg.slicing_loss(initial_image.unsqueeze(0), initial_ref_image.unsqueeze(0))
            else:
                initial_slicing_loss = 0
            
            gram_loss_writer.writerow([0, initial_gram_loss.item(), initial_slicing_loss.item() if initial_slicing_loss != 0 else 0])
            csv_file.flush()
            print(f"Initial losses recorded - gram: {initial_gram_loss.item()}, slicing: {initial_slicing_loss.item() if initial_slicing_loss != 0 else 0}")

    # Initialize FLOPS recording if requested
    flops_csv_file = None
    flops_writer = None
    flops_csv_file_handle = None
    flops_data = []
    
    if record_flops:
        os.makedirs("temp", exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        flops_csv_file = f"temp/flops_{timestamp}.csv"
        
        flops_csv_file_handle = open(flops_csv_file, 'w', newline='')
        flops_writer = csv.writer(flops_csv_file_handle)
        flops_writer.writerow(['iteration', 'flops', 'runtime_ms'])
        flops_csv_file_handle.flush()
        print(f"FLOPS recording initialized: {flops_csv_file}")

    if_condition_printed = False

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 200 == 0: # 1000 -> 200
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        
        gt_image = viewpoint_cam.original_image.cuda()
        image_name = viewpoint_cam.image_name
        #print(f"Processing image: {image_name}") # frame_00004
        
        stylized_images = []
        if images_stylized_path and hasattr(viewpoint_cam, 'stylized_images') and viewpoint_cam.stylized_images:
            stylized_images = [img.cuda() for img in viewpoint_cam.stylized_images]
        
        mask_images = []
        if hasattr(viewpoint_cam, 'mask_images') and viewpoint_cam.mask_images:
            mask_images = [mask.cuda() for mask in viewpoint_cam.mask_images]
            #print(f"Using {len(mask_images)} preprocessed mask images for {image_name}")
        elif not mask_images and masks_path:
            print(f"No preprocessed mask images found for {image_name}, but masks_path is defined")
            #else:
            #    print(f"Loaded {len(mask_images)} mask images for {image_name}")
        
        if style_data and image_name in style_data:
            print(f"Found style data for image {image_name}: {style_data[image_name]}")
        
        Ll1 = l1_loss(image, gt_image) # torch.Size([3, 1036, 1600])
        #loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        # Content loss
        content_loss = vgg.content_loss(image.unsqueeze(0), stylized_images[0].unsqueeze(0)) if stylized_images else 0

        #loss = vgg.slicing_loss(image.unsqueeze(0), gt_image.unsqueeze(0)) # Vanilla SWD loss w/o masks
        #ref_image = stylized_image*mask_image + gt_image*(1-mask_image) # Region-based: only the masked area will be style-transferred
        sample_patch = False
        if sample_patch:
            import random
            _, H, W = image.shape
            crop_h = 256
            crop_w = 256
            top = random.randint(0, H - crop_h)
            left = random.randint(0, W - crop_w)
            image = image[:, top:top+crop_h, left:left+crop_w]
            gt_image = gt_image[:, top:top+crop_h, left:left+crop_w]
            stylized_images = [img[:, top:top+crop_h, left:left+crop_w] for img in stylized_images]
            mask_images = [m[:, top:top+crop_h, left:left+crop_w] for m in mask_images]
        
        loss = 0
        if mask_images:
            combined_mask = None
            for mask in mask_images:
                if combined_mask is None:
                    combined_mask = mask.clone()
                else:
                    combined_mask = torch.maximum(combined_mask, mask)
            
            comp_image = None
            if stylized_images and len(stylized_images) == len(mask_images):
                comp_image = gt_image.clone()
                for i, (mask, stylized_img) in enumerate(zip(mask_images, stylized_images)):
                    comp_image = comp_image * (1 - mask) + stylized_img * mask
            
            if stylized_images and combined_mask is not None:
                if not if_condition_printed:
                    print("INFO: Style transfer with combined mask is active (stylized_images and combined_mask both available)")
                    if_condition_printed = True
                    
                ref_image = stylized_images[0]
                
                # Record FLOPS and runtime if requested
                if record_flops:
                    import torch.profiler
                    with torch.profiler.profile(with_flops=True) as prof:
                        loss = vgg.region_based_swd_loss(image.unsqueeze(0), comp_image.unsqueeze(0), mask=mask_images)
                    
                    key_averages = prof.key_averages()
                    flops = sum([item.flops for item in key_averages])
                    runtime_ms = sum([item.cpu_time_total for item in key_averages]) / 1000
                    
                    flops_data.append((iteration, flops, runtime_ms))
                    
                    # Write to CSV every 50 iterations
                    if iteration % 50 == 0 and flops_writer:
                        flops_writer.writerow([iteration, flops, runtime_ms])
                        flops_csv_file_handle.flush()
                else:
                    loss = vgg.region_based_swd_loss(image.unsqueeze(0), comp_image.unsqueeze(0), mask=mask_images)
            elif stylized_images:
                ref_image = stylized_images[0]
                loss = vgg.slicing_loss(image.unsqueeze(0), ref_image.unsqueeze(0))
            else:
                loss = 0
        else:
            if stylized_images:
                ref_image = stylized_images[0]
                loss = vgg.slicing_loss(image.unsqueeze(0), ref_image.unsqueeze(0))
            else:
                loss = 0
        
        # regularization
        lambda_normal = opt.lambda_normal if iteration > 7000 else 0.0
        lambda_dist = opt.lambda_dist if iteration > 3000 else 0.0

        rend_dist = render_pkg["rend_dist"]
        rend_normal  = render_pkg['rend_normal']
        surf_normal = render_pkg['surf_normal']
        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        normal_loss = lambda_normal * (normal_error).mean()
        dist_loss = lambda_dist * (rend_dist).mean()

        # loss
        total_loss = loss + dist_loss + normal_loss + 0.1*content_loss
        
        total_loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_dist_for_log = 0.4 * dist_loss.item() + 0.6 * ema_dist_for_log
            ema_normal_for_log = 0.4 * normal_loss.item() + 0.6 * ema_normal_for_log

            if record_gramloss and gram_loss_writer and iteration % 50 == 0:
                target_image = comp_image.unsqueeze(0) if 'comp_image' in locals() and comp_image is not None else gt_image.unsqueeze(0)
                
                current_gram_loss = vgg.gram_loss(image.unsqueeze(0), target_image)
                
                current_slicing_loss = 0
                if stylized_images:
                    ref_image = stylized_images[0]
                    current_slicing_loss = vgg.slicing_loss(image.unsqueeze(0), ref_image.unsqueeze(0))
                
                gram_loss_writer.writerow([iteration, current_gram_loss.item(), current_slicing_loss.item() if current_slicing_loss != 0 else 0])
                csv_file.flush()
                print(f"Losses at iteration {iteration} - gram: {current_gram_loss.item()}, slicing: {current_slicing_loss.item() if current_slicing_loss != 0 else 0}")


            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "distort": f"{ema_dist_for_log:.{5}f}",
                    "normal": f"{ema_normal_for_log:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz)}"
                }
                progress_bar.set_postfix(loss_dict)

                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            if tb_writer is not None:
                tb_writer.add_scalar('train_loss_patches/dist_loss', ema_dist_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/normal_loss', ema_normal_for_log, iteration)

            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)


            # Densification
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.opacity_cull, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

        with torch.no_grad():        
            if network_gui.conn == None:
                network_gui.try_connect(dataset.render_items)
            while network_gui.conn != None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, keep_alive, scaling_modifer, render_mode = network_gui.receive()
                    if custom_cam != None:
                        render_pkg = render(custom_cam, gaussians, pipe, background, scaling_modifer)   
                        net_image = render_net_image(render_pkg, dataset.render_items, render_mode, custom_cam)
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    metrics_dict = {
                        "#": gaussians.get_opacity.shape[0],
                        "loss": ema_loss_for_log
                        # Add more metrics as needed
                    }
                    # Send the data
                    network_gui.send(net_image_bytes, dataset.source_path, metrics_dict)
                    if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                        break
                except Exception as e:
                    # raise e
                    network_gui.conn = None

    if record_gramloss and csv_file is not None:
        csv_file.close()
        print(f"Gram loss recording saved to: {gram_loss_csv_file}")

    if record_flops and flops_csv_file_handle is not None:
        if flops_data:
            valid_flops = [data[1] for data in flops_data if data[1] is not None and data[1] > 0]
            avg_flops = sum(valid_flops) / len(valid_flops) if valid_flops else 0
            avg_runtime = sum(data[2] for data in flops_data) / len(flops_data)
            
            flops_writer.writerow(['AVERAGE', avg_flops, avg_runtime])
            print(f"FLOPS Recording Summary:")
            print(f"  Average FLOPS: {avg_flops:.2e}")
            print(f"  Average Runtime: {avg_runtime:.2f} ms")
            print(f"  Total measurements: {len(flops_data)}")
        
        flops_csv_file_handle.close()
        print(f"FLOPS recording saved to: {flops_csv_file}")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

@torch.no_grad()
def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/reg_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0).to("cuda")
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        from utils.general_utils import colormap
                        depth = render_pkg["surf_depth"]
                        norm = depth.max()
                        depth = depth / norm
                        depth = colormap(depth.cpu().numpy()[0], cmap='turbo')
                        tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), depth[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)

                        try:
                            rend_alpha = render_pkg['rend_alpha']
                            rend_normal = render_pkg["rend_normal"] * 0.5 + 0.5
                            surf_normal = render_pkg["surf_normal"] * 0.5 + 0.5
                            tb_writer.add_images(config['name'] + "_view_{}/rend_normal".format(viewpoint.image_name), rend_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/surf_normal".format(viewpoint.image_name), surf_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/rend_alpha".format(viewpoint.image_name), rend_alpha[None], global_step=iteration)

                            rend_dist = render_pkg["rend_dist"]
                            rend_dist = colormap(rend_dist.cpu().numpy()[0])
                            tb_writer.add_images(config['name'] + "_view_{}/rend_dist".format(viewpoint.image_name), rend_dist[None], global_step=iteration)
                        except:
                            pass

                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--style_json", type=str, default=None, help="Name of stylization JSON config file")
    parser.add_argument("--extra_iterations", type=int, default = 5000)
    parser.add_argument("--record_gramloss", action="store_true", help="Record gram loss values during training")
    parser.add_argument("--record_flops", action="store_true", help="Record FLOPS and runtime for region_based_swd_loss function")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.extra_iterations, args.style_json, args.record_gramloss, args.record_flops)

    # All done
    print("\nTraining complete.")
