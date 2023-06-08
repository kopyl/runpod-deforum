import os
import shutil
import subprocess
import time
import gc
import sys

import torch
import random
from collections import OrderedDict
from types import SimpleNamespace
from omegaconf import OmegaConf

sys.path.insert(0, "src")
import clip

from ldm.util import instantiate_from_config
from helpers.render import (
    render_animation,
    render_input_video,
    render_image_batch,
    render_interpolation,
)
from helpers.model_load import make_linear_decode
from helpers.aesthetics import load_aesthetics_model


MODEL_CACHE = os.getenv("SD_MODELS_DIR", "diffusion_models_cache")


def predict(
    model_checkpoint="Protogen_V2.2.ckpt",
    max_frames=10,
    animation_prompts="0: a beautiful apple, trending on Artstation | 5: a beautiful banana, trending on Artstation",
    width=512,
    height=512,
    num_inference_steps=50,
    guidance_scale=7,
    sampler="euler_ancestral",
    seed=None,
    fps=15,
    clip_name="ViT-L/14",
    use_init=False,
    init_image=None,
    strength=0.5,
    use_mask=False,
    mask_file=None,
    invert_mask=False,
    animation_mode="2D",
    border="replicate",
    angle="0:(0)",
    zoom="0:(1.04)",
    translation_x="0:(10*sin(2*3.14*t/10))",
    translation_y="0:(0)",
    translation_z="0:(10)",
    rotation_3d_x="0:(0)",
    rotation_3d_y="0:(0)",
    rotation_3d_z="0:(0)",
    flip_2d_perspective=False,
    perspective_flip_theta="0:(0)",
    perspective_flip_phi="0:(t%15)",
    perspective_flip_gamma="0:(0)",
    perspective_flip_fv="0:(53)",
    noise_schedule="0: (0.02)",
    strength_schedule="0: (0.65)",
    contrast_schedule="0: (1.0)",
    hybrid_video_comp_alpha_schedule="0:(1)",
    hybrid_video_comp_mask_blend_alpha_schedule="0:(0.5)",
    hybrid_video_comp_mask_contrast_schedule="0:(1)",
    hybrid_video_comp_mask_auto_contrast_cutoff_high_schedule="0:(100)",
    hybrid_video_comp_mask_auto_contrast_cutoff_low_schedule="0:(0)",
    kernel_schedule="0: (5)",
    sigma_schedule="0: (1.0)",
    amount_schedule="0: (0.2)",
    threshold_schedule="0: (0.0)",
    color_coherence="Match Frame 0 LAB",
    color_coherence_video_every_N_frames=1,
    diffusion_cadence="1",
    use_depth_warping=True,
    midas_weight=0.3,
    near_plane=200,
    far_plane=10000,
    fov=40,
    padding_mode="border",
    sampling_mode="bicubic",
    video_init_path=None,
    extract_nth_frame=1,
    overwrite_extracted_frames=True,
    use_mask_video=False,
    video_mask_path=None,
    hybrid_video_generate_inputframes=False,
    hybrid_video_use_first_frame_as_init_image=True,
    hybrid_video_motion="None",
    hybrid_video_flow_method="Farneback",
    hybrid_video_composite=False,
    hybrid_video_comp_mask_type="None",
    hybrid_video_comp_mask_inverse=False,
    hybrid_video_comp_mask_equalize="None",
    hybrid_video_comp_mask_auto_contrast=False,
    hybrid_video_comp_save_extra_frames=False,
    hybrid_video_use_video_as_mse_image=False,
    interpolate_key_frames=False,
    interpolate_x_frames=4,
    resume_from_timestring=False,
    resume_timestring="",
) -> str:
    """Run a single prediction on the model"""

    # sanity checks:
    if use_init:
        assert init_image, "Please provide init_image when use_init is set to True."
    if use_mask:
        assert mask_file, "Please provide mask_file when use_mask is set to True."

    animation_prompts_dict = {}
    animation_prompts = animation_prompts.split("|")
    assert len(animation_prompts) > 0, "Please provide valid prompt for animation."
    if len(animation_prompts) == 1:
        animation_prompts = {0: animation_prompts[0]}
    else:
        for frame_prompt in animation_prompts:
            frame_prompt = frame_prompt.split(":")
            assert (
                len(frame_prompt) == 2
            ), "Please follow the 'frame_num: prompt' format."
            frame_id, prompt = frame_prompt[0].strip(), frame_prompt[1].strip()
            assert (
                frame_id.isdigit() and 0 <= int(frame_id) <= max_frames
            ), "frame_num should be an integer and 0<= frame_num <= max_frames"
            assert (
                int(frame_id) not in animation_prompts_dict
            ), f"Duplicate prompts for frame_num {frame_id}. "
            assert len(prompt) > 0, "prompt cannot be empty"
            animation_prompts_dict[int(frame_id)] = prompt
        animation_prompts = OrderedDict(sorted(animation_prompts_dict.items()))

    root = {"device": "cuda", "models_path": "models", "configs_path": "configs"}
    model_config = (
        "v2-inference.yaml"
        if model_checkpoint in ["v2-1_768-ema-pruned.ckpt", "v2-1_512-ema-pruned.ckpt"]
        else "v1-inference.yaml"
    )
    ckpt_config_path = f"configs/{model_config}"
    ckpt_path = os.path.join(MODEL_CACHE, model_checkpoint)
    local_config = OmegaConf.load(ckpt_config_path)

    model = load_model_from_config(local_config, ckpt_path, map_location="cuda")
    model.to("cuda")
    root["model"] = model

    root = SimpleNamespace(**root)

    autoencoder_version = "sd-v1"  # TODO this will be different for different models
    root.model.linear_decode = make_linear_decode(autoencoder_version, "cuda")

    # using some of the default settings for simplicity
    args_dict = {
        "W": width,
        "H": height,
        "bit_depth_output": 8,
        "seed": seed,
        "sampler": sampler,
        "steps": num_inference_steps,
        "scale": guidance_scale,
        "ddim_eta": 0.0,
        "dynamic_threshold": None,
        "static_threshold": None,
        "save_samples": False,
        "save_settings": False,
        "display_samples": False,
        "save_sample_per_step": False,
        "show_sample_per_step": False,
        "prompt_weighting": True,
        "normalize_prompt_weights": True,
        "log_weighted_subprompts": False,
        "n_batch": 1,
        "batch_name": "StableFun",
        "filename_format": "{timestring}_{index}_{prompt}.png",
        "seed_behavior": "iter",
        "seed_iter_N": 1,
        "make_grid": False,
        "grid_rows": 2,
        "outdir": "cog_temp_output",
        "use_init": use_init,
        "strength": strength,
        "strength_0_no_init": True,
        "init_image": init_image,
        "use_mask": use_mask,
        "use_alpha_as_mask": False,
        "mask_file": mask_file,
        "invert_mask": invert_mask,
        "mask_brightness_adjust": 1.0,
        "mask_contrast_adjust": 1.0,
        "overlay_mask": True,
        "mask_overlay_blur": 5,
        "mean_scale": 0,
        "var_scale": 0,
        "exposure_scale": 0,
        "exposure_target": 0.5,
        "colormatch_scale": 0,
        "colormatch_image": "https://www.saasdesign.io/wp-content/uploads/2021/02/palette-3-min-980x588.png",
        "colormatch_n_colors": 4,
        "ignore_sat_weight": 0,
        "clip_name": clip_name,
        "clip_scale": 0,
        "aesthetics_scale": 0,
        "cutn": 1,
        "cut_pow": 0.0001,
        "init_mse_scale": 0,
        "init_mse_image": "https://cdn.pixabay.com/photo/2022/07/30/13/10/green-longhorn-beetle-7353749_1280.jpg",
        "blue_scale": 0,
        "gradient_wrt": "x0_pred",
        "gradient_add_to": "both",
        "decode_method": "linear",
        "grad_threshold_type": "dynamic",
        "clamp_grad_threshold": 0.2,
        "clamp_start": 0.2,
        "clamp_stop": 0.01,
        "grad_inject_timing": [1, 2, 3, 4, 5, 6, 7, 8, 9],
        "cond_uncond_sync": True,
        "n_samples": 1,
        "precision": "autocast",
        "C": 4,
        "f": 8,
        "prompt": "",
        "timestring": "",
        "init_latent": None,
        "init_sample": None,
        "init_sample_raw": None,
        "mask_sample": None,
        "init_c": None,
        "seed_internal": 0,
    }

    anim_args_dict = {
        "animation_mode": animation_mode,
        "max_frames": max_frames,
        "border": border,
        "angle": angle,
        "zoom": zoom,
        "translation_x": translation_x,
        "translation_y": translation_y,
        "translation_z": translation_z,
        "rotation_3d_x": rotation_3d_x,
        "rotation_3d_y": rotation_3d_y,
        "rotation_3d_z": rotation_3d_z,
        "flip_2d_perspective": flip_2d_perspective,
        "perspective_flip_theta": perspective_flip_theta,
        "perspective_flip_phi": perspective_flip_phi,
        "perspective_flip_gamma": perspective_flip_gamma,
        "perspective_flip_fv": perspective_flip_fv,
        "noise_schedule": noise_schedule,
        "strength_schedule": strength_schedule,
        "contrast_schedule": contrast_schedule,
        "hybrid_video_comp_alpha_schedule": hybrid_video_comp_alpha_schedule,
        "hybrid_video_comp_mask_blend_alpha_schedule": hybrid_video_comp_mask_blend_alpha_schedule,
        "hybrid_video_comp_mask_contrast_schedule": hybrid_video_comp_mask_contrast_schedule,
        "hybrid_video_comp_mask_auto_contrast_cutoff_high_schedule": hybrid_video_comp_mask_auto_contrast_cutoff_high_schedule,
        "hybrid_video_comp_mask_auto_contrast_cutoff_low_schedule": hybrid_video_comp_mask_auto_contrast_cutoff_low_schedule,
        "kernel_schedule": kernel_schedule,
        "sigma_schedule": sigma_schedule,
        "amount_schedule": amount_schedule,
        "threshold_schedule": threshold_schedule,
        "color_coherence": color_coherence,
        "color_coherence_video_every_N_frames": color_coherence_video_every_N_frames,
        "diffusion_cadence": diffusion_cadence,
        "use_depth_warping": use_depth_warping,
        "midas_weight": midas_weight,
        "near_plane": near_plane,
        "far_plane": far_plane,
        "fov": fov,
        "padding_mode": padding_mode,
        "sampling_mode": sampling_mode,
        "save_depth_maps": False,
        "video_init_path": str(video_init_path),
        "extract_nth_frame": extract_nth_frame,
        "overwrite_extracted_frames": overwrite_extracted_frames,
        "use_mask_video": use_mask_video,
        "video_mask_path": str(video_mask_path),
        "hybrid_video_generate_inputframes": hybrid_video_generate_inputframes,
        "hybrid_video_use_first_frame_as_init_image": hybrid_video_use_first_frame_as_init_image,
        "hybrid_video_motion": hybrid_video_motion,
        "hybrid_video_flow_method": hybrid_video_flow_method,
        "hybrid_video_composite": hybrid_video_composite,
        "hybrid_video_comp_mask_type": hybrid_video_comp_mask_type,
        "hybrid_video_comp_mask_inverse": hybrid_video_comp_mask_inverse,
        "hybrid_video_comp_mask_equalize": hybrid_video_comp_mask_equalize,
        "hybrid_video_comp_mask_auto_contrast": hybrid_video_comp_mask_auto_contrast,
        "hybrid_video_comp_save_extra_frames": hybrid_video_comp_save_extra_frames,
        "hybrid_video_use_video_as_mse_image": hybrid_video_use_video_as_mse_image,
        "interpolate_key_frames": interpolate_key_frames,
        "interpolate_x_frames": interpolate_x_frames,
        "resume_from_timestring": resume_from_timestring,
        "resume_timestring": resume_timestring,
    }

    args = SimpleNamespace(**args_dict)
    anim_args = SimpleNamespace(**anim_args_dict)

    if os.path.exists(args.outdir):
        shutil.rmtree(args.outdir)
    os.makedirs(args.outdir, exist_ok=True)

    args.timestring = time.strftime("%Y%m%d%H%M%S")
    args.strength = max(0.0, min(1.0, args.strength))

    # Load clip model if using clip guidance
    if (args.clip_scale > 0) or (args.aesthetics_scale > 0):
        root.clip_model = (
            clip.load(args.clip_name, jit=False)[0]
            .eval()
            .requires_grad_(False)
            .to(root.device)
        )
        if args.aesthetics_scale > 0:
            root.aesthetics_model = load_aesthetics_model(args, root)

    if args.seed is None:
        args.seed = random.randint(0, 2 ** 32 - 1)
    if not args.use_init:
        args.init_image = None
    if args.sampler == "plms" and (args.use_init or anim_args.animation_mode != "None"):
        print(f"Init images aren't supported with PLMS yet, switching to KLMS")
        args.sampler = "klms"
    if args.sampler != "ddim":
        args.ddim_eta = 0

    if anim_args.animation_mode == "None":
        anim_args.max_frames = 1
    elif anim_args.animation_mode == "Video Input":
        args.use_init = True

    # clean up unused memory
    gc.collect()
    torch.cuda.empty_cache()

    # dispatch to appropriate renderer
    if anim_args.animation_mode == "2D" or anim_args.animation_mode == "3D":
        render_animation(args, anim_args, animation_prompts, root)
    elif anim_args.animation_mode == "Video Input":
        render_input_video(args, anim_args, animation_prompts, root)
    elif anim_args.animation_mode == "Interpolation":
        render_interpolation(args, anim_args, animation_prompts, root)

    # make video
    image_path = os.path.join(args.outdir, f"{args.timestring}_%05d.png")
    mp4_path = f"/tmp/out.mp4"

    # make video
    cmd = [
        "ffmpeg",
        "-y",
        "-vcodec",
        "png",
        "-r",
        str(fps),
        "-start_number",
        str(0),
        "-i",
        image_path,
        "-frames:v",
        str(anim_args.max_frames),
        "-c:v",
        "libx264",
        "-vf",
        f"fps={fps}",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "17",
        "-preset",
        "veryfast",
        mp4_path,
    ]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print(stderr)
        raise RuntimeError(stderr)

    return mp4_path


def load_model_from_config(
    config, ckpt, verbose=False, device="cuda", print_flag=False, map_location="cuda"
):
    print(f"..loading model")
    _, extension = os.path.splitext(ckpt)
    if extension.lower() == ".safetensors":
        import safetensors.torch

        pl_sd = safetensors.torch.load_file(ckpt, device=map_location)
    else:
        pl_sd = torch.load(ckpt, map_location=map_location)
    try:
        sd = pl_sd["state_dict"]
    except:
        sd = pl_sd
    torch.set_default_dtype(torch.float16)
    model = instantiate_from_config(config.model)
    torch.set_default_dtype(torch.float32)
    m, u = model.load_state_dict(sd, strict=False)
    if print_flag:
        if len(m) > 0 and verbose:
            print("missing keys:")
            print(m)
        if len(u) > 0 and verbose:
            print("unexpected keys:")
            print(u)

    model = model.half().to(device)
    model.eval()
    return model
