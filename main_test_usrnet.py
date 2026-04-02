import os
import os.path
import json
import argparse
import logging
from collections import OrderedDict

import numpy as np
from scipy.io import loadmat
from scipy import ndimage

import torch

from utils import utils_deblur
from utils import utils_logger
from utils import utils_sisr as sr
from utils import utils_image as util
from models.network_usrnet import USRNet as net


"""
Refactored testing script for USRNet.

Examples
--------
1) Synthetic benchmark mode (generate LR from GT/HR and evaluate)
python main_test_usrnet.py \
    --model_name usrnet \
    --testset_name set5 \
    --scale 2 3 4 \
    --kernel_indices 1 2 3

2) Test a custom trained checkpoint on a GT folder with x4 synthetic degradation
python main_test_usrnet.py \
    --model_name my_usrnet \
    --model_path SR/my-usrnet/models/4200_G.pth \
    --folder_gt testsets/DIV2K/DIV2K_valid_HR \
    --scale 4 \
    --kernel_indices 1

3) Direct inference on existing LR images, optionally with GT for metrics
python main_test_usrnet.py \
    --model_name my_usrnet \
    --model_path SR/my-usrnet/models/4200_G.pth \
    --folder_lq testsets/my_lq \
    --folder_gt testsets/my_gt \
    --scale 4 \
    --kernel_indices 1
"""


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y", "t"}:
        return True
    if value in {"false", "0", "no", "n", "f"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def load_json_with_comments(json_path):
    json_str = ""
    with open(json_path, "r", encoding="utf-8") as f:
        for line in f:
            json_str += line.split("//")[0] + "\n"
    return json.loads(json_str)


def default_model_config(model_name):
    if "tiny" in model_name.lower():
        return {
            "n_iter": 6,
            "h_nc": 32,
            "in_nc": 4,
            "out_nc": 3,
            "nc": [16, 32, 64, 64],
            "nb": 2,
            "act_mode": "R",
            "downsample_mode": "strideconv",
            "upsample_mode": "convtranspose",
        }
    return {
        "n_iter": 8,
        "h_nc": 64,
        "in_nc": 4,
        "out_nc": 3,
        "nc": [64, 128, 256, 512],
        "nb": 2,
        "act_mode": "R",
        "downsample_mode": "strideconv",
        "upsample_mode": "convtranspose",
    }


def model_config_from_opt(opt_path):
    opt = load_json_with_comments(opt_path)
    net_opt = opt["netG"]
    config = {
        "n_iter": net_opt["n_iter"],
        "h_nc": net_opt["h_nc"],
        "in_nc": net_opt["in_nc"],
        "out_nc": net_opt["out_nc"],
        "nc": net_opt["nc"],
        "nb": net_opt["nb"],
        "act_mode": net_opt["act_mode"],
        "downsample_mode": net_opt["downsample_mode"],
        "upsample_mode": net_opt["upsample_mode"],
    }
    return config, opt


def merge_model_config(base_cfg, args):
    cfg = dict(base_cfg)
    if args.n_iter is not None:
        cfg["n_iter"] = args.n_iter
    if args.h_nc is not None:
        cfg["h_nc"] = args.h_nc
    if args.in_nc is not None:
        cfg["in_nc"] = args.in_nc
    if args.out_nc is not None:
        cfg["out_nc"] = args.out_nc
    if args.nc is not None:
        cfg["nc"] = args.nc
    if args.nb is not None:
        cfg["nb"] = args.nb
    if args.act_mode is not None:
        cfg["act_mode"] = args.act_mode
    if args.downsample_mode is not None:
        cfg["downsample_mode"] = args.downsample_mode
    if args.upsample_mode is not None:
        cfg["upsample_mode"] = args.upsample_mode
    return cfg


def resolve_model_path(args):
    if args.model_path:
        return args.model_path
    return os.path.join(args.model_pool, args.model_name + ".pth")


def resolve_scales(args, train_opt):
    if args.scale is not None:
        return args.scale
    if train_opt is not None and "scale" in train_opt:
        return [int(train_opt["scale"])]
    if "gan" in args.model_name.lower():
        return [4]
    return [2, 3, 4]


def resolve_result_name(args, synthetic_mode, folder_lq, folder_gt):
    if args.result_name:
        return args.result_name
    if synthetic_mode:
        data_name = os.path.basename(os.path.normpath(folder_gt))
    else:
        data_name = os.path.basename(os.path.normpath(folder_lq))
    return f"{data_name}_{args.model_name}"


def resolve_save_dir(args, result_name):
    if args.save_dir:
        return args.save_dir
    return os.path.join(args.results, result_name)


def resolve_device(device_str):
    if device_str:
        return torch.device(device_str)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_state_dict_safely(model, model_path, strict=True):
    state = torch.load(model_path, map_location="cpu")
    if isinstance(state, dict):
        for key in ("params", "params_ema", "state_dict", "model", "model_state_dict"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break

    if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
        state = {k[7:]: v for k, v in state.items()}

    model.load_state_dict(state, strict=strict)


def load_kernel_bank(kernel_path):
    ext = os.path.splitext(kernel_path)[1].lower()

    if ext == ".mat":
        data = loadmat(kernel_path)
        if "kernels" in data:
            kernels = data["kernels"]
            if isinstance(kernels, np.ndarray) and kernels.dtype == np.object_:
                return [np.asarray(k, dtype=np.float64) for k in kernels.flatten()]
            if kernels.ndim == 2:
                return [kernels.astype(np.float64)]
            if kernels.ndim == 3:
                return [kernels[i].astype(np.float64) for i in range(kernels.shape[0])]
            if kernels.ndim == 4:
                return [kernels[0, i].astype(np.float64) for i in range(kernels.shape[1])]
        raise ValueError(f"Cannot find valid 'kernels' in: {kernel_path}")

    if ext == ".npy":
        kernels = np.load(kernel_path)
        if kernels.ndim == 2:
            return [kernels.astype(np.float64)]
        if kernels.ndim == 3:
            return [kernels[i].astype(np.float64) for i in range(kernels.shape[0])]
        raise ValueError(f"Unsupported .npy kernel shape: {kernels.shape}")

    raise ValueError(f"Unsupported kernel file: {kernel_path}")


def select_kernels(kernel_bank, kernel_indices):
    if kernel_indices is None or len(kernel_indices) == 0:
        selected = []
        for idx, kernel in enumerate(kernel_bank, start=1):
            kernel = kernel.astype(np.float64)
            kernel /= np.sum(kernel)
            selected.append((idx, kernel))
        return selected

    selected = []
    total = len(kernel_bank)
    for idx in kernel_indices:
        if idx < 1 or idx > total:
            raise ValueError(f"kernel index {idx} out of range, expected 1..{total}")
        kernel = kernel_bank[idx - 1].astype(np.float64)
        kernel /= np.sum(kernel)
        selected.append((idx, kernel))
    return selected


def build_output_path(input_path, input_root, save_dir, suffix):
    rel_path = os.path.relpath(input_path, input_root)
    stem, _ = os.path.splitext(rel_path)
    out_path = os.path.join(save_dir, stem + suffix + ".png")
    util.mkdir(os.path.dirname(out_path))
    return out_path


def image_stem(input_path, input_root):
    rel_path = os.path.relpath(input_path, input_root)
    stem, _ = os.path.splitext(rel_path)
    return stem.replace("\\", "/")


def prepare_sigma(noise_level_model, device):
    sigma = torch.tensor(noise_level_model / 255.0).float().view(1, 1, 1, 1)
    return sigma.to(device)


def inference_usrnet(model, img_l, kernel, sf, sigma, device):
    x = util.single2tensor4(img_l).to(device)
    k = util.single2tensor4(kernel[..., np.newaxis].astype(np.float32)).to(device)
    with torch.no_grad():
        img_e = model(x, k, sf, sigma)
    return util.tensor2uint(img_e)


def maybe_add_noise(img_l, noise_level_img, seed):
    if noise_level_img <= 0:
        return img_l
    if seed >= 0:
        np.random.seed(seed)
    return img_l + np.random.normal(0, noise_level_img / 255.0, img_l.shape)


def synthesize_lq_from_gt(img_h, kernel, sf, noise_level_img, seed):
    img_l = ndimage.convolve(img_h, kernel[..., np.newaxis], mode="wrap")
    img_l = sr.downsample_np(img_l, sf, center=False)
    img_l = util.uint2single(img_l)
    img_l = maybe_add_noise(img_l, noise_level_img, seed)
    return img_l


def prepare_metric_pair(img_e, img_h):
    if img_e.shape == img_h.shape:
        return img_e, img_h

    h = min(img_e.shape[0], img_h.shape[0])
    w = min(img_e.shape[1], img_h.shape[1])
    if img_e.ndim == 3 and img_h.ndim == 3:
        return img_e[:h, :w, :], img_h[:h, :w, :]
    return img_e[:h, :w], img_h[:h, :w]


def init_metric_dict():
    metrics = OrderedDict()
    metrics["psnr"] = []
    metrics["ssim"] = []
    metrics["psnr_y"] = []
    metrics["ssim_y"] = []
    return metrics


def update_metrics(metrics, img_e, img_h, border):
    psnr = util.calculate_psnr(img_e, img_h, border=border)
    ssim = util.calculate_ssim(img_e, img_h, border=border)
    metrics["psnr"].append(psnr)
    metrics["ssim"].append(ssim)

    if np.ndim(img_h) == 3:
        img_e_y = util.rgb2ycbcr(img_e, only_y=True)
        img_h_y = util.rgb2ycbcr(img_h, only_y=True)
        psnr_y = util.calculate_psnr(img_e_y, img_h_y, border=border)
        ssim_y = util.calculate_ssim(img_e_y, img_h_y, border=border)
        metrics["psnr_y"].append(psnr_y)
        metrics["ssim_y"].append(ssim_y)
    else:
        metrics["psnr_y"] = metrics["psnr"]
        metrics["ssim_y"] = metrics["ssim"]

    return psnr, ssim


def build_gt_lookup(gt_paths):
    gt_map = {}
    for path in gt_paths:
        stem = os.path.splitext(os.path.basename(path))[0]
        gt_map[stem] = path
    return gt_map


def parse_args():
    parser = argparse.ArgumentParser(description="USRNet testing/inference script")

    parser.add_argument("--model_name", type=str, default="usrnet",
                        help="Model name for logging/result naming. Example: usrnet, usrgan, usrnet_tiny, my_usrnet")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to the checkpoint. If omitted, uses <model_pool>/<model_name>.pth")
    parser.add_argument("--opt", type=str, default=None,
                        help="Optional training JSON file. If provided, netG config will be read from it")

    parser.add_argument("--testset_name", type=str, default="set5",
                        help="Default testset name under --testsets when folder_gt/folder_lq are not given")
    parser.add_argument("--folder_gt", type=str, default=None,
                        help="GT/HR image folder. In synthetic mode this is the input HR folder")
    parser.add_argument("--folder_lq", type=str, default=None,
                        help="Existing LR image folder. If omitted, synthetic degradation mode is used")

    parser.add_argument("--model_pool", type=str, default="model_zoo", help="Path to model_zoo")
    parser.add_argument("--testsets", type=str, default="testsets", help="Path to testsets root")
    parser.add_argument("--results", type=str, default="results", help="Path to results root")
    parser.add_argument("--result_name", type=str, default=None, help="Optional result folder name")
    parser.add_argument("--save_dir", type=str, default=None, help="Optional explicit output folder")

    parser.add_argument("--kernel_path", type=str, default=os.path.join("kernels", "kernels_12.mat"),
                        help="Path to a .mat or .npy kernel bank/file")
    parser.add_argument("--kernel_indices", nargs="+", type=int, default=None,
                        help="1-based kernel indices. Default: use all kernels in the bank")

    parser.add_argument("--scale", "--scales", nargs="+", type=int, default=None, dest="scale",
                        help="Scale factor(s). Default: [4] for GAN, else [2,3,4], or opt['scale'] when --opt is used")
    parser.add_argument("--noise_level_img", type=float, default=0.0,
                        help="Noise level added to synthetic LR image, in [0,255]")
    parser.add_argument("--noise_level_model", type=float, default=None,
                        help="Noise level fed to the model, in [0,255]. Default: same as noise_level_img")
    parser.add_argument("--border", type=int, default=None,
                        help="Border for PSNR/SSIM. Default: sf**2 to match original USRNet script")

    parser.add_argument("--save_L", type=str2bool, default=True, help="Save LR images")
    parser.add_argument("--save_E", type=str2bool, default=True, help="Save restored images")
    parser.add_argument("--save_LEH", type=str2bool, default=False, help="Save concatenated LR/E/GT visualization")
    parser.add_argument("--show_img", type=str2bool, default=False, help="Visualize intermediate images")
    parser.add_argument("--strict", type=str2bool, default=True, help="Strictly load checkpoint weights")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for synthetic noise. Set < 0 to disable deterministic noise")
    parser.add_argument("--device", type=str, default=None,
                        help="Device string, e.g. cuda, cuda:0, cpu")

    parser.add_argument("--n_channels", type=int, default=3, help="Image channels for reading images")
    parser.add_argument("--n_iter", type=int, default=None, help="Override netG.n_iter")
    parser.add_argument("--h_nc", type=int, default=None, help="Override netG.h_nc")
    parser.add_argument("--in_nc", type=int, default=None, help="Override netG.in_nc")
    parser.add_argument("--out_nc", type=int, default=None, help="Override netG.out_nc")
    parser.add_argument("--nc", nargs="+", type=int, default=None, help="Override netG.nc, e.g. --nc 64 128 256 512")
    parser.add_argument("--nb", type=int, default=None, help="Override netG.nb")
    parser.add_argument("--act_mode", type=str, default=None, help="Override netG.act_mode")
    parser.add_argument("--downsample_mode", type=str, default=None, help="Override netG.downsample_mode")
    parser.add_argument("--upsample_mode", type=str, default=None, help="Override netG.upsample_mode")

    return parser.parse_args()


def main():
    args = parse_args()

    train_opt = None
    if args.opt is not None:
        base_cfg, train_opt = model_config_from_opt(args.opt)
    else:
        base_cfg = default_model_config(args.model_name)
    model_cfg = merge_model_config(base_cfg, args)

    model_path = resolve_model_path(args)
    scales = resolve_scales(args, train_opt)
    noise_level_model = args.noise_level_img if args.noise_level_model is None else args.noise_level_model

    synthetic_mode = args.folder_lq is None
    if synthetic_mode:
        folder_gt = args.folder_gt or os.path.join(args.testsets, args.testset_name)
        folder_lq = None
    else:
        folder_lq = args.folder_lq
        folder_gt = args.folder_gt

    result_name = resolve_result_name(args, synthetic_mode, folder_lq, folder_gt)
    save_dir = resolve_save_dir(args, result_name)
    util.mkdir(save_dir)

    logger_name = result_name
    utils_logger.logger_info(logger_name, log_path=os.path.join(save_dir, logger_name + ".log"))
    logger = logging.getLogger(logger_name)

    device = resolve_device(args.device)
    sigma = prepare_sigma(noise_level_model, device)

    model = net(
        n_iter=model_cfg["n_iter"],
        h_nc=model_cfg["h_nc"],
        in_nc=model_cfg["in_nc"],
        out_nc=model_cfg["out_nc"],
        nc=model_cfg["nc"],
        nb=model_cfg["nb"],
        act_mode=model_cfg["act_mode"],
        downsample_mode=model_cfg["downsample_mode"],
        upsample_mode=model_cfg["upsample_mode"],
    )
    load_state_dict_safely(model, model_path, strict=args.strict)
    model.eval()
    for _, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)

    num_params = sum(map(lambda x: x.numel(), model.parameters()))
    logger.info("Model path: %s", model_path)
    logger.info("Params number: %d", num_params)
    logger.info("Synthetic mode: %s", synthetic_mode)
    logger.info("Scales: %s", scales)
    logger.info("Kernel path: %s", args.kernel_path)
    logger.info("Model config: %s", model_cfg)
    logger.info("Noise level image/model: %.4f / %.4f", args.noise_level_img, noise_level_model)

    kernel_bank = load_kernel_bank(args.kernel_path)
    selected_kernels = select_kernels(kernel_bank, args.kernel_indices)

    if synthetic_mode:
        gt_paths = util.get_image_paths(folder_gt)
        source_root = folder_gt
        logger.info("GT folder: %s", folder_gt)

        all_runs = []
        for sf in scales:
            border = args.border if args.border is not None else sf ** 2

            for kernel_idx, kernel in selected_kernels:
                logger.info("Running synthetic mode - scale x%d, kernel %d", sf, kernel_idx)
                metrics = init_metric_dict()

                for idx, gt_path in enumerate(gt_paths, start=1):
                    img_name = image_stem(gt_path, source_root)

                    img_h = util.imread_uint(gt_path, n_channels=args.n_channels)
                    img_h = util.modcrop(img_h, np.lcm(sf, 8))

                    img_l = synthesize_lq_from_gt(
                        img_h=img_h,
                        kernel=kernel,
                        sf=sf,
                        noise_level_img=args.noise_level_img,
                        seed=args.seed,
                    )

                    if args.show_img:
                        util.imshow(util.single2uint(img_l), title=f"LR - {img_name}")

                    img_e = inference_usrnet(
                        model=model,
                        img_l=img_l,
                        kernel=kernel,
                        sf=sf,
                        sigma=sigma,
                        device=device,
                    )

                    if args.save_E:
                        e_path = build_output_path(
                            gt_path,
                            source_root,
                            save_dir,
                            f"_x{sf}_k{kernel_idx}_{args.model_name}",
                        )
                        util.imsave(img_e, e_path)

                    img_l_uint = util.single2uint(img_l)
                    if args.save_L:
                        l_path = build_output_path(
                            gt_path,
                            source_root,
                            save_dir,
                            f"_x{sf}_k{kernel_idx}_LR",
                        )
                        util.imsave(img_l_uint, l_path)

                    if args.save_LEH:
                        k_v = kernel / np.max(kernel) * 1.2
                        k_v = util.single2uint(np.tile(k_v[..., np.newaxis], [1, 1, 3]))
                        k_v = util.imresize_np(k_v, 3, True)
                        k_v = util.single2uint(k_v) if k_v.dtype != np.uint8 else k_v

                        img_i = util.imresize_np(img_l_uint, sf, True)
                        img_i = util.single2uint(img_i) if img_i.dtype != np.uint8 else img_i
                        if img_i.ndim == 2:
                            img_i = np.expand_dims(img_i, axis=2)
                        if img_i.shape[2] == 1:
                            img_i = np.repeat(img_i, 3, axis=2)

                        kh, kw = k_v.shape[:2]
                        ih, iw = img_i.shape[:2]
                        img_i[:min(kh, ih), max(0, iw - kw):, :] = k_v[:min(kh, ih), :min(kw, iw), :]

                        vis = np.concatenate([img_i, img_e, img_h], axis=1)
                        leh_path = build_output_path(
                            gt_path,
                            source_root,
                            save_dir,
                            f"_x{sf}_k{kernel_idx}_LEH",
                        )
                        util.imsave(vis, leh_path)

                    img_e_eval, img_h_eval = prepare_metric_pair(img_e, img_h)
                    psnr, ssim = update_metrics(metrics, img_e_eval, img_h_eval, border=border)

                    logger.info(
                        "%4d --> %s -- x%d -- k%d -- PSNR: %.2f dB; SSIM: %.4f",
                        idx, img_name, sf, kernel_idx, psnr, ssim
                    )

                ave_psnr = sum(metrics["psnr"]) / len(metrics["psnr"])
                ave_ssim = sum(metrics["ssim"]) / len(metrics["ssim"])
                logger.info(
                    "Average PSNR/SSIM(RGB) - %s - x%d - k%d -- PSNR: %.2f dB; SSIM: %.4f",
                    result_name, sf, kernel_idx, ave_psnr, ave_ssim
                )
                if len(metrics["psnr_y"]) > 0:
                    ave_psnr_y = sum(metrics["psnr_y"]) / len(metrics["psnr_y"])
                    ave_ssim_y = sum(metrics["ssim_y"]) / len(metrics["ssim_y"])
                    logger.info(
                        "Average PSNR/SSIM(Y)   - %s - x%d - k%d -- PSNR: %.2f dB; SSIM: %.4f",
                        result_name, sf, kernel_idx, ave_psnr_y, ave_ssim_y
                    )

                all_runs.append(
                    {"scale": sf, "kernel": kernel_idx, "psnr": ave_psnr, "ssim": ave_ssim}
                )

        logger.info("Summary: %s", all_runs)
        return

    # direct LR mode
    lq_paths = util.get_image_paths(folder_lq)
    source_root = folder_lq
    logger.info("LQ folder: %s", folder_lq)

    need_h = folder_gt is not None
    gt_paths = util.get_image_paths(folder_gt) if need_h else None
    gt_map = build_gt_lookup(gt_paths) if need_h else {}
    if need_h:
        logger.info("GT folder: %s", folder_gt)

    for sf in scales:
        border = args.border if args.border is not None else sf ** 2

        for kernel_idx, kernel in selected_kernels:
            logger.info("Running direct mode - scale x%d, kernel %d", sf, kernel_idx)
            metrics = init_metric_dict()

            for idx, lq_path in enumerate(lq_paths, start=1):
                img_name = image_stem(lq_path, source_root)

                img_l_uint = util.imread_uint(lq_path, n_channels=args.n_channels)
                img_l = util.uint2single(img_l_uint)

                if args.show_img:
                    util.imshow(util.single2uint(img_l), title=f"LQ - {img_name}")

                img_e = inference_usrnet(
                    model=model,
                    img_l=img_l,
                    kernel=kernel,
                    sf=sf,
                    sigma=sigma,
                    device=device,
                )

                if args.save_E:
                    e_path = build_output_path(
                        lq_path,
                        source_root,
                        save_dir,
                        f"_x{sf}_k{kernel_idx}_{args.model_name}",
                    )
                    util.imsave(img_e, e_path)

                if args.save_L:
                    l_path = build_output_path(
                        lq_path,
                        source_root,
                        save_dir,
                        f"_x{sf}_k{kernel_idx}_LR",
                    )
                    util.imsave(img_l_uint, l_path)

                if not need_h:
                    logger.info("%4d --> %s -- x%d -- k%d", idx, img_name, sf, kernel_idx)
                    continue

                stem = os.path.splitext(os.path.basename(lq_path))[0]
                gt_path = gt_map.get(stem, gt_paths[idx - 1] if idx - 1 < len(gt_paths) else None)
                if gt_path is None:
                    logger.warning("Missing GT for %s, skip metrics.", lq_path)
                    continue

                img_h = util.imread_uint(gt_path, n_channels=args.n_channels)
                img_h = util.modcrop(img_h, sf)
                img_e_eval, img_h_eval = prepare_metric_pair(img_e, img_h)
                psnr, ssim = update_metrics(metrics, img_e_eval, img_h_eval, border=border)

                if args.save_LEH:
                    vis = np.concatenate([img_l_uint, img_e_eval, img_h_eval], axis=1)
                    leh_path = build_output_path(
                        lq_path,
                        source_root,
                        save_dir,
                        f"_x{sf}_k{kernel_idx}_LEH",
                    )
                    util.imsave(vis, leh_path)

                logger.info(
                    "%4d --> %s -- x%d -- k%d -- PSNR: %.2f dB; SSIM: %.4f",
                    idx, img_name, sf, kernel_idx, psnr, ssim
                )

            if need_h and len(metrics["psnr"]) > 0:
                ave_psnr = sum(metrics["psnr"]) / len(metrics["psnr"])
                ave_ssim = sum(metrics["ssim"]) / len(metrics["ssim"])
                logger.info(
                    "Average PSNR/SSIM(RGB) - %s - x%d - k%d -- PSNR: %.2f dB; SSIM: %.4f",
                    result_name, sf, kernel_idx, ave_psnr, ave_ssim
                )
                if len(metrics["psnr_y"]) > 0:
                    ave_psnr_y = sum(metrics["psnr_y"]) / len(metrics["psnr_y"])
                    ave_ssim_y = sum(metrics["ssim_y"]) / len(metrics["ssim_y"])
                    logger.info(
                        "Average PSNR/SSIM(Y)   - %s - x%d - k%d -- PSNR: %.2f dB; SSIM: %.4f",
                        result_name, sf, kernel_idx, ave_psnr_y, ave_ssim_y
                    )


if __name__ == "__main__":
    main()
