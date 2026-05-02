import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
import dnnlib
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image import StructuralSimilarityIndexMeasure
import torch.nn.functional as F
from masks import build_mask


def plot_inpainting_results(
    col_imgs,
    cond_imgs,
    save_path="./CIFAR_inpainting.png",
    conditioning_mask=None,
    num_images_to_plot=10,
):
    """
    Plots the conditioning images (original and masked/sub-image) and generated images in a grid.

    Args:
        col_imgs (list of torch.Tensor): List of batches of generated images, each of shape [batch, C, H, W].
        cond_imgs (torch.Tensor): Conditioning images, shape [batch, C, H, W].
        save_path (str): Path to save the resulting figure.
        conditioning_mask (torch.Tensor or None): Optional mask of shape [batch, C, H, W] or [batch, 1, H, W] or [batch, H, W].
            Should contain 1s (show) and 0s (mask). If None, defaults to left-half masking.
    """

    plt.figure(figsize=(12, 10), facecolor="black")  # Black background
    plt.subplots_adjust(wspace=0.05, hspace=0.05)  # Remove vertical spacing completely

    col_imgs = col_imgs[0:num_images_to_plot]
    cond_imgs = cond_imgs[0:num_images_to_plot]

    num_rows = cond_imgs.shape[0]
    num_cols = 2 + len(col_imgs)

    for row in range(num_rows):
        # Original conditioning image
        ax = plt.subplot(num_rows, num_cols, row * num_cols + 1)
        ax.set_facecolor("black")
        orig_cond_img = cond_imgs[row].permute(1, 2, 0).cpu().numpy()
        orig_cond_img = (orig_cond_img + 1) / 2
        orig_cond_img = np.clip(orig_cond_img, 0, 1)
        plt.imshow(orig_cond_img)
        plt.axis("off")

        # Masked/sub-image using conditioning_mask if provided, else left-half mask
        ax = plt.subplot(num_rows, num_cols, row * num_cols + 2)
        ax.set_facecolor("black")
        cond_img = cond_imgs[row].permute(1, 2, 0).cpu().numpy()
        cond_img = (cond_img + 1) / 2
        cond_img = np.clip(cond_img, 0, 1)
        if conditioning_mask is not None:
            # Get mask for this row, broadcast to 3 channels if needed
            mask = conditioning_mask[row]
            if mask.ndim == 2:
                mask = mask[None, :, :]  # [1, H, W]
            if mask.shape[0] == 1 and cond_img.shape[2] == 3:
                mask = np.repeat(mask.cpu().numpy(), 3, axis=0)
            else:
                mask = mask.cpu().numpy()
            # mask shape: [C, H, W], cond_img: [H, W, C]
            mask = np.transpose(mask, (1, 2, 0))  # [H, W, C]
            # If mask is not float, cast to float
            mask = mask.astype(np.float32)
            # Show masked region as gray (0.25), unmasked as image
            masked_img = cond_img * mask + 0.25 * (1 - mask)
            plt.imshow(masked_img)
        else:
            # Default: left-half mask
            width = cond_img.shape[1]
            cond_img_masked = cond_img.copy()
            cond_img_masked[:, width // 2 :, :] = 0.25
            plt.imshow(cond_img_masked)
        plt.axis("off")

        # Generated images
        for col in range(len(col_imgs)):
            ax = plt.subplot(num_rows, num_cols, row * num_cols + col + 3)
            ax.set_facecolor("black")
            gen_img = col_imgs[col][row].permute(1, 2, 0).cpu().numpy()
            gen_img = (gen_img + 1) / 2
            gen_img = np.clip(gen_img, 0, 1)
            plt.imshow(gen_img)
            plt.axis("off")

    plt.tight_layout(pad=0.1)
    plt.subplots_adjust(hspace=0.1, wspace=0.2)
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def plot_intermediates(
    intermediates,
    save_path="./CIFAR_inpainting_intermediates.png",
):
    intermediates = intermediates[:, 0, :, :, :]
    # plot intermediates in a grid
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    # intermediates: [B, 3, 32, 32]
    B = intermediates.shape[0]
    num_cols = 8
    num_rows = int(np.ceil(B / num_cols))

    plt.figure(figsize=(num_cols * 1.5, num_rows * 1.5))
    for idx in range(B):
        ax = plt.subplot(num_rows, num_cols, idx + 1)
        img = intermediates[idx].permute(1, 2, 0).cpu().numpy()  # [32, 32, 3]
        img = (img + 1) / 2
        img = np.clip(img, 0, 1)
        ax.imshow(img)
        ax.set_axis_off()
        ax.set_facecolor("black")
    plt.tight_layout(pad=0.1)
    plt.subplots_adjust(hspace=0.1, wspace=0.1)
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def save_col_imgs_to_npz(
    col_imgs, output_dir="npz_output", filename="inpaint_results.npz"
):
    """
    Saves a list of PyTorch tensors (batches of images) as separate arrays in a .npz file.

    Args:
        col_imgs (list of torch.Tensor): List of batches of images, each of shape [batch_size, channels, height, width].
        output_dir (str): Directory to save the .npz file.
        filename (str): Name of the .npz file.
    """
    import numpy as np
    import os

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Convert the list of PyTorch tensors to numpy arrays
    col_imgs_np = []
    for col in col_imgs:
        # Convert each batch of images from PyTorch tensor to numpy array
        col_np = col.cpu().numpy()
        col_imgs_np.append(col_np)

    # Save as npz file with each column as a separate array
    np.savez(os.path.join(output_dir, filename), *col_imgs_np)

    print(
        f"Saved {len(col_imgs)} columns of images to {os.path.join(output_dir, filename)}"
    )


# network_pkl = 'training-runs/00020-cifar10-32x32-uncond-ddpmpp-edm-gpus8-batch512-fp32-noise_uncond=True-beta1=0.9-beta2=0.95-lr=0.001-disco=True-t0=0.01/network-snapshot-200000.pkl'
# network_pkl = 'training-runs/00039-cifar10-32x32-uncond-ddpmpp-edm-gpus7-batch224-fp32-noise_uncond=True-beta1=0.9-beta2=0.95-lr=0.001-disco=True-t0=0.01-t_independent_cs=True-gamma=1.0/network-snapshot-102861.pkl'


def main():
    from generate import generate_images

    # set random seed
    torch.manual_seed(31410)
    np.random.seed(31410)

    parser = argparse.ArgumentParser(description="Inpainting driver")
    parser.add_argument(
        "--network_pkl",
        type=str,
        default="training-runs/00040-cifar10-32x32-uncond-ddpmpp-edm-gpus7-batch224-fp32-noise_uncond=True-beta1=0.9-beta2=0.95-lr=0.001-disco=True-t0=0.01-t_independent_cs=True-gamma=0.5/network-snapshot-200000.pkl",
        help="Path to network pickle",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="out_inpaint",
        help="Output directory",
    )
    parser.add_argument(
        "--replacement_heuristic",
        action="store_true",
        help="Use replacement heuristic during sampling",
    )
    parser.add_argument(
        "--no_plot",
        action="store_true",
        help="Do not plot the results",
    )
    parser.add_argument(
        "--save_images",
        action="store_true",
        help="Save the results as pngs",
    )
    parser.add_argument(
        "--langevin",
        action="store_true",
        help="Use Langevin dynamics sampler",
    )
    parser.add_argument(
        "--plot_intermediates",
        action="store_true",
        help="Plot the intermediate results",
    )
    parser.add_argument(
        "--mask",
        type=str,
        choices=["wide", "narrow", "sr2x", "alt_lines", "expand", "half", "none"],
        default="top",
        help="Conditioning mask orientation",
    )
    parser.add_argument(
        "--pixel_obs_prob",
        type=float,
        default=0.5,
        help="Probability of a pixel being observed",
    )
    parser.add_argument(
        "--max_batch_size", type=int, default=64, help="Maximum batch size"
    )
    parser.add_argument("--steps", type=int, default=1024, help="Sampling steps")
    parser.add_argument("--S_churn", type=float, default=80, help="S_churn")
    parser.add_argument("--S_min", type=float, default=0.01, help="S_min")
    parser.add_argument("--S_max", type=float, default=80, help="S_max")
    parser.add_argument("--sigma_min", type=float, default=0.002, help="sigma_min")
    parser.add_argument("--sigma_max", type=float, default=80, help="sigma_max")
    parser.add_argument("--S_noise", type=float, default=1.007, help="S_noise")
    parser.add_argument("--rho", type=float, default=7, help="rho")
    parser.add_argument("--num_images", type=int, default=10, help="Number of images")
    parser.add_argument(
        "--solver",
        help="Ablate ODE solver",
        metavar="euler|heun",
        choices=["euler", "heun"],
    )
    parser.add_argument(
        "--disc",
        dest="discretization",
        help="Ablate time step discretization {t_i}",
        metavar="vp|ve|iddpm|edm",
        choices=["vp", "ve", "iddpm", "edm"],
    )
    parser.add_argument(
        "--schedule",
        help="Ablate noise schedule sigma(t)",
        metavar="vp|ve|linear",
        choices=["vp", "ve", "linear"],
    )
    parser.add_argument(
        "--scaling",
        help="Ablate signal scaling s(t)",
        metavar="vp|none",
        choices=["vp", "none"],
    )

    parser.add_argument(
        "--ancestral_sampling",
        action="store_true",
        help="Use ancestral sampling",
    )
    parser.add_argument(
        "--repaint_enable",
        action="store_true",
        help="Use repaint",
    )
    parser.add_argument(
        "--repaint_jump",
        type=int,
        default=10,
        help="Repaint jump",
    )
    parser.add_argument(
        "--repaint_repeats",
        type=int,
        default=1,
        help="Repaint repeats",
    )
    parser.add_argument(
        "--gradient_guidance",
        action="store_true",
        help="Use gradient guidance",
    )
    parser.add_argument(
        "--tds",
        action="store_true",
        help="Use TDS",
    )
    parser.add_argument(
        "--tds_n_steps_each",
        type=int,
        default=1,
        help="TDS n steps each",
    )
    parser.add_argument(
        "--plot",
        type=str,
        default=None,
        help="Path to save plot PNG (e.g., my_dir/out.png). If set, a plot is generated and saved.",
    )
    args = parser.parse_args()

    if args.plot_intermediates:
        assert args.num_images == 1, "num_images must be 1 for plotting intermediates"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Derive mask size from network
    with dnnlib.util.open_url(args.network_pkl, verbose=True) as f:
        net = pickle.load(f)["ema"].to(device)
        height = width = int(net.img_resolution)
        channels = int(net.img_channels)
        del net

    conditioning_mask = build_mask(
        args.num_images,
        args.mask,
        channels,
        height,
        width,
        device=device,
    )

    if args.langevin:
        langevin_kwargs = {
            "step_size_init": 1.0,
            "step_size_final": 1.0,
            "temperature_init": 0.1,
            "temperature_final": 0.1,
        }
    else:
        langevin_kwargs = {}

    n_cols = 1
    col_imgs = []
    cond_imgs = None
    for col_idx in range(0, n_cols):
        seeds = list(range(col_idx * args.num_images, (col_idx + 1) * args.num_images))
        # print(seeds)
        images, xs, cond_imgs, denoised_xs = generate_images(
            network_pkl=args.network_pkl,
            outdir=args.outdir,
            inpaint=True,
            num_steps=args.steps,
            S_churn=args.S_churn,
            S_min=args.S_min,
            S_max=args.S_max,
            S_noise=args.S_noise,
            rho=args.rho,
            seeds=seeds,
            return_intermediate=True,
            replacement_heuristic=args.replacement_heuristic,
            init_dist=False,
            conditioning_mask=conditioning_mask,
            save_images=args.save_images,
            max_batch_size=args.max_batch_size,
            sigma_min=args.sigma_min,
            sigma_max=args.sigma_max,
            ancestral_sampling=args.ancestral_sampling,
            repaint_enable=args.repaint_enable,
            repaint_jump=args.repaint_jump,
            repaint_repeats=args.repaint_repeats,
            solver=args.solver,
            discretization=args.discretization,
            schedule=args.schedule,
            scaling=args.scaling,
            gradient_guidance=args.gradient_guidance,
            tds=args.tds,
            tds_n_steps_each=args.tds_n_steps_each,
            **langevin_kwargs,
        )
        # print(f"images.shape: {images.shape}")

        # in the generated images, set the pixels in the conditioning mask to the true conditioning image
        images = images * (1 - conditioning_mask) + cond_imgs * conditioning_mask

        col_imgs.append(images)
        # col_imgs.append(xs[0])

    save_col_imgs_to_npz(
        col_imgs, output_dir="npz_output", filename="inpaint_results.npz"
    )

    # Determine plot path
    plot_path = (
        args.plot if args.plot else os.path.join(args.outdir, "CIFAR_inpainting.png")
    )
    plot_dir = os.path.dirname(plot_path)
    if plot_dir:
        os.makedirs(plot_dir, exist_ok=True)

    if not args.no_plot:
        plot_inpainting_results(
            col_imgs,
            cond_imgs,
            save_path=plot_path,
            conditioning_mask=conditioning_mask,
        )
        if args.plot_intermediates:
            # Save intermediates with "_inpainting" attached to the plot filename
            base, ext = os.path.splitext(plot_path)
            intermediates_path = f"{base}_denoised_intermediates{ext}"
            plot_intermediates(denoised_xs, save_path=intermediates_path)

            intermediates_path = f"{base}_intermediates{ext}"
            plot_intermediates(xs, save_path=intermediates_path)

        print("Saved plot to", plot_path)

    generated_imgs = torch.stack(col_imgs)[0].clamp(-1, 1).float()

    lpips, ssim = get_lpips_ssim(generated_imgs, cond_imgs)

    print(f"Model: {args.network_pkl}")
    print(f"Mask: {args.mask}")
    print(f"Replacement Heuristic: {args.replacement_heuristic}")
    print(f"Steps: {args.steps}")
    print(f"S_churn: {args.S_churn}")
    print(f"S_min: {args.S_min}")
    print(f"S_max: {args.S_max}")
    print(f"S_noise: {args.S_noise}")
    print(f"rho: {args.rho}")

    print("-" * 20)
    print(f"LPIPS: {lpips}")
    print(f"SSIM: {ssim}")


def get_lpips_ssim(generated_imgs, cond_imgs):
    # upsample to 224x224
    generated_imgs_up = F.interpolate(
        generated_imgs, size=224, mode="bilinear", align_corners=False
    )
    cond_imgs_up = F.interpolate(
        cond_imgs, size=224, mode="bilinear", align_corners=False
    )

    assert generated_imgs_up.shape[1] == 3, "generated_imgs_up should have 3 channels"
    assert cond_imgs_up.shape[1] == 3, "cond_imgs_up should have 3 channels"
    assert (
        generated_imgs_up.min() < -0.75 and generated_imgs_up.max() > 0.75
    ), f"generated_imgs_up should be between -1 and 1, but got {generated_imgs_up.min()}, {generated_imgs_up.max()}"
    assert (
        cond_imgs_up.min() < -0.75 and cond_imgs_up.max() > 0.75
    ), f"cond_imgs_up should be between -1 and 1, but got {cond_imgs_up.min()}, {cond_imgs_up.max()}"

    lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", reduction="mean").to(
        torch.device("cuda")
    )(generated_imgs_up, cond_imgs_up)

    ssim = StructuralSimilarityIndexMeasure(data_range=(-1, 1)).to(
        torch.device("cuda")
    )(generated_imgs_up, cond_imgs_up)

    return lpips, ssim


if __name__ == "__main__":
    main()
