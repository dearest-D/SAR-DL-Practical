"""
Part 1 - SAR image despeckling with DiffusionSAR vs. a classical boxcar filter.

Run:   python DiffusionSAR.py
       python DiffusionSAR.py --boxcar 7 --grid_r 32

Pipeline (DiffusionSAR):
  0. Amplitude -> intensity (A^2); the model was trained on single-look intensity speckle
  1. 2% linear stretch of the intensity image to [0, 1]           (ENVI-style)
  2. Log + Yeo-Johnson transform of the stretched image           (Ma et al., 2024)
  3. Conditional diffusion sampling (DDIM) on overlapping 128x128 patches
  4. Inverse Yeo-Johnson + inverse log  -> despeckled stretched image
  5. Map the result back to the original intensity range (residual back-projection)
  6. Replace pixels that were clipped by the 2% stretch with their 3x3 local mean
  7. Intensity -> amplitude (square root)

Reference:
  Ma, Y., Ke, P., Aghababaei, H., Chang, L., & Wei, J. (2024). Despeckling SAR Images
  With Log-Yeo-Johnson Transformation and Conditional Diffusion Models.
  IEEE TGRS, 62, 5215417. https://doi.org/10.1109/TGRS.2024.3419083
"""
import argparse
import os
import time
from types import SimpleNamespace

import numpy as np
import tifffile
import torch
import matplotlib.pyplot as plt
from scipy import stats
from scipy.ndimage import uniform_filter
from scipy.special import inv_boxcox

from diffusionsar_unet import DiffusionUNet

HERE = os.path.dirname(os.path.abspath(__file__))
NP_MAX = 10000.0


# --------------------------------------------------------------------------------------
# 1. 2% linear stretch (Python port of Enhance_Percent2.m)
# --------------------------------------------------------------------------------------
def percent_stretch(image, threshold=0.02, n_bins=4080):
    """Clip the lowest/highest `threshold` fraction of pixels and scale to [0, 1]."""
    v = image.astype(np.float64).ravel()
    vmin, vmax = v.min(), v.max() + np.finfo(float).eps
    scale = n_bins / (vmax - vmin)
    q = np.floor((v - vmin) * scale + 0.5).astype(np.int64) + 1          # 1 .. n_bins+1
    hist = np.bincount(q, minlength=n_bins + 2)
    n_clip = round(threshold * v.size)

    cdf = np.cumsum(hist)                                                 # count(q <= val)
    val = int(np.argmax(cdf > n_clip))
    if val > 1 and cdf[val] - n_clip > n_clip - cdf[val - 1]:
        val -= 1
    lower = (val - 1) / scale + vmin

    ccdf = np.cumsum(hist[::-1])[::-1]                                     # count(q >= val)
    val = int(np.nonzero(ccdf > n_clip)[0].max())
    if val < n_bins + 1 and ccdf[val] - n_clip > n_clip - ccdf[val + 1]:
        val += 1
    upper = (val - 1) / scale + vmin

    stretched = np.clip((image - lower) / (upper - lower), 0, 1)
    return stretched.astype(np.float32), (lower, upper)


# --------------------------------------------------------------------------------------
# 2. Log + Yeo-Johnson transform and its inverse
# --------------------------------------------------------------------------------------
def estimate_lambda(shape, looks=1):
    """Yeo-Johnson lambda that makes log-Gamma speckle (L looks) close to Gaussian."""
    speckle = np.random.gamma(looks, 1.0 / looks, shape)
    _, lam = stats.yeojohnson(np.log(speckle).ravel())
    return lam


def log_yeojohnson(x, lam):
    img = x * NP_MAX
    img[img == 0] = 1
    max_num = img.max()
    img = stats.yeojohnson(np.log(img) / np.log(max_num), lam)
    img = img / stats.yeojohnson(1, lam)
    return img.astype(np.float32), max_num


def inverse_log_yeojohnson(y, lam, max_num):
    y = np.clip(y, 0, 1) * stats.yeojohnson(1, lam)
    y = np.clip(inv_boxcox(y, lam) - 1, 0, 1)
    return np.exp(y * np.log(max_num)) / NP_MAX


# --------------------------------------------------------------------------------------
# 3. DiffusionSAR model + patch-wise DDIM sampling
# --------------------------------------------------------------------------------------
def to_namespace(d):
    return SimpleNamespace(**{k: to_namespace(v) if isinstance(v, dict) else v for k, v in d.items()})


def load_model(path, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    config = to_namespace(ckpt["config"])
    model = DiffusionUNet(config)
    # weights are stored in float16 to halve the file size; computation is float32
    model.load_state_dict({k: v.float() for k, v in ckpt["state_dict"].items()}, strict=True)
    return model.to(device).eval(), config


def grid_positions(size, patch, stride):
    pos = list(range(0, size - patch + 1, stride))
    if pos[-1] != size - patch:
        pos.append(size - patch)
    return pos


@torch.no_grad()
def diffusion_sample(model, config, x_cond, x_init, steps=25, stride=16, batch=16):
    """DDIM (eta=0) sampling; noise predictions of overlapping patches are averaged."""
    device = x_cond.device
    p = config.data.patch_size
    betas = torch.linspace(config.diffusion.beta_start, config.diffusion.beta_end, 1000,
                           dtype=torch.float64).float().to(device)
    alphas = torch.cat([torch.ones(1, device=device), (1 - betas).cumprod(0)])   # alpha_bar(t+1)
    T = config.diffusion.num_diffusion_timesteps
    seq = list(range(0, T, T // steps))
    seq_next = [-1] + seq[:-1]

    H, W = x_cond.shape[-2:]
    corners = [(i, j) for i in grid_positions(H, p, stride) for j in grid_positions(W, p, stride)]
    count = torch.zeros_like(x_init)
    for i, j in corners:
        count[..., i:i + p, j:j + p] += 1
    cond_patches = torch.cat([x_cond[..., i:i + p, j:j + p] for i, j in corners])

    print(f"  {len(corners)} patches x {len(seq)} steps on {device}")
    xt = x_init
    for k, (t, t_next) in enumerate(zip(reversed(seq), reversed(seq_next))):
        at, at_next = alphas[t + 1], alphas[t_next + 1]
        xt_patches = torch.cat([xt[..., i:i + p, j:j + p] for i, j in corners])
        eps = torch.zeros_like(xt)
        for b in range(0, len(corners), batch):
            inp = torch.cat([cond_patches[b:b + batch], xt_patches[b:b + batch]], dim=1)
            tt = torch.full((inp.shape[0],), float(t), device=device)
            out = model(inp, tt)[:, :1]                                  # 1st channel = noise
            for n, (i, j) in enumerate(corners[b:b + batch]):
                eps[..., i:i + p, j:j + p] += out[n]
        eps = eps / count
        x0 = (xt - eps * (1 - at).sqrt()) / at.sqrt()
        xt = at_next.sqrt() * x0 + (1 - at_next).sqrt() * eps
        print(f"\r  step {k + 1}/{len(seq)}", end="", flush=True)
    print()
    return xt


def despeckle_diffusionsar(amplitude, model, config, device, steps, stride):
    """DiffusionSAR was trained on single-look INTENSITY speckle (Gamma, L=1), so the
    amplitude image is squared before despeckling and the result is square-rooted."""
    intensity = amplitude.astype(np.float32) ** 2
    stretched, (lower, upper) = percent_stretch(intensity, 0.02)
    lam = estimate_lambda(stretched.shape)
    x0, max_num = log_yeojohnson(stretched, lam)

    to_t = lambda a: torch.from_numpy(a)[None, None].to(device)
    y = diffusion_sample(model, config, to_t(stretched), to_t(x0), steps=steps, stride=stride)
    pred = inverse_log_yeojohnson(y.squeeze().cpu().numpy(), lam, max_num)

    # back to the original intensity range (same as residual_noise.m)
    out = intensity + (pred - stretched) * (upper - lower) + lower
    out = remove_clipped_outliers(out, stretched)
    return np.sqrt(out).astype(np.float32)                              # intensity -> amplitude


def remove_clipped_outliers(image, stretched, tol=1e-6):
    """Pixels clipped to 0 or 1 by the 2% stretch are replaced by their 3x3 local mean
    (Python port of removeIsolatedOutliersLog.m)."""
    image = image.astype(np.float32)
    mask = (stretched <= tol) | (stretched >= 1 - tol)
    local_mean = uniform_filter(image, size=3, mode="nearest")
    cleaned = np.where(mask, local_mean, image)
    return np.maximum(cleaned, 0).astype(np.float32)


# --------------------------------------------------------------------------------------
# 4. Classical boxcar filter
# --------------------------------------------------------------------------------------
def boxcar(amplitude, window=5):
    """Average the intensity (|A|^2) in a window x window box, return amplitude."""
    return np.sqrt(uniform_filter(amplitude.astype(np.float64) ** 2, size=window)).astype(np.float32)


# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="SAR despeckling: DiffusionSAR vs. boxcar")
    ap.add_argument("--image", default=os.path.join(HERE, "NISAR_HH_amplitude.tif"))
    ap.add_argument("--model", default=os.path.join(HERE, "model", "DiffusionSAR_pretrained.pth"))
    ap.add_argument("--boxcar", type=int, default=5, help="boxcar window size")
    ap.add_argument("--steps", type=int, default=25, help="DDIM sampling steps")
    ap.add_argument("--grid_r", type=int, default=64,
                    help="stride between overlapping 128x128 patches (16 = finest, ~16x slower; 64 = fast)")
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--seed", type=int, default=61)
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stride = args.grid_r

    amp = tifffile.imread(args.image).astype(np.float32)
    print(f"Image: {args.image}  shape={amp.shape}")

    t0 = time.time()
    box = boxcar(amp, args.boxcar)
    print(f"Boxcar {args.boxcar}x{args.boxcar}: {time.time() - t0:.2f} s")

    print("DiffusionSAR:")
    t0 = time.time()
    model, config = load_model(args.model, device)
    dif = despeckle_diffusionsar(amp, model, config, device, args.steps, stride)
    print(f"DiffusionSAR: {time.time() - t0:.1f} s")

    os.makedirs(args.out, exist_ok=True)
    tifffile.imwrite(os.path.join(args.out, "boxcar.tif"), box)
    tifffile.imwrite(os.path.join(args.out, "DiffusionSAR.tif"), dif.astype(np.float32))

    titles = ["Original (noisy) NISAR HH amplitude",
              f"Boxcar filter ({args.boxcar}x{args.boxcar})",
              "DiffusionSAR"]
    print("\nMean intensity (radiometric preservation):")
    for img, title in zip([amp, box, dif], titles):
        print(f"  {title:40s} {np.mean(img.astype(np.float64) ** 2):.4f}")

    # Display: each image with its own 2%-98% grey-level stretch
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.8))
    for ax, img, title in zip(axes, [amp, box, dif], titles):
        vmin, vmax = np.percentile(img, [2, 98])
        ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(os.path.join(args.out, "comparison.png"), dpi=150)
    print(f"Results saved to {args.out}")
    plt.show()


if __name__ == "__main__":
    main()
