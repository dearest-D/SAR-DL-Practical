# SAR Image Despeckling (course exercise)

Part 1 of the practical exercise *DL Meets SAR*: despeckling a NISAR HH amplitude image with the
pre-trained **DiffusionSAR** model and comparing it with a classical **boxcar filter**.

## Run in Google Colab (nothing to install)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dearest-D/SAR-DL-Practical/blob/main/DiffusionSAR.ipynb)

1. Click the **Open in Colab** button above (sign in with a Google account).
2. *Runtime → Change runtime type →* **T4 GPU** → *Save*.
3. *Runtime → Run all*.

The image, the pre-trained model and the code are downloaded automatically from the
[release](https://github.com/dearest-D/SAR-DL-Practical/releases/tag/v1.0) (~170 MB).
After about 1–2 minutes the original image, the boxcar result and the DiffusionSAR result are
shown side by side. Without a GPU the notebook still works, but takes ~10–15 minutes.

## Run on your own computer

Download and extract [Part1_Denoising.zip](https://github.com/dearest-D/SAR-DL-Practical/releases/download/v1.0/Part1_Denoising.zip), then

```bash
cd Part1_Denoising
pip install -r requirements.txt
python DiffusionSAR.py          # options: --boxcar 7  --grid_r 16  --steps 25
```

The script uses an NVIDIA GPU automatically if available (CUDA version of PyTorch), otherwise the CPU.
Running time: about 30–60 s on a GPU, about 10–20 min on a CPU.

## Files

| File | Content |
|---|---|
| `DiffusionSAR.ipynb` | Colab notebook |
| `DiffusionSAR.py` | boxcar filter + DiffusionSAR despeckling, side-by-side display |
| `diffusionsar_unet.py` | DiffusionSAR network (conditional U-Net with residual and attention blocks) |
| Release `Part1_Denoising.zip` | `DiffusionSAR.py`, `diffusionsar_unet.py`, `requirements.txt`, `NISAR_HH_amplitude.tif` (512 × 512) and the pre-trained weights |

## Reference

Ma, Y., Ke, P., Aghababaei, H., Chang, L., & Wei, J. (2024). Despeckling SAR Images With
Log-Yeo–Johnson Transformation and Conditional Diffusion Models. *IEEE Transactions on Geoscience
and Remote Sensing*, 62, 5215417. https://doi.org/10.1109/TGRS.2024.3419083
