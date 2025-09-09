"""utils.visualization
Định tính: visualize_comparison, make_grid_triplet.
"""

from typing import Optional
import numpy as np
import torch

try:
    import torchvision.utils as vutils
except Exception:  # pragma: no cover
    vutils = None

__all__ = ["visualize_comparison", "make_grid_triplet"]


def visualize_comparison(
    blur,
    restored,
    sharp,
    titles=("Input (Blur)", "Output (Model)", "Ground Truth"),
    figsize=(9, 3),
    save_path: Optional[str] = None,
    cmap=None,
    show=True,
):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "Cần matplotlib để visualize. Cài: pip install matplotlib"
        ) from e

    def to_np(img):
        if isinstance(img, torch.Tensor):
            x = img.detach().cpu()
            if x.ndim == 4:
                x = x[0]
            if x.ndim == 3 and x.shape[0] in (1, 3):
                x = x if x.shape[0] != 1 else x.repeat(3, 1, 1)
                x = x.permute(1, 2, 0)
            x = x.clip(0, 1).numpy()
            return x
        return np.clip(img, 0, 1)

    imgs = [to_np(blur), to_np(restored), to_np(sharp)]
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    for ax, im, tt in zip(axes, imgs, titles):
        if im.ndim == 2:
            ax.imshow(im, cmap=cmap or "gray")
        else:
            ax.imshow(im)
        ax.set_title(tt)
        ax.axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def make_grid_triplet(blur, restored, sharp, pad_value=0.5):
    if vutils is None:
        return None

    def norm_tensor(x):
        if x.ndim == 4:
            x = x[0]
        if x.ndim == 2:
            x = x.unsqueeze(0)
        return x.clamp(0, 1)

    tensors = [norm_tensor(t) for t in (blur, restored, sharp)]
    cat = torch.stack(tensors, 0)
    grid = vutils.make_grid(cat, nrow=3, padding=2, pad_value=pad_value)
    return grid
