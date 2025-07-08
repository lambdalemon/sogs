import json
import os
from typing import Any, Callable, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torchpq.clustering import KMeans
from torch import Tensor
from plas import sort_with_plas

from plyfile import PlyData, PlyElement
from pathlib import Path
from PIL import Image
import OpenEXR

"""Uses quantization and sorting to compress splats into PNG files and uses
K-means clustering to compress the spherical harmonic coefficents.

.. warning::
    This class requires the `Pillow <https://pypi.org/project/pillow/>`_,
    `plas <https://github.com/fraunhoferhhi/PLAS.git>`_
    and `torchpq <https://github.com/DeMoriarty/TorchPQ?tab=readme-ov-file#install>`_ packages to be installed.

.. warning::
    This class might throw away a few lowest opacities splats if the number of
    splats is not a square number.

.. note::
    The splats parameters are expected to be pre-activation values. It expects
    the following fields in the splats dictionary: "means", "scales", "quats",
    "opacities", "sh0", "shN".

References:
    - `Compact 3D Scene Representation via Self-Organizing Gaussian Grids <https://arxiv.org/abs/2312.13299>`_
    - `Making Gaussian Splats more smaller <https://aras-p.info/blog/2023/09/27/Making-Gaussian-Splats-more-smaller/>`_
"""


def sigmoid(x):
    return 1 / (1 + torch.exp(-x))


def sh0_to_color(x):
    return 0.2820948 * x + 0.5


def unity_vec3_serialize(x):
    return {k:float(v) for k,v in zip("xyz", x)}


def preprocess_splats(splats, include_sh):
    means = splats["means"]

    scales = splats["scales"] * np.log2(np.e)
    scales_min = torch.amin(scales, dim=0)
    scales_max = torch.amax(scales, dim=0)
    scales = (scales - scales_min) / (scales_max - scales_min)

    quats = F.normalize(splats["quats"], dim=-1)
    quats *= torch.where(quats[:,0] < 0, -1, 1)[:, None]
    assert torch.count_nonzero(quats[:,0] < 0) == 0
    quats = quats[:,1:]
    quats = (quats + 1) / 2

    n_gs = splats["sh0"].shape[0]
    sh0 = sh0_to_color(splats["sh0"]).clamp(0, 1).reshape(n_gs, -1)
    opacities = sigmoid(splats["opacities"]).reshape(n_gs, -1)
    colors = torch.cat((sh0, opacities), dim=1)

    if include_sh:
        shN = splats["shN"].clamp(-6.0, 6.0).reshape(n_gs, -1)
        kmeans = KMeans(n_clusters=2**14, distance="manhattan", verbose=True)
        labels = kmeans.fit(shN.permute(1, 0).contiguous())
        centroids = kmeans.centroids.permute(1, 0)
        shN_min = torch.min(centroids)
        shN_max = torch.max(centroids)
        centroids = (centroids - shN_min) / (shN_max - shN_min)
        centroids = centroids.reshape(centroids.shape[0], -1, 3)
    else:
        labels = torch.zeros(n_gs, 1).cuda()
        centroids = torch.zeros(2**14, 15, 3).cuda()
        shN_min = 0
        shN_max = 0

    splats = {
        "means": means,
        "scales": scales,
        "quats": quats,
        "colors": colors,
        "shN": labels,
    }

    meta = {
        "scalesMin": unity_vec3_serialize(scales_min),
        "scalesMax": unity_vec3_serialize(scales_max),
        "shNMin" : float(shN_min),
        "shNMax" : float(shN_max),
    }
    return splats, centroids, meta


def postprocess_splats(splats, centroids):
    splats["shN_centroids"] = centroids
    for k in splats:
        splats[k] = splats[k].detach().cpu().numpy()

    for k in ("scales", "quats", "colors", "shN_centroids"):
        splats[k] = (splats[k] * (2**8 - 1)).round().astype(np.uint8)

    shN_labels = splats.pop("shN")
    shN_labels = np.frombuffer(np.ascontiguousarray(shN_labels, dtype=np.uint16), dtype=np.float16)
    means = splats["means"].astype(np.float16)
    splats["means"] = np.hstack((means, shN_labels[:, np.newaxis]))

    n_gs = len(splats["means"])
    n_sidelen = int(n_gs**0.5)
    for k in splats.keys() - ["shN_centroids"]:
        splats[k] = splats[k].reshape((n_sidelen, n_sidelen, -1))
    splats["shN_centroids"] = splats["shN_centroids"][::-1]

    return splats


def run_compression(compress_dir: str, splats: Dict[str, Tensor], include_sh: bool) -> None:
    """Run compression

    Args:
        compress_dir (str): directory to save compressed files
        splats (Dict[str, Tensor]): Gaussian splats to compress
    """

    n_gs = len(splats["means"])
    n_sidelen = int(n_gs**0.5)
    n_crop = n_gs - n_sidelen**2
    if n_crop != 0:
        splats = _crop_n_splats(splats, n_crop)
        print(
            f"Warning: Number of Gaussians was not square. Removed {n_crop} Gaussians."
        )

    splats, centroids, meta = preprocess_splats(splats, include_sh)

    splats = sort_splats(splats)

    splats = postprocess_splats(splats, centroids)

    for k,v in splats.items():
        if v.dtype == np.float16:
            write_exr(compress_dir, k, v)
        elif v.dtype == np.uint8:
            write_png(compress_dir, k, v)
        else:
            raise TypeError

    with open(os.path.join(compress_dir, "vrcsplat.json"), "w") as f:
        json.dump(meta, f, indent=2)


def write_png(compress_dir, param_name, img):
    filename = f"{param_name}.png"
    Image.fromarray(img).save(os.path.join(compress_dir, filename), format="png", optimize=True)
    print(f"✓ {filename}")


def write_exr(compress_dir, param_name, img):
    filename = f"{param_name}.exr"
    img = np.ascontiguousarray(img)
    channels = { "RGBA" : img }
    header = { "compression" : OpenEXR.ZIP_COMPRESSION,
               "type" : OpenEXR.scanlineimage }
    with OpenEXR.File(header, channels) as outfile:
        outfile.write(os.path.join(compress_dir, filename))
    print(f"✓ {filename}")


def _crop_n_splats(splats: Dict[str, Tensor], n_crop: int) -> Dict[str, Tensor]:
    opacities = splats["opacities"]
    keep_indices = torch.argsort(opacities, descending=True)[:-n_crop]
    for k, v in splats.items():
        splats[k] = v[keep_indices]
    return splats


def sort_splats(splats: Dict[str, Tensor], verbose: bool = True) -> Dict[str, Tensor]:
    """Sort splats with Parallel Linear Assignment Sorting from the paper.

    Args:
        splats (Dict[str, Tensor]): splats
        verbose (bool, optional): Whether to print verbose information. Default to True.

    Returns:
        Dict[str, Tensor]: sorted splats
    """
    n_gs = len(splats["means"])
    n_sidelen = int(n_gs**0.5)
    assert n_sidelen**2 == n_gs, "Must be a perfect square"

    sort_keys = [k for k in splats if k != "shN"]
    params_to_sort = torch.cat([splats[k].reshape(n_gs, -1) for k in sort_keys], dim=-1)
    shuffled_indices = torch.randperm(
        params_to_sort.shape[0], device=params_to_sort.device
    )
    params_to_sort = params_to_sort[shuffled_indices]
    grid = params_to_sort.reshape((n_sidelen, n_sidelen, -1))
    _, sorted_indices = sort_with_plas(
        grid.permute(2, 0, 1), improvement_break=1e-4, verbose=verbose
    )
    sorted_indices = sorted_indices.squeeze().flatten()
    sorted_indices = shuffled_indices[sorted_indices]
    for k, v in splats.items():
        splats[k] = v[sorted_indices]
    return splats

@torch.no_grad()
def read_ply(path):
    """
    Reads a .ply file and reconstructs a dictionary of PyTorch tensors on GPU.
    """
    plydata = PlyData.read(path)
    vd = plydata['vertex'].data

    def has_col(col_name):
        return col_name in vd.dtype.names

    xyz = np.stack([vd['x'], vd['y'], vd['z']], axis=-1)
    f_dc = np.stack([vd[f"f_dc_{i}"] for i in range(3)], axis=-1)

    rest_cols = [c for c in vd.dtype.names if c.startswith('f_rest_')]
    rest_cols_sorted = sorted(rest_cols, key=lambda c: int(c.split('_')[-1]))
    if len(rest_cols_sorted) > 0:
        f_rest = np.stack([vd[c] for c in rest_cols_sorted], axis=-1)
    else:
        f_rest = np.empty((len(vd), 0), dtype=np.float32)

    opacities = vd['opacity']
    scale = np.stack([vd[f"scale_{i}"] for i in range(3)], axis=-1)
    rotation = np.stack([vd[f"rot_{i}"] for i in range(4)], axis=-1)

    splats = {}
    splats["means"] = torch.from_numpy(xyz).float().cuda()
    splats["opacities"] = torch.from_numpy(opacities).float().cuda()
    splats["scales"] = torch.from_numpy(scale).float().cuda()
    splats["quats"] = torch.from_numpy(rotation).float().cuda()

    sh0_tensor = torch.from_numpy(f_dc).float()
    sh0_tensor = sh0_tensor.unsqueeze(-1).transpose(1, 2)
    splats["sh0"] = sh0_tensor.cuda()

    if f_rest.any():
        if f_rest.shape[1] % 3 != 0:
            raise ValueError(f"Number of f_rest columns ({f_rest.shape[1]}) not divisible by 3.")
        num_rest_per_channel = f_rest.shape[1] // 3
        shn_tensor = torch.from_numpy(
            f_rest.reshape(-1, 3, num_rest_per_channel)
        ).float().transpose(1, 2)
        splats["shN"] = shn_tensor.cuda()

    return splats
