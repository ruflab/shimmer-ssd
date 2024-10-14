from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import matplotlib.path as mpath
import numpy as np
import torch
from matplotlib import patches as patches
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from tqdm import tqdm
from transformers import BertModel, BertTokenizer

from simple_shapes_dataset.cli.graph import build_dependency_graph


@dataclass
class Dataset:
    classes: np.ndarray
    locations: np.ndarray
    sizes: np.ndarray
    rotations: np.ndarray
    colors: np.ndarray
    colors_hls: np.ndarray
    unpaired: np.ndarray


def get_transformed_coordinates(
    coordinates: np.ndarray, origin: np.ndarray, scale: float, rotation: float
) -> np.ndarray:
    center = np.array([[0.5, 0.5]])
    rotation_m = np.array(
        [
            [np.cos(rotation), -np.sin(rotation)],
            [np.sin(rotation), np.cos(rotation)],
        ]
    )
    rotated_coordinates = (coordinates - center) @ rotation_m.T
    return origin + scale * rotated_coordinates


def get_diamond_patch(
    location: np.ndarray,
    scale: int,
    rotation: float,
    color: np.ndarray,
) -> patches.Polygon:
    x, y = location[0], location[1]
    coordinates = np.array([[0.5, 0.0], [1, 0.3], [0.5, 1], [0, 0.3]])
    origin = np.array([[x, y]])
    patch = patches.Polygon(
        get_transformed_coordinates(coordinates, origin, scale, rotation),
        facecolor=color,
    )
    return patch


def get_triangle_patch(
    location: np.ndarray,
    scale: int,
    rotation: float,
    color: np.ndarray,
) -> patches.Polygon:
    x, y = location[0], location[1]
    origin = np.array([[x, y]])
    coordinates = np.array([[0.5, 1.0], [0.2, 0.0], [0.8, 0.0]])
    patch = patches.Polygon(
        get_transformed_coordinates(coordinates, origin, scale, rotation),
        facecolor=color,
    )
    return patch


def get_egg_patch(
    location: np.ndarray, scale: int, rotation: float, color: np.ndarray
) -> patches.PathPatch:
    x, y = location[0], location[1]
    origin = np.array([[x, y]])
    coordinates = np.array(
        [
            [0.5, 0],
            [0.8, 0],
            [0.9, 0.1],
            [0.9, 0.3],
            [0.9, 0.5],
            [0.7, 1],
            [0.5, 1],
            [0.3, 1],
            [0.1, 0.5],
            [0.1, 0.3],
            [0.1, 0.1],
            [0.2, 0],
            [0.5, 0],
        ]
    )
    codes = [
        mpath.Path.MOVETO,
        mpath.Path.CURVE4,
        mpath.Path.CURVE4,
        mpath.Path.CURVE4,
        mpath.Path.CURVE4,
        mpath.Path.CURVE4,
        mpath.Path.CURVE4,
        mpath.Path.CURVE4,
        mpath.Path.CURVE4,
        mpath.Path.CURVE4,
        mpath.Path.CURVE4,
        mpath.Path.CURVE4,
        mpath.Path.CURVE4,
    ]
    path = mpath.Path(
        get_transformed_coordinates(coordinates, origin, scale, rotation),
        codes,
    )
    patch = patches.PathPatch(path, facecolor=color)
    return patch


def generate_image(
    ax: Axes,
    cls: int,
    location: np.ndarray,
    scale: int,
    rotation: float,
    color: np.ndarray,
    imsize: int = 32,
) -> None:
    color = color.astype(np.float32) / 255
    patch: patches.Patch
    if cls == 0:
        patch = get_diamond_patch(location, scale, rotation, color)
    elif cls == 1:
        patch = get_egg_patch(location, scale, rotation, color)
    elif cls == 2:
        patch = get_triangle_patch(location, scale, rotation, color)
    else:
        raise ValueError("Class does not exist.")

    ax.add_patch(patch)
    ax.set_xticks([])
    ax.set_yticks([])  # type: ignore
    ax.grid(False)
    ax.set_xlim(0, imsize)
    ax.set_ylim(0, imsize)


def generate_scale(n_samples: int, min_val: int, max_val: int) -> np.ndarray:
    assert max_val > min_val
    return np.random.randint(min_val, max_val + 1, n_samples)


def generate_color(
    n_samples: int, min_lightness: int = 0, max_lightness: int = 256
) -> tuple[np.ndarray, np.ndarray]:
    import cv2

    assert 0 <= max_lightness <= 256
    hls = np.random.randint(
        [0, min_lightness, 0],
        [181, max_lightness, 256],
        size=(1, n_samples, 3),
        dtype=np.uint8,
    )
    rgb = cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)[0]  # type: ignore
    return rgb.astype(int), hls[0].astype(int)


def generate_rotation(n_samples: int) -> np.ndarray:
    rotations = np.random.rand(n_samples) * 2 * np.pi
    return rotations


def generate_location(n_samples: int, max_scale: int, imsize: int) -> np.ndarray:
    assert max_scale <= imsize
    margin = max_scale // 2
    locations = np.random.randint(margin, imsize - margin, (n_samples, 2))
    return locations


def generate_class(n_samples: int) -> np.ndarray:
    return np.random.randint(3, size=n_samples)


def generate_unpaired_attr(n_samples: int) -> np.ndarray:
    return np.random.rand(n_samples)


def generate_dataset(
    n_samples: int,
    min_scale: int,
    max_scale: int,
    min_lightness: int,
    max_lightness: int,
    imsize: int,
    classes: np.ndarray | None = None,
) -> Dataset:
    if classes is None:
        classes = generate_class(n_samples)
    sizes = generate_scale(n_samples, min_scale, max_scale)
    locations = generate_location(n_samples, max_scale, imsize)
    rotation = generate_rotation(n_samples)
    colors_rgb, colors_hls = generate_color(n_samples, min_lightness, max_lightness)
    unpaired = generate_unpaired_attr(n_samples)
    return Dataset(
        classes=classes,
        locations=locations,
        sizes=sizes,
        rotations=rotation,
        colors=colors_rgb,
        colors_hls=colors_hls,
        unpaired=unpaired,
    )


def save_dataset(path_root: Path, dataset: Dataset, imsize: int) -> None:
    dpi = 1
    enumerator = tqdm(
        enumerate(
            zip(
                dataset.classes,
                dataset.locations,
                dataset.sizes,
                dataset.rotations,
                dataset.colors,
                strict=True,
            )
        ),
        total=len(dataset.classes),
    )
    for k, (cls, location, size, rotation, color) in enumerator:
        path_file = path_root / f"{k}.png"

        fig, ax = plt.subplots(figsize=(imsize / dpi, imsize / dpi), dpi=dpi)
        ax = cast(plt.Axes, ax)
        generate_image(ax, cls, location, size, rotation, color, imsize)
        ax.set_facecolor("black")
        plt.tight_layout(pad=0)
        plt.savefig(path_file, dpi=dpi, format="png")
        plt.close(fig)


def load_labels_old(path_root: Path) -> Dataset:
    labels = np.load(path_root)
    return Dataset(
        classes=labels[:, 0],
        locations=labels[:, 1:3],
        sizes=labels[:, 3],
        rotations=labels[:, 4],
        colors=labels[:, 5:8],
        colors_hls=labels[:, 8:11],
        unpaired=np.zeros_like(labels[:, 0]),
    )


def load_labels(path_root: Path) -> Dataset:
    labels = np.load(path_root)
    return Dataset(
        classes=labels[:, 0],
        locations=labels[:, 1:3],
        sizes=labels[:, 3],
        rotations=labels[:, 4],
        colors=labels[:, 5:8],
        colors_hls=labels[:, 8:11],
        unpaired=labels[:, 11],
    )


def save_labels(path_root: Path, dataset: Dataset) -> None:
    labels = np.concatenate(
        [
            dataset.classes.reshape((-1, 1)),
            dataset.locations,
            dataset.sizes.reshape((-1, 1)),
            dataset.rotations.reshape((-1, 1)),
            dataset.colors,
            dataset.colors_hls,
            dataset.unpaired.reshape((-1, 1)),
        ],
        axis=1,
    ).astype(np.float32)
    np.save(path_root, labels)


def save_bert_latents(
    sentences: list[str],
    bert_path: str,
    output_path: Path,
    split: str,
    device: torch.device,
    compute_statistics: bool = False,
) -> None:
    transformer = cast(BertModel, BertModel.from_pretrained(bert_path))
    transformer.eval()
    transformer.to(device)  # type: ignore
    for p in transformer.parameters():
        p.requires_grad_(False)
    tokenizer = BertTokenizer.from_pretrained(bert_path)

    batch_size = 64
    latents = []
    n_batches = len(sentences) // batch_size
    if n_batches * batch_size < len(sentences):
        n_batches += 1
    for k in tqdm(range(n_batches)):
        start_k = k * batch_size
        end_k = min((k + 1) * batch_size, len(sentences))
        batch = sentences[start_k:end_k]
        tokens = tokenizer(batch, return_tensors="pt", padding=True).to(device)
        z = transformer(**tokens)["last_hidden_state"][:, 0]  # type: ignore
        latents.append(z.cpu().numpy())
    all_latents = np.concatenate(latents, axis=0)
    np.save(output_path / f"{split}_latent.npy", all_latents)

    if compute_statistics:
        mean = all_latents.mean(axis=0)
        std = all_latents.std(axis=0)
        np.save(output_path / "latent_mean.npy", mean)
        np.save(output_path / "latent_std.npy", std)


def get_domain_alignment(
    seed: int,
    allowed_indices: Sequence[int] | np.ndarray,
    alignment_groups_props: dict[frozenset[str], float],
) -> dict[frozenset[str], np.ndarray]:
    dataset_size = len(allowed_indices)

    alignment_groups_amounts = {
        domain_group: int(dataset_size * alignment_groups_props[domain_group])
        for domain_group in alignment_groups_props
    }

    rng = np.random.default_rng(seed)
    rng_streams = {
        domain_group: stream
        for domain_group, stream in zip(
            alignment_groups_props.keys(),
            rng.spawn(len(alignment_groups_props)),
            strict=True,
        )
    }

    selection = {
        domain_group: (
            np.array([], dtype=np.int64),
            rng_stream.permutation(allowed_indices),
        )
        for domain_group, rng_stream in rng_streams.items()
    }

    dependency_graph = build_dependency_graph(list(alignment_groups_props))

    while len(dependency_graph.nodes):
        roots = dependency_graph.get_roots()
        for root in roots:
            define_domain_split(
                selection,
                alignment_groups_amounts,
                root,
            )
        dependency_graph.remove_nodes(roots)

    return {
        domain_group: np.sort(selected)
        for domain_group, (selected, _) in selection.items()
    }


def define_domain_split(
    selection: dict[frozenset[str], tuple[np.ndarray, np.ndarray]],
    alignment_groups_amounts: dict[frozenset[str], int],
    domain_group: frozenset[str],
) -> None:
    nb_selected = alignment_groups_amounts[domain_group]
    assert nb_selected >= 0
    _, selected = selection[domain_group]
    selected = selected[:nb_selected]

    for target_domain_group in selection:
        if target_domain_group <= domain_group:
            target_selected, target_remaining = selection[target_domain_group]
            new_selected = np.unique(np.concatenate([target_selected, selected]))
            nb_added = len(new_selected) - len(target_selected)
            new_remaining = np.setdiff1d(target_remaining, selected, assume_unique=True)
            selection[target_domain_group] = (new_selected, new_remaining)

            alignment_groups_amounts[target_domain_group] -= nb_added
            assert alignment_groups_amounts[target_domain_group] >= 0


def get_deterministic_name(
    domain_alignment: Mapping[frozenset[str], float], seed: int, max_size: int
) -> str:
    domain_names = {
        ",".join(sorted(list(domain))): prop
        for domain, prop in domain_alignment.items()
    }
    sorted_domain_names = sorted(
        list(domain_names.items()),
        key=lambda x: x[0],
    )

    alignment_split_name = (
        "_".join([f"{domain}:{prop}" for domain, prop in sorted_domain_names])
        + f"_seed:{seed}_ms:{max_size}"
    )
    return alignment_split_name
