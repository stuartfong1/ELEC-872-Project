"""Shared code for the phase 2 t-SNE scripts.

The two scripts (2.1.1 and 2.1.2) both load the same config, model and dataset,
pick the same random subset of people, and embed a subset of the dataset.
That shared logic lives here so that the two scripts only contain the
selection and plotting specific to each experiment.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.manifold import TSNE
from torch import nn
from torch.utils.data import DataLoader, Subset

from data.face_metadata_dataset import FaceMetadataDataset
from data.processors import Processor
from model.model_registry import ModelRegistry

# Distortions compared in phase 2. Mask and Rescale-based processors are excluded.
# IDENTITY must stay first: the scripts use processor index 0 as "undistorted".
PROCESSORS_TO_COMPARE: list[Processor] = [
    Processor.IDENTITY,
    Processor.JPEG75_420,
    Processor.JPEG75_444,
    Processor.JPEG85_420,
    Processor.JPEG85_444,
    Processor.JPEG95_420,
    Processor.JPEG95_444,
    Processor.SHARPEN_2,
    Processor.SHARPEN_4,
]
IDENTITY_PROCESSOR_INDEX = 0

# Number of people shown in each plot, and the seed used to pick them and run t-SNE.
# How people are picked ("random" or "nearest") is set by people_selection in the YAML config.
NUMBER_OF_PEOPLE_TO_PLOT = 10
RANDOM_SEED = 0

# Folder where the PNG figures are saved
FIGURE_DIR = Path(__file__).parent / "figures"


def load_config_from_arguments() -> dict:
    """Parse --config from the command line and load that YAML file.

    Returns
    -------
    dict
        Settings from the YAML config.

    """
    # Read YAML config
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    # Just in case our dataset is in $SLURM_TMPDIR
    parser.add_argument("--data_dir", type=Path)
    args = parser.parse_args()

    with Path.open(args.config) as config_file:
        config = yaml.safe_load(config_file)

    # Override dataset directory in YAML path if specified in arguments
    if args.data_dir is not None:
        config["dataset_dir"] = args.data_dir / "lfw-deepfunneled" / "lfw-deepfunneled"
        # Also override the path to the people_csv_path.csv file
        # for specifying the subjects in the test split
        config["people_csv_path"] = args.data_dir / "peopleDevTest.csv"

    return config



def load_model_and_dataset(config: dict) -> tuple[nn.Module, FaceMetadataDataset, str]:
    """Load the checkpointed model and the LFW test dataset.

    Parameters
    ----------
    config : dict
        Settings from the YAML config. weights_path must point to a checkpoint.

    Returns
    -------
    model : nn.Module
        Embedding model in eval mode, on device.
    dataset : FaceMetadataDataset
        LFW test set with the processors in PROCESSORS_TO_COMPARE.
    device : str
        Device the model is on.

    """
    # The registry builds the model, loads config["weights_path"],
    # and returns the matching preprocessing transform
    ModelRegistry.set_config(config)
    retrieved_model = ModelRegistry.get_model(config["model_name"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = retrieved_model.model.to(device).eval()

    dataset = FaceMetadataDataset(
        dataset_dir=config["dataset_dir"],
        people_csv_path=config["people_csv_path"],
        transform=retrieved_model.transform,
        processors=PROCESSORS_TO_COMPARE,
    )
    return model, dataset, device


def select_people(
    model: nn.Module,
    dataset: FaceMetadataDataset,
    config: dict,
    device: str,
) -> list[int]:
    """Pick NUMBER_OF_PEOPLE_TO_PLOT identity labels using config["people_selection"].

    Parameters
    ----------
    model : nn.Module
        Embedding model in eval mode (only used by "nearest").
    dataset : FaceMetadataDataset
        Dataset whose identities are selected.
    config : dict
        Settings from the YAML config. people_selection is either
        "random" (seeded random sample) or "nearest" (people whose mean
        embeddings are closest to each other, see select_nearest_people).
    device : str
        Device the model is on.

    Returns
    -------
    list[int]
        Sorted identity labels of the selected people.

    """
    match config["people_selection"]:
        case "random":
            random_generator = random.Random(RANDOM_SEED)
            number_of_identities = len(dataset.identity_names)
            return sorted(random_generator.sample(range(number_of_identities), NUMBER_OF_PEOPLE_TO_PLOT))
        case "nearest":
            return select_nearest_people(model, dataset, config, device)
        case _:
            raise ValueError(f"Unknown people_selection: {config['people_selection']}")


def select_nearest_people(
    model: nn.Module,
    dataset: FaceMetadataDataset,
    config: dict,
    device: str,
) -> list[int]:
    """Pick the tightest group of people in embedding space.

    Every person is summarised by the centroid of their undistorted photo
    embeddings, renormalised to unit norm. Writing K = NUMBER_OF_PEOPLE_TO_PLOT,
    each person in turn is treated as an anchor with its K-1 nearest centroids
    (by cosine distance) as its neighbourhood. The anchor whose neighbourhood
    has the smallest mean distance to the anchor is chosen, and the anchor plus
    its neighbours are returned.

    Parameters
    ----------
    model : nn.Module
        Embedding model in eval mode.
    dataset : FaceMetadataDataset
        Dataset whose identities are selected.
    config : dict
        Settings from the YAML config (batch_size and num_workers).
    device : str
        Device the model is on.

    Returns
    -------
    list[int]
        Sorted identity labels of the K selected people.

    """
    # Embed every undistorted photo of every valid person
    flat_indices = [
        flat_dataset_index(dataset, source_image_index, IDENTITY_PROCESSOR_INDEX)
        for source_image_index in range(len(dataset.image_paths))
    ]
    embedded_items = embed_dataset_items(model, dataset, flat_indices, config, device)
    embeddings = embedded_items["embeddings"]
    identity_labels = embedded_items["identity_labels"]

    # Centroid of each person, projected back onto the unit sphere so that
    # 1 - dot product is the cosine distance between centroids
    number_of_identities = len(dataset.identity_names)
    centroids = np.stack([
        embeddings[identity_labels == identity_label].mean(axis=0)
        for identity_label in range(number_of_identities)
    ])
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)
    centroid_distances = 1.0 - centroids @ centroids.T
    # A person must not count as their own neighbour
    np.fill_diagonal(centroid_distances, np.inf)

    # Loop over every person as the anchor and score their neighbourhood
    number_of_neighbours = NUMBER_OF_PEOPLE_TO_PLOT - 1
    best_mean_distance = np.inf
    best_group: list[int] = []
    for anchor_identity_label in range(number_of_identities):
        nearest_identity_labels = np.argsort(centroid_distances[anchor_identity_label])[:number_of_neighbours]
        mean_distance = centroid_distances[anchor_identity_label, nearest_identity_labels].mean()
        if mean_distance < best_mean_distance:
            best_mean_distance = mean_distance
            best_group = [anchor_identity_label, *nearest_identity_labels.tolist()]

    print(f"Nearest group: mean anchor-to-neighbour centroid cosine distance = {best_mean_distance:.4f}")
    return sorted(best_group)


def source_image_indices_of_person(dataset: FaceMetadataDataset, identity_label: int) -> list[int]:
    """List the source image indices (into dataset.image_paths) of one person.

    Parameters
    ----------
    dataset : FaceMetadataDataset
        Dataset to search.
    identity_label : int
        Identity label of the person.

    Returns
    -------
    list[int]
        Source image indices of that person, in sorted file-name order.

    """
    return [
        source_image_index
        for source_image_index, image_identity_label in enumerate(dataset.identity_labels)
        if image_identity_label == identity_label
    ]


def flat_dataset_index(dataset: FaceMetadataDataset, source_image_index: int, processor_index: int) -> int:
    """Convert (source image, processor) into the dataset's flat index.

    Parameters
    ----------
    dataset : FaceMetadataDataset
        Dataset whose indexing scheme is used.
    source_image_index : int
        Index into dataset.image_paths.
    processor_index : int
        Index into dataset.processors.

    Returns
    -------
    int
        source_image_index * number_of_processors + processor_index.

    """
    return source_image_index * len(dataset.processors) + processor_index


def embed_dataset_items(
    model: nn.Module,
    dataset: FaceMetadataDataset,
    flat_indices: list[int],
    config: dict,
    device: str,
) -> dict[str, np.ndarray]:
    """Embed the chosen dataset items and L2-normalise the embeddings.

    Parameters
    ----------
    model : nn.Module
        Embedding model in eval mode.
    dataset : FaceMetadataDataset
        Dataset to draw items from.
    flat_indices : list[int]
        Flat dataset indices of the items to embed.
    config : dict
        Settings from the YAML config (batch_size and num_workers).
    device : str
        Device the model is on.

    Returns
    -------
    dict[str, np.ndarray]
        Arrays aligned with flat_indices:
        embeddings (N, embedding_dim) with unit L2 norm,
        distortion_labels (N,), identity_labels (N,),
        source_image_indices (N,).

    """
    # shuffle=False keeps the output in the same order as flat_indices
    dataloader = DataLoader(
        Subset(dataset, flat_indices),
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
    )

    embedding_batches = []
    distortion_label_batches = []
    identity_label_batches = []
    source_image_index_batches = []
    with torch.no_grad():
        for images, distortion_labels, identity_labels, source_image_indices in dataloader:
            embedding_batches.append(model(images.to(device)).cpu())
            distortion_label_batches.append(distortion_labels)
            identity_label_batches.append(identity_labels)
            source_image_index_batches.append(source_image_indices)

    # Normalise to match the distance used by the triplet loss in train.py.
    # On the unit sphere, squared Euclidean distance = 2 * cosine distance.
    embeddings = F.normalize(torch.cat(embedding_batches))

    return {
        "embeddings": embeddings.numpy(),
        "distortion_labels": torch.cat(distortion_label_batches).numpy(),
        "identity_labels": torch.cat(identity_label_batches).numpy(),
        "source_image_indices": torch.cat(source_image_index_batches).numpy(),
    }


def run_tsne(embeddings: np.ndarray) -> np.ndarray:
    """Project embeddings to 2D with t-SNE.

    Parameters
    ----------
    embeddings : np.ndarray
        Array of shape (N, embedding_dim).

    Returns
    -------
    np.ndarray
        Array of shape (N, 2) with the t-SNE coordinates.

    """
    # Perplexity must be smaller than the number of points
    perplexity = min(30.0, len(embeddings) - 1)
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        random_state=RANDOM_SEED,
    )
    return tsne.fit_transform(embeddings)
