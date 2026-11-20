"""Distribution of JPEG-induced embedding shifts, and a t-SNE of a high-variation group.

For every photo in the LFW test set, the photo is embedded under two JPEG
settings (JPEG75_420 and JPEG95_444), and the cosine distance between the two
embeddings is measured. This script saves two figures:

    1. A histogram of these per-photo distances over the whole test set.
    2. A t-SNE plot (same style as 2.1.3) of a group of K people chosen as:
         - anchor: the person whose photos have the largest median JPEG distance,
         - neighbours: the K-1 people whose centroids are closest to the anchor's.
       The hope is that people near a high-variation person also vary a lot.

The embeddings computed for the histogram are reused for the t-SNE plot,
so the dataset is only embedded once.

Usage
-----
    PYTHONPATH=. python phase2/2.1.4_tsne_distances.py --config config/resnet50_tsne.yaml

"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from tsne_utils import (
    FIGURE_DIR,
    NUMBER_OF_PEOPLE_TO_PLOT,
    embed_dataset_items,
    flat_dataset_index,
    load_config_from_arguments,
    load_model_and_dataset,
    run_tsne,
)

from data.processors import Processor

# The two JPEG settings to compare, each with its own marker shape in the t-SNE plot
JPEG_PROCESSORS_AND_MARKERS: list[tuple[Processor, str]] = [
    (Processor.JPEG75_420, "o"),
    (Processor.JPEG95_444, "^"),
]


def per_photo_jpeg_distances(
    embedded_items: dict[str, np.ndarray],
    jpeg_processor_indices: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Cosine distance between the two JPEG embeddings of each photo.

    Parameters
    ----------
    embedded_items : dict[str, np.ndarray]
        Output of ``embed_dataset_items`` containing every photo under both
        JPEG settings (embeddings have unit L2 norm).
    jpeg_processor_indices : list[int]
        Distortion labels of the two JPEG settings, in the order
        [strong compression, weak compression].

    Returns
    -------
    photo_distances : np.ndarray
        Array of shape (number_of_photos,) with one cosine distance per photo.
    photo_identity_labels : np.ndarray
        Array of shape (number_of_photos,) with the identity of each photo.

    """
    first_setting_rows = np.flatnonzero(embedded_items["distortion_labels"] == jpeg_processor_indices[0])
    second_setting_rows = np.flatnonzero(embedded_items["distortion_labels"] == jpeg_processor_indices[1])

    # Both row lists are in increasing source-image order, so row i of each is the same photo
    assert np.array_equal(
        embedded_items["source_image_indices"][first_setting_rows],
        embedded_items["source_image_indices"][second_setting_rows],
    )

    # Unit-norm embeddings: cosine distance = 1 - dot product (row-wise)
    photo_distances = 1.0 - np.sum(
        embedded_items["embeddings"][first_setting_rows] * embedded_items["embeddings"][second_setting_rows],
        axis=1,
    )
    photo_identity_labels = embedded_items["identity_labels"][first_setting_rows]
    return photo_distances, photo_identity_labels


def select_high_variation_group(
    embedded_items: dict[str, np.ndarray],
    photo_distances: np.ndarray,
    photo_identity_labels: np.ndarray,
    number_of_identities: int,
) -> tuple[list[int], np.ndarray]:
    """Pick the highest-variation person and the K-1 people nearest to them.

    Parameters
    ----------
    embedded_items : dict[str, np.ndarray]
        Output of ``embed_dataset_items`` (embeddings have unit L2 norm).
    photo_distances : np.ndarray
        Per-photo JPEG cosine distances.
    photo_identity_labels : np.ndarray
        Identity of each entry of ``photo_distances``.
    number_of_identities : int
        Number of people in the dataset.

    Returns
    -------
    selected_identity_labels : list[int]
        The anchor first, followed by its K-1 nearest people (closest first).
    median_distance_by_identity : np.ndarray
        Array of shape (number_of_identities,) with each person's median JPEG distance.

    """
    # Median JPEG distance of each person over their photos.
    # The median is used instead of the mean so that one or two photos with a
    # very large distance cannot make a person the anchor on their own.
    median_distance_by_identity = np.array([
        np.median(photo_distances[photo_identity_labels == identity_label])
        for identity_label in range(number_of_identities)
    ])
    # The anchor is the person with the largest median distance
    anchor_identity_label = int(np.argmax(median_distance_by_identity))

    # Centroid of each person over all their embeddings (both JPEG settings),
    # projected back onto the unit sphere
    centroids = np.stack([
        embedded_items["embeddings"][embedded_items["identity_labels"] == identity_label].mean(axis=0)
        for identity_label in range(number_of_identities)
    ])
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)

    # Cosine distance from the anchor's centroid to every centroid.
    # The anchor itself is at distance 0, so it is always first in the sort; skip it.
    distances_to_anchor = 1.0 - centroids @ centroids[anchor_identity_label]
    distances_to_anchor[anchor_identity_label] = np.inf
    nearest_identity_labels = np.argsort(distances_to_anchor)[: NUMBER_OF_PEOPLE_TO_PLOT - 1]

    selected_identity_labels = [anchor_identity_label, *nearest_identity_labels.tolist()]
    return selected_identity_labels, median_distance_by_identity


def save_histogram(photo_distances: np.ndarray) -> None:
    """Save a histogram of the per-photo JPEG distances.

    Parameters
    ----------
    photo_distances : np.ndarray
        Per-photo cosine distances between the two JPEG settings.

    """
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.hist(photo_distances, bins=50, color="tab:blue")

    # Mark the mean and median, since the distribution is expected to be skewed
    axis.axvline(photo_distances.mean(), color="black", linestyle="--", label=f"Mean = {photo_distances.mean():.4f}")
    axis.axvline(np.median(photo_distances), color="black", linestyle=":", label=f"Median = {np.median(photo_distances):.4f}")

    first_name, second_name = (processor.name for processor, _ in JPEG_PROCESSORS_AND_MARKERS)
    axis.set_title(f"Cosine distance between {first_name} and {second_name} embeddings of the same photo")
    axis.set_xlabel("Cosine distance")
    axis.set_ylabel("Number of photos")
    axis.legend()
    figure.tight_layout()

    figure.savefig(FIGURE_DIR / "2.1.4_jpeg_distance_histogram.png", dpi=200, bbox_inches="tight")


def save_tsne_plot(
    embedded_items: dict[str, np.ndarray],
    selected_identity_labels: list[int],
    jpeg_processor_indices: list[int],
    identity_names: list[str],
) -> None:
    """Save a t-SNE plot of the selected people under both JPEG settings (2.1.3 style).

    Parameters
    ----------
    embedded_items : dict[str, np.ndarray]
        Output of ``embed_dataset_items`` for the whole test set.
    selected_identity_labels : list[int]
        People to plot, with the anchor first.
    jpeg_processor_indices : list[int]
        Distortion labels of the two JPEG settings.
    identity_names : list[str]
        Names of every person, indexed by identity label.

    """
    # Keep only the rows of the selected people, then run t-SNE on just those
    is_selected = np.isin(embedded_items["identity_labels"], selected_identity_labels)
    selected_identity_label_per_row = embedded_items["identity_labels"][is_selected]
    selected_distortion_label_per_row = embedded_items["distortion_labels"][is_selected]
    tsne_coordinates = run_tsne(embedded_items["embeddings"][is_selected])

    # Colour = person (tab10 has exactly 10 colours), marker = JPEG setting
    figure, axis = plt.subplots(figsize=(8, 7))
    colour_map = plt.get_cmap("tab10")
    for colour_index, identity_label in enumerate(selected_identity_labels):
        is_this_person = selected_identity_label_per_row == identity_label
        for processor_index, (_, marker) in zip(jpeg_processor_indices, JPEG_PROCESSORS_AND_MARKERS):
            is_this_point = is_this_person & (selected_distortion_label_per_row == processor_index)
            axis.scatter(
                tsne_coordinates[is_this_point, 0],
                tsne_coordinates[is_this_point, 1],
                color=colour_map(colour_index),
                marker=marker,
                s=30,
            )

    # Two legends: colour = person (anchor marked), marker = JPEG setting
    person_legend_handles = [
        Line2D(
            [], [],
            color=colour_map(colour_index),
            marker="o",
            linestyle="",
            label=identity_names[identity_label] + (" (anchor)" if colour_index == 0 else ""),
        )
        for colour_index, identity_label in enumerate(selected_identity_labels)
    ]
    jpeg_legend_handles = [
        Line2D([], [], color="grey", marker=marker, linestyle="", label=processor.name)
        for processor, marker in JPEG_PROCESSORS_AND_MARKERS
    ]
    person_legend = axis.legend(handles=person_legend_handles, title="Person", bbox_to_anchor=(1.02, 1), loc="upper left")
    axis.add_artist(person_legend)
    axis.legend(handles=jpeg_legend_handles, title="JPEG setting", bbox_to_anchor=(1.02, 0), loc="lower left")

    axis.set_title("t-SNE around the person with the largest JPEG variation")
    axis.set_xticks([])
    axis.set_yticks([])
    figure.tight_layout()

    figure.savefig(FIGURE_DIR / "2.1.4_tsne_high_variation.png", dpi=200, bbox_inches="tight")


def main() -> None:
    """Embed the test set under two JPEG settings, then save the histogram and t-SNE plot."""
    config = load_config_from_arguments()
    model, dataset, device = load_model_and_dataset(config)

    # Position of each JPEG setting within dataset.processors (its distortion label)
    jpeg_processor_indices = [
        dataset.processors.index(processor) for processor, _ in JPEG_PROCESSORS_AND_MARKERS
    ]

    # Every photo in the test set, under both JPEG settings
    flat_indices = [
        flat_dataset_index(dataset, source_image_index, processor_index)
        for source_image_index in range(len(dataset.image_paths))
        for processor_index in jpeg_processor_indices
    ]
    embedded_items = embed_dataset_items(model, dataset, flat_indices, config, device)

    # Per-photo distances and the high-variation group
    photo_distances, photo_identity_labels = per_photo_jpeg_distances(embedded_items, jpeg_processor_indices)
    selected_identity_labels, median_distance_by_identity = select_high_variation_group(
        embedded_items, photo_distances, photo_identity_labels, len(dataset.identity_names),
    )

    # Summary statistics, including how much the anchor's neighbours vary
    print(f"Per-photo JPEG distance over {len(photo_distances)} photos: "
          f"mean {photo_distances.mean():.4f}, median {np.median(photo_distances):.4f}, max {photo_distances.max():.4f}")
    print(f"Median of the per-person median JPEG distances: {np.median(median_distance_by_identity):.4f}")
    print("Selected people (anchor first) and their median JPEG distance:")
    for identity_label in selected_identity_labels:
        print(f"  {dataset.identity_names[identity_label]:<30}: {median_distance_by_identity[identity_label]:.4f}")

    FIGURE_DIR.mkdir(exist_ok=True)
    save_histogram(photo_distances)
    save_tsne_plot(embedded_items, selected_identity_labels, jpeg_processor_indices, dataset.identity_names)


if __name__ == "__main__":
    main()
