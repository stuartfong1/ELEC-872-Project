"""t-SNE of one photo per person under different distortions, plus distance statistics.

To make the compression variation comparable with the variation between photos,
both are embedded in the same t-SNE map:
    1. Every photo of each selected person, undistorted (as in 2.1.1).
    2. The first photo of each selected person, under every distortion.
If the distorted copies of a photo stay much closer together than different
photos of the same person, the distortions are unlikely to cause misclassification.

Since t-SNE does not preserve distances, cosine distances in the original
embedding space are also printed:
    - between different undistorted photos of the same person,
    - between undistorted photos of different people,
    - between each distorted photo and its undistorted original, per distortion.

Usage
-----
    PYTHONPATH=. python phase2/2.1.2_tsne_compression.py --config config/resnet50_tsne.yaml

"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from tsne_utils import (
    FIGURE_DIR,
    IDENTITY_PROCESSOR_INDEX,
    embed_dataset_items,
    flat_dataset_index,
    load_config_from_arguments,
    load_model_and_dataset,
    run_tsne,
    select_people,
    source_image_indices_of_person,
)

# One marker per processor, in the same order as PROCESSORS_TO_COMPARE
PROCESSOR_MARKERS = ["o", "v", "^", "<", ">", "s", "p", "*", "D"]


def print_distance_statistics(embedded_items: dict[str, np.ndarray], dataset_processor_names: list[str]) -> None:
    """Print mean cosine distances between photos, people, and distortions.

    Parameters
    ----------
    embedded_items : dict[str, np.ndarray]
        Output of embed_dataset_items (embeddings have unit L2 norm).
    dataset_processor_names : list[str]
        Names of the processors, indexed by distortion label.

    """
    embeddings = embedded_items["embeddings"]
    distortion_labels = embedded_items["distortion_labels"]
    identity_labels = embedded_items["identity_labels"]
    source_image_indices = embedded_items["source_image_indices"]

    # ----- Undistorted photos: same person vs different people -----
    # Embeddings are unit-norm, so cosine distance = 1 - dot product.
    is_undistorted = distortion_labels == IDENTITY_PROCESSOR_INDEX
    undistorted_embeddings = embeddings[is_undistorted]
    undistorted_identity_labels = identity_labels[is_undistorted]
    cosine_distances = 1.0 - undistorted_embeddings @ undistorted_embeddings.T

    # Only count each unordered pair once (strict upper triangle)
    is_distinct_pair = np.triu(np.ones_like(cosine_distances, dtype=bool), k=1)
    is_same_person = undistorted_identity_labels[:, None] == undistorted_identity_labels[None, :]
    same_person_distances = cosine_distances[is_distinct_pair & is_same_person]
    different_person_distances = cosine_distances[is_distinct_pair & ~is_same_person]

    print("Cosine distance between undistorted photos (mean ± std)")
    print(f"  Same person, different photos : {same_person_distances.mean():.4f} ± {same_person_distances.std():.4f}")
    print(f"  Different people              : {different_person_distances.mean():.4f} ± {different_person_distances.std():.4f}")

    # ----- Each distorted photo vs its own undistorted original -----
    # Map each source image to the row of its undistorted embedding
    undistorted_row_by_source_image = {
        int(source_image_indices[row]): row for row in np.flatnonzero(is_undistorted)
    }
    print("Cosine distance from each distorted photo to its undistorted original (mean ± std)")
    for processor_index, processor_name in enumerate(dataset_processor_names):
        if processor_index == IDENTITY_PROCESSOR_INDEX:
            continue
        distorted_rows = np.flatnonzero(distortion_labels == processor_index)
        original_rows = [undistorted_row_by_source_image[int(source_image_indices[row])] for row in distorted_rows]
        # Row-wise dot product between each distorted embedding and its original
        distortion_distances = 1.0 - np.sum(embeddings[distorted_rows] * embeddings[original_rows], axis=1)
        print(f"  {processor_name:<14}: {distortion_distances.mean():.4f} ± {distortion_distances.std():.4f}")


def main() -> None:
    """Embed undistorted photos plus distorted copies of one photo per person, then plot."""
    config = load_config_from_arguments()
    model, dataset, device = load_model_and_dataset(config)
    selected_identity_labels = select_people(model, dataset, config, device)

    flat_indices = []
    for identity_label in selected_identity_labels:
        person_source_image_indices = source_image_indices_of_person(dataset, identity_label)

        # (1) Every photo of this person, undistorted
        flat_indices.extend(
            flat_dataset_index(dataset, source_image_index, IDENTITY_PROCESSOR_INDEX)
            for source_image_index in person_source_image_indices
        )

        # (2) The person's first photo under every other distortion
        # (its undistorted version was already added in (1))
        first_source_image_index = person_source_image_indices[0]
        flat_indices.extend(
            flat_dataset_index(dataset, first_source_image_index, processor_index)
            for processor_index in range(len(dataset.processors))
            if processor_index != IDENTITY_PROCESSOR_INDEX
        )

    embedded_items = embed_dataset_items(model, dataset, flat_indices, config, device)
    print_distance_statistics(embedded_items, dataset.processor_names)
    tsne_coordinates = run_tsne(embedded_items["embeddings"])

    # The "chosen" photo of each person is the one that receives every distortion
    chosen_source_image_indices = {
        source_image_indices_of_person(dataset, identity_label)[0] for identity_label in selected_identity_labels
    }
    is_chosen_photo = np.isin(embedded_items["source_image_indices"], list(chosen_source_image_indices))

    figure, axis = plt.subplots(figsize=(9, 7))
    colour_map = plt.get_cmap("tab10")
    for colour_index, identity_label in enumerate(selected_identity_labels):
        is_this_person = embedded_items["identity_labels"] == identity_label
        person_colour = colour_map(colour_index)

        # Other photos of this person: small faded dots for context
        is_other_photo = is_this_person & ~is_chosen_photo
        axis.scatter(
            tsne_coordinates[is_other_photo, 0],
            tsne_coordinates[is_other_photo, 1],
            color=person_colour,
            alpha=0.35,
            s=15,
        )

        # The chosen photo under each distortion: one marker per distortion
        for processor_index, marker in enumerate(PROCESSOR_MARKERS):
            is_this_point = is_this_person & is_chosen_photo & (embedded_items["distortion_labels"] == processor_index)
            axis.scatter(
                tsne_coordinates[is_this_point, 0],
                tsne_coordinates[is_this_point, 1],
                color=person_colour,
                marker=marker,
                edgecolors="black",
                linewidths=0.5,
                s=60,
            )

    # Two legends: colour = person, marker = distortion
    person_legend_handles = [
        Line2D([], [], color=colour_map(colour_index), marker="o", linestyle="", label=dataset.identity_names[identity_label])
        for colour_index, identity_label in enumerate(selected_identity_labels)
    ]
    processor_legend_handles = [
        Line2D([], [], color="grey", markeredgecolor="black", marker=marker, linestyle="", label=processor_name)
        for marker, processor_name in zip(PROCESSOR_MARKERS, dataset.processor_names)
    ]
    person_legend = axis.legend(handles=person_legend_handles, title="Person", bbox_to_anchor=(1.02, 1), loc="upper left")
    axis.add_artist(person_legend)
    axis.legend(handles=processor_legend_handles, title="Distortion (chosen photo)", bbox_to_anchor=(1.02, 0), loc="lower left")

    axis.set_title("t-SNE: one photo per person under each distortion\n(faded dots: other undistorted photos)")
    axis.set_xticks([])
    axis.set_yticks([])
    figure.tight_layout()

    FIGURE_DIR.mkdir(exist_ok=True)
    figure.savefig(FIGURE_DIR / "2.1.2_tsne_compression.png", dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()
