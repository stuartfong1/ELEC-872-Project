"""t-SNE of different (undistorted) images of the same people.

This shows the variation between different photos of the same person, which is
the baseline that the compression variation in 2.1.2 is compared against.

For each of the randomly chosen people, every one of their photos is embedded
with the IDENTITY processor (no distortion), and the points are coloured by person.

Usage
-----
    PYTHONPATH=. python phase2/2.1.1_tsne_subject.py --config config/resnet50_tsne.yaml

"""
from __future__ import annotations

import matplotlib.pyplot as plt
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


def main() -> None:
    """Embed every undistorted photo of the selected people and plot a t-SNE."""
    config = load_config_from_arguments()
    model, dataset, device = load_model_and_dataset(config)
    selected_identity_labels = select_people(model, dataset, config, device)

    # Every photo of every selected person, without any distortion
    flat_indices = [
        flat_dataset_index(dataset, source_image_index, IDENTITY_PROCESSOR_INDEX)
        for identity_label in selected_identity_labels
        for source_image_index in source_image_indices_of_person(dataset, identity_label)
    ]
    embedded_items = embed_dataset_items(model, dataset, flat_indices, config, device)
    tsne_coordinates = run_tsne(embedded_items["embeddings"])

    # One colour per person (tab10 has exactly 10 colours)
    figure, axis = plt.subplots(figsize=(8, 7))
    colour_map = plt.get_cmap("tab10")
    for colour_index, identity_label in enumerate(selected_identity_labels):
        is_this_person = embedded_items["identity_labels"] == identity_label
        axis.scatter(
            tsne_coordinates[is_this_person, 0],
            tsne_coordinates[is_this_person, 1],
            color=colour_map(colour_index),
            label=dataset.identity_names[identity_label],
            s=30,
        )

    axis.set_title("t-SNE of different photos of each person")
    axis.set_xticks([])
    axis.set_yticks([])
    axis.legend(title="Person", bbox_to_anchor=(1.02, 1), loc="upper left")
    figure.tight_layout()

    FIGURE_DIR.mkdir(exist_ok=True)
    figure.savefig(FIGURE_DIR / "2.1.1_tsne_subject.png", dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()
