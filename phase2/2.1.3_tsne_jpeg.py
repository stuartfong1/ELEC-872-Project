"""t-SNE of every photo of the selected people under two JPEG compression settings.

Points are coloured by person and the marker shape shows the JPEG setting.

Usage
-----
    PYTHONPATH=. python phase2/2.1.3_tsne_jpeg.py --config config/resnet50_tsne.yaml

"""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tsne_utils import (
    FIGURE_DIR,
    embed_dataset_items,
    flat_dataset_index,
    load_config_from_arguments,
    load_model_and_dataset,
    run_tsne,
    select_people,
    source_image_indices_of_person,
)

from data.processors import Processor

# The two JPEG settings to compare, each with its own marker shape
JPEG_PROCESSORS_AND_MARKERS: list[tuple[Processor, str]] = [
    (Processor.JPEG75_420, "o"),
    (Processor.JPEG95_444, "^"),
]


def main() -> None:
    """Embed every photo of the selected people under two JPEG settings and plot a t-SNE."""
    config = load_config_from_arguments()
    model, dataset, device = load_model_and_dataset(config)
    selected_identity_labels = select_people(model, dataset, config, device)

    # Position of each JPEG setting within dataset.processors (its distortion label)
    jpeg_processor_indices = [
        dataset.processors.index(processor) for processor, _ in JPEG_PROCESSORS_AND_MARKERS
    ]

    # Every photo of every selected person, under both JPEG settings
    flat_indices = [
        flat_dataset_index(dataset, source_image_index, processor_index)
        for identity_label in selected_identity_labels
        for source_image_index in source_image_indices_of_person(dataset, identity_label)
        for processor_index in jpeg_processor_indices
    ]
    embedded_items = embed_dataset_items(model, dataset, flat_indices, config, device)
    tsne_coordinates = run_tsne(embedded_items["embeddings"])

    # Colour = person (tab10 has exactly 10 colours), marker = JPEG setting
    figure, axis = plt.subplots(figsize=(8, 7))
    colour_map = plt.get_cmap("tab10")
    for colour_index, identity_label in enumerate(selected_identity_labels):
        is_this_person = embedded_items["identity_labels"] == identity_label
        for processor_index, (_, marker) in zip(jpeg_processor_indices, JPEG_PROCESSORS_AND_MARKERS):
            is_this_point = is_this_person & (embedded_items["distortion_labels"] == processor_index)
            axis.scatter(
                tsne_coordinates[is_this_point, 0],
                tsne_coordinates[is_this_point, 1],
                color=colour_map(colour_index),
                marker=marker,
                s=30,
            )

    # Two legends: colour = person, marker = JPEG setting
    person_legend_handles = [
        Line2D([], [], color=colour_map(colour_index), marker="o", linestyle="", label=dataset.identity_names[identity_label])
        for colour_index, identity_label in enumerate(selected_identity_labels)
    ]
    jpeg_legend_handles = [
        Line2D([], [], color="grey", marker=marker, linestyle="", label=processor.name)
        for processor, marker in JPEG_PROCESSORS_AND_MARKERS
    ]
    person_legend = axis.legend(handles=person_legend_handles, title="Person", bbox_to_anchor=(1.02, 1), loc="upper left")
    axis.add_artist(person_legend)
    axis.legend(handles=jpeg_legend_handles, title="JPEG setting", bbox_to_anchor=(1.02, 0), loc="lower left")

    axis.set_title("t-SNE of different photos of each person under two JPEG settings")
    axis.set_xticks([])
    axis.set_yticks([])
    figure.tight_layout()

    FIGURE_DIR.mkdir(exist_ok=True)
    figure.savefig(FIGURE_DIR / "2.1.3_tsne_jpeg.png", dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()
