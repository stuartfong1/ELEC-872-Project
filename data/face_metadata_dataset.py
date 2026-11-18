from __future__ import annotations

import csv
from pathlib import Path

import torch
from data.processors import Processor
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2


class FaceMetadataDataset(Dataset):
    """LFW test-set dataset that returns every image under every distortion.

    The LFW directory should be structured as follows:

        lfw_dir/
            AJ_Cook/
                AJ_Cook_0001.jpg
            AJ_Lamas/
                AJ_Lamas_0001.jpg
                AJ_Lamas_0002.jpg
            ...

    The people in the test set are read from peopleDevTest.csv (header name,images).
    Only people with at least minimum_images_per_person images are kept.

    Every (source image, processor) pair is one item of the dataset, so the
    dataset has number_of_source_images * number_of_processors items.
    The flat index is laid out as

        index = source_image_index * number_of_processors + processor_index

    so all distortions of one source image are next to each other.
    Distortions are applied on the fly to the raw PIL image, before the model
    preprocessing transform (resize, center crop, normalize).

    """

    def __init__(
        self,
        dataset_dir: str | Path,
        people_csv_path: str | Path,
        transform = None,
        *,
        minimum_images_per_person: int = 5,
        processors: list[Processor] | None = None,
    ) -> None:
        """Initialize the FaceMetadataDataset class.

        Parameters
        ----------
        dataset_dir : str or Path
            Root directory of the dataset, containing one folder per person.
        people_csv_path : str or Path
            Path to peopleDevTest.csv, with header name,images.
        transform : Callable[[Image.Image], torch.Tensor] or None, optional
            Model preprocessing transform applied after the distortion.
            If None, the image is only converted to a float tensor in [0, 1].
        minimum_images_per_person : int, optional
            People with fewer images than this are dropped. Default is 5.
        processors : list[Processor] or None, optional
            Distortions to apply to every image. If None (default), every
            member of the Processor enum is used, in definition order.

        """
        if transform is None:
            self.transform = v2.Compose([
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
            ])
        else:
            self.transform = transform

        # The distortion label of an item is its position in this list
        # (0, 1, 2, ...), not the 1-based value of the Processor enum.
        if processors is None:
            processors = list(Processor)
        self.processors: list[Processor] = processors
        # Human-readable names, e.g. "JPEG75_420"
        self.processor_names: list[str] = [processor.name for processor in processors]
        # The callable of each processor is built once here and reused.
        # Note: the random Rescale processors hold their own seeded RNG, so the
        # random scale drawn for an image depends on the order items are loaded.
        self.processor_functions = [processor.get_processing() for processor in processors]

        # File paths of each (undistorted) source image
        self.image_paths: list[Path] = []
        # Integer identities (person) of each source image
        self.identity_labels: list[int] = []
        # Names of the kept people, indexed by identity label
        self.identity_names: list[str] = []

        self.dataset_dir = Path(dataset_dir)
        with open(people_csv_path, newline="") as people_csv_file:
            people_rows = list(csv.DictReader(people_csv_file))

        # Sort by name so that each person always gets the same identity label
        for people_row in sorted(people_rows, key=lambda row: row["name"]):
            person_name = people_row["name"]
            person_image_paths = sorted((self.dataset_dir / person_name).glob("*.jpg"))

            # Sanity check: the image count in the CSV should match what is on disk
            assert len(person_image_paths) == int(people_row["images"]), (
                f"{person_name}: CSV lists {people_row['images']} images, "
                f"found {len(person_image_paths)} on disk"
            )

            # Drop people with too few images
            if len(person_image_paths) < minimum_images_per_person:
                continue

            # The new identity label is the number of people kept so far
            identity_label = len(self.identity_names)
            self.identity_names.append(person_name)
            self.image_paths.extend(person_image_paths)
            self.identity_labels.extend([identity_label] * len(person_image_paths))

    def __len__(self) -> int:
        """Return the number of (source image, processor) pairs.

        Returns
        -------
        int
            Number of source images times the number of processors.

        """
        return len(self.image_paths) * len(self.processors)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int, int]:
        """Return one distorted, preprocessed image and its labels.

        Parameters
        ----------
        index : int
            Flat index, equal to
            source_image_index * number_of_processors + processor_index.

        Returns
        -------
        image : torch.Tensor
            Distorted image after the model preprocessing transform.
        distortion_label : int
            Index into self.processors of the distortion that was applied.
        identity_label : int
            Identity (person) of the image, indexing self.identity_names.
        source_image_index : int
            Index into self.image_paths of the undistorted source image.
            Items sharing this value are the same photo under different distortions.

        """
        # Recover the source image and the processor from the flat index
        source_image_index, distortion_label = divmod(index, len(self.processors))

        # Load the raw image, distort it, then apply the model preprocessing.
        # The second convert("RGB") forces PIL to fully decode the in-memory JPEG
        # returned by JPEGCompression and guarantees 3 channels.
        raw_image = Image.open(self.image_paths[source_image_index]).convert("RGB")
        distorted_image = self.processor_functions[distortion_label](raw_image).convert("RGB")
        image = self.transform(distorted_image)

        return (
            image,
            distortion_label,
            self.identity_labels[source_image_index],
            source_image_index,
        )
