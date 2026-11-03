from __future__ import annotations

import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2


class FaceDataset(Dataset):
    """VGGFace2 face dataset that returns triplets or single labelled images.

    The dataset directory should be structured as follows:

        dataset_dir/
            train/
                n000002/
                    0001_01.jpg
                    ...
                ...
            val/
                n000001/
                    0001_01.jpg
                    ...
                ...

    During training (return_triplets=True), every image is used once per epoch as an
    anchor, paired with a random positive image (same identity) and a random negative
    image (different identity).
    During validation (return_triplets=False), each item is just one image and
    its identity label for nearest-neighbour evaluation.

    """

    def __init__(
        self,
        dataset_dir: str | Path,
        split: str,
        transform = None,
        *,
        return_triplets: bool = True,
    ) -> None:
        """Initialize the FaceDataset class.

        Parameters
        ----------
        dataset_dir : str or Path
            Root directory of the dataset, containing the split folders.
        split : str
            Name of the split folder to index, e.g. "train" or "val".
        transform : Callable[[Image.Image], torch.Tensor] or None, optional
            Transform applied to each loaded PIL image, typically the ResNet50
            preprocessing transform. If None, raw PIL images are returned.
        return_triplets : bool, optional
            If True (default), each item is an (anchor, positive, negative) triplet
            with labels. If False, each item is a single (image, label) pair.

        """
        if transform is None:
            self.transform = v2.Compose([
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
            ])
        else:
            self.transform = transform
        self.return_triplets = return_triplets

        # File paths of each image
        self.image_paths: list[str | Path] = []
        # Integer identities (person) of each image
        self.identity_labels: list[int] = []
        # For each identity, list the image indices belonging to that person
        self.image_indices_by_identity: list[list[int]] = []  # person[image_idx]
        split_dir = Path(dataset_dir) / split

        # Loop through the sorted (deterministic) list of identity folders
        # Sorting ensures that each person always gets the same label.
        for identity_label, identity_dir in enumerate(sorted(split_dir.iterdir())):
            identity_image_paths = sorted(identity_dir.glob("*.jpg"))

            # Each person's images are appended so that they are all next to each other,
            # so each person's images corresponds to a range of indices.
            # The start index is the current length of the image list
            first_index = len(self.image_paths)
            # Add the image paths to the list
            self.image_paths.extend(identity_image_paths)
            # Add the label of the person
            self.identity_labels.extend([identity_label] * len(identity_image_paths))
            # Add the range of image indices that contain this person
            self.image_indices_by_identity.append(
                list(range(first_index, first_index + len(identity_image_paths))),
            )

    def sample_triplet(
        self,
        anchor_index: int,
        random_generator: random.Random,
    ) -> tuple[int, int, int]:
        """Sample a positive and a negative image index for the given anchor.

        Parameters
        ----------
        anchor_index : int
            Global index of the anchor image.
        random_generator : random.Random
            Source of randomness. The ``random`` module itself can also be
            passed, since it exposes the same ``choice``/``randrange`` methods.

        Returns
        -------
        anchor : int
            Index of the anchor image
        positive : int
            Index of the positive image
        negative : int
            Index of the negative image

        """
        anchor_identity = self.identity_labels[anchor_index]
        anchor_identity_indices = self.image_indices_by_identity[anchor_identity]

        # The positive image must be a different image of the same person.
        positive_index = anchor_index
        assert len(anchor_identity_indices) > 1
        while positive_index == anchor_index:
            positive_index = random_generator.choice(anchor_identity_indices)

        # The negative image is drawn as follows:
        #   1. Pick a random person that is not the anchor
        #   2. Sample a random image from that person
        negative_identity = anchor_identity
        while negative_identity == anchor_identity:
            negative_identity = random_generator.randrange(len(self.image_indices_by_identity))
        negative_index = random_generator.choice(self.image_indices_by_identity[negative_identity])

        return anchor_index, positive_index, negative_index

    def load_image(self, index: int) -> torch.Tensor:
        """Load one image from path and apply the transform.

        Parameters
        ----------
        index : int
            Global index of the image to load.

        Returns
        -------
        torch.Tensor
            Transformed image tensor

        """
        # Convert to RGB
        image = Image.open(self.image_paths[index]).convert("RGB")
        return self.transform(image)

    def __len__(self) -> int:
        """Return the number of images in the split.

        Returns
        -------
        int
            Total number of images, which is also the number of anchors per epoch.

        """
        return len(self.image_paths)

    def __getitem__(
        self, index: int,
    ) -> (
            tuple[torch.Tensor, int]
            | tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]
        ):
        """Return a single labelled image or a labelled triplet.

        Parameters
        ----------
        index : int
            Index of the image (used as the anchor in triplet mode).

        Returns
        -------
        tuple[torch.Tensor, int]
            If return_triplets is False: (image, label).
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]
            If return_triplets is True: (
                anchor_img, positive_img, negative_img, anchor_label, negative_label
            ).
            The positive label is omitted because it always equals the anchor label.

        """
        # Single-image mode used for validation: just the image and its identity.
        if not self.return_triplets:
            return self.load_image(index), self.identity_labels[index]

        # Triplet mode used for training.
        # Positive and negative images are randomly sampled each time
        # so each epoch has different triplets.
        # The global random module is used here because PyTorch reseeds it separately in
        # every DataLoader worker, so workers don't produce identical triplets.
        anchor_index, positive_index, negative_index = self.sample_triplet(index, random)

        anchor_img = self.load_image(anchor_index)
        positive_img = self.load_image(positive_index)
        negative_img = self.load_image(negative_index)

        # Labels are returned so the miner in train.py can tell which embeddings
        # in a batch belong to the same person.
        anchor_label = self.identity_labels[anchor_index]
        negative_label = self.identity_labels[negative_index]
        return anchor_img, positive_img, negative_img, anchor_label, negative_label
