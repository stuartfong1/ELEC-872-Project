import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from data.face_dataset import FaceDataset
from model.model_registry import ModelRegistry
from pytorch_metric_learning import losses, miners
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def dataset_triplet_indices(
    batch_size: int, device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Get the indices of the dataset triplets within the concatenated batch.

    During training, the anchor, positive, and negative images are
    concatenated into one batch of size 3 * batch_size.
    This function returns the anchor, positive, and negative indices
    which is the format expected by pytorch-metric-learning losses.

    Parameters
    ----------
    batch_size : int
        Number of triplets in the batch (not the concatenated size).
    device : str
        Device on which to create the index tensors.

    Returns
    -------
    anchor_idx : torch.Tensor[batch_size]
        Anchor index
    positive_idx : torch.Tensor[batch_size]
        positive index
    negative_idx : torch.Tensor[batch_size]
        negative index

    """
    # Anchors occupy [0, batch_size), positives [batch_size, 2 * batch_size),
    # and negatives [2 * batch_size, 3 * batch_size), so the i-th triplet is
    # simply (i, i + batch_size, i + 2 * batch_size).
    anchor_indices = torch.arange(batch_size, device=device)
    return anchor_indices, anchor_indices + batch_size, anchor_indices + 2 * batch_size


def train(config: dict) -> None:
    """Train the face recognition model.

    After each epoch, the model is evaluated with nearest-neighbour retrieval on
    the validation split, and a checkpoint is saved if the validation
    precision@1 improves.

    Parameters
    ----------
    config : dict
        Settings from the YAML config file

    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Get the model and its image transform from the registry.
    # The same transform is used for train and val.
    ModelRegistry.set_config(config)
    retrieved_model = ModelRegistry.get_model(config["model_name"])
    model = retrieved_model.model.to(device)
    image_transform = retrieved_model.transform

    # During training, each item is an (anchor, positive, negative) triplet with
    # labels sampled every epoch.
    train_dataset = FaceDataset(
        dataset_dir=config["dataset_dir"],
        split="train",
        transform=image_transform,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
    )

    # During validation, each item is a single image and its label.
    # Validation embeds every image once and evaluates nearest-neighbour
    # retrieval rather than computing a triplet loss.
    val_dataset = FaceDataset(
        dataset_dir=config["dataset_dir"],
        split="val",
        transform=image_transform,
        return_triplets=False,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
    )

    # Triplet loss
    loss_function = losses.TripletMarginLoss(margin=config["triplet_margin"])
    # Semi-hard miner
    # From the current batch, choose negatives that are
    # farther than the positive but still within the margin
    miner = miners.TripletMarginMiner(margin=config["triplet_margin"], type_of_triplets="semihard")
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    # precision@1: nearest-neighbour classification accuracy
    # MAP@R: rewards ranking all of a person's other images near the top
    accuracy_calculator = AccuracyCalculator(
        include=("precision_at_1", "mean_average_precision_at_r"),
        device=torch.device("cpu"),
    )

    checkpoint_dir = Path(config["checkpoint_dir"]) / config["exp_name"]
    checkpoint_dir.mkdir(exist_ok=True)

    best_val_accuracy = 0.0

    for epoch in range(config["num_epochs"]):
        model.train()
        total_loss = 0.0
        # step = 0
        for anchor_img, positive_img, negative_img, anchor_label, negative_label in tqdm(train_dataloader):
            # Concatenate the triplet into one batch so the
            # model uses a single forward pass.
            # Positives use the same label as the anchor.
            images = torch.cat([anchor_img, positive_img, negative_img]).to(device)
            labels = torch.cat([anchor_label, anchor_label, negative_label]).to(device)

            embeddings = model(images)

            # Choose semi-hard triplets using the labels
            if config["hard_negative_mining"]:
                indices_tuple = miner(embeddings, labels)
            else:
                indices_tuple = dataset_triplet_indices(len(anchor_img), device)
            loss = loss_function(embeddings, labels, indices_tuple)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # step += 1
            # if step % 100 == 0 and step > 0:
            #     train_loss = total_loss / step
            #     val_metrics = validate(model, val_dataloader, accuracy_calculator, device)
            #     val_accuracy = val_metrics["precision_at_1"]
            #     print(
            #         f"[Epoch {epoch + 1} step {step}] \ttrain loss: {train_loss:.3f}"
            #         f" \tval precision@1: {val_accuracy:.3f}"
            #         f" \tval MAP@R: {val_metrics['mean_average_precision_at_r']:.3f}",
            #     )

        # Average training loss over the epoch and validation metrics
        train_loss = total_loss / len(train_dataloader)
        val_metrics = validate(model, val_dataloader, accuracy_calculator, device)
        val_accuracy = val_metrics["precision_at_1"]
        print(
            f"[Epoch {epoch + 1}] \ttrain loss: {train_loss:.3f}"
            f" \tval precision@1: {val_accuracy:.3f}"
            f" \tval MAP@R: {val_metrics['mean_average_precision_at_r']:.3f}",
        )

        # Save checkpoints if validation metrics improved
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                checkpoint_dir / f"epoch_{epoch}_acc_{val_accuracy:.3f}.pt",
            )


def validate(
    model: nn.Module,
    val_dataloader: DataLoader,
    accuracy_calculator: AccuracyCalculator,
    device: str,
) -> dict[str, float]:
    """Evaluate the model with nearest-neighbour retrieval on the validation set.

    Parameters
    ----------
    model : nn.Module
        Embedding model to evaluate
    val_dataloader : DataLoader
        Dataloader yielding (image, label) batches from the validation split
    accuracy_calculator : AccuracyCalculator
        pytorch-metric-learning calculator
    device : str
        Device on which to run the model

    Returns
    -------
    dict[str, float]
        Metric names and their values

    """
    torch.cuda.empty_cache()
    model.eval()

    # Embed every validation image once, collecting embeddings and labels.
    val_embeddings = []
    val_labels = []
    with torch.no_grad():
        for images, labels in tqdm(val_dataloader):
            val_embeddings.append(model(images.to(device)))
            val_labels.append(labels.to(device))

    # Normalize to match the distance used by the triplet loss
    val_embeddings = F.normalize(torch.cat(val_embeddings))
    val_labels = torch.cat(val_labels)

    # Perform nearest-neighbour retrieval
    return accuracy_calculator.get_accuracy(val_embeddings, val_labels)


if __name__ == "__main__":
    # Read YAML config
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    # Just in case our dataset is in $SLURM_TMPDIR
    parser.add_argument("--data_dir", type=str)
    args = parser.parse_args()

    with Path.open(args.config) as config_file:
        config = yaml.safe_load(config_file)

    # Override dataset directory in YAML path if specified in arguments
    if args.data_dir is not None:
        config["dataset_dir"] = args.data_dir

    train(config)
