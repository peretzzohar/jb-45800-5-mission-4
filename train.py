import argparse
import random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


IMAGE_SIZE = 28
NUM_CLASSES = 26
CASE_SENSITIVE_CLASSES = 52
ARCHITECTURE = [IMAGE_SIZE * IMAGE_SIZE, 256, 128, 64, NUM_CLASSES]
PROJECT_ROOT = Path(__file__).resolve().parent


class LetterMLP(nn.Module):
    """A 784 -> 256 -> 128 -> 64 classifier with a configurable output size."""

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()

        architecture = [
            IMAGE_SIZE * IMAGE_SIZE,
            256,
            128,
            64,
            num_classes,
        ]

        self.network = nn.Sequential(
            nn.Linear(architecture[0], architecture[1]),
            nn.ReLU(),
            nn.Linear(architecture[1], architecture[2]),
            nn.ReLU(),
            nn.Linear(architecture[2], architecture[3]),
            nn.ReLU(),
            nn.Linear(architecture[3], architecture[4]),
        )

    def forward(self, x):
        return self.network(x)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train an EMNIST Letters classifier."
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "emnist01",
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "models/emnist_letters_model.pt",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs/training_progress.png",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=25,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--case-sensitive",
        action="store_true",
        help="Train 52 letter classes (A-Z and a-z) from EMNIST ByClass CSV files.",
    )

    return parser.parse_args()


def load_csv(path: Path, case_sensitive: bool = False):
    """Load an EMNIST CSV file: each row is label, pixel_0, ..., pixel_783."""

    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    data = np.loadtxt(
        path,
        delimiter=",",
        dtype=np.float32,
    )

    if data.ndim == 1:
        data = data.reshape(1, -1)

    expected_columns = IMAGE_SIZE * IMAGE_SIZE + 1

    if data.shape[1] != expected_columns:
        raise ValueError(
            f"Expected {expected_columns} columns in {path}, "
            f"found {data.shape[1]}."
        )

    raw_labels = data[:, 0].astype(np.int64)

    if case_sensitive:
        letter_rows = (raw_labels >= 10) & (raw_labels <= 61)

        data = data[letter_rows]
        raw_labels = raw_labels[letter_rows]

        labels = torch.from_numpy(raw_labels - 10)
    else:
        labels = torch.from_numpy(raw_labels - 1)

    images = data[:, 1:].reshape(
        -1,
        IMAGE_SIZE,
        IMAGE_SIZE,
    )

    images = np.transpose(
        images,
        axes=(0, 2, 1),
    )

    pixels = torch.from_numpy(
        images.reshape(
            -1,
            IMAGE_SIZE * IMAGE_SIZE,
        ) / 255.0
    )

    return pixels, labels


def save_training_progress_image(history: dict, output_path: Path):
    """Save a PNG chart of loss/accuracy over the training run."""

    if plt is None:
        return None

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    epochs = list(
        range(
            1,
            len(history["loss"]) + 1,
        )
    )

    fig, (ax_loss, ax_acc) = plt.subplots(
        2,
        1,
        figsize=(6, 6),
        sharex=True,
    )

    ax_loss.plot(
        epochs,
        history["loss"],
        label="train loss",
        color="tab:blue",
    )

    ax_loss.set_ylabel("Loss")
    ax_loss.grid(True, alpha=0.3)
    ax_loss.legend(loc="best")

    ax_acc.plot(
        epochs,
        history["validation_accuracy"],
        label="validation accuracy",
        color="tab:red",
    )

    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_xlabel("Epoch")
    ax_acc.grid(True, alpha=0.3)
    ax_acc.legend(loc="best")

    fig.tight_layout()

    fig.savefig(
        str(output_path),
        dpi=150,
    )

    plt.close(fig)

    return output_path


def main():
    args = parse_args()

    if not 0 < args.validation_fraction < 1:
        raise ValueError(
            "validation-fraction must be between 0 and 1."
        )

    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        raise ValueError(
            "epochs and batch-size must be positive, "
            "and lr must be greater than 0."
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    num_classes = (
        CASE_SENSITIVE_CLASSES
        if args.case_sensitive
        else NUM_CLASSES
    )

    architecture = [
        IMAGE_SIZE * IMAGE_SIZE,
        256,
        128,
        64,
        num_classes,
    ]

    print("Loading data...")

    filename_prefix = (
        "emnist-byclass"
        if args.case_sensitive
        else "emnist-letters"
    )

    X_train, y_train = load_csv(
        args.data_dir / f"{filename_prefix}-train.csv",
        case_sensitive=args.case_sensitive,
    )

    X_test, y_test = load_csv(
        args.data_dir / f"{filename_prefix}-test.csv",
        case_sensitive=args.case_sensitive,
    )

    train_dataset = TensorDataset(
        X_train,
        y_train,
    )

    validation_size = max(
        1,
        round(
            len(train_dataset)
            * args.validation_fraction
        ),
    )

    training_size = (
        len(train_dataset)
        - validation_size
    )

    split_generator = torch.Generator().manual_seed(
        args.seed
    )

    training_dataset, validation_dataset = (
        torch.utils.data.random_split(
            train_dataset,
            [
                training_size,
                validation_size,
            ],
            generator=split_generator,
        )
    )

    train_loader = DataLoader(
        training_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=split_generator,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
    )

    model = LetterMLP(
        num_classes=num_classes
    )

    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
    )

    history = {
        "loss": [],
        "validation_accuracy": [],
    }

    for epoch in range(args.epochs):
        model.train()

        total_loss = 0.0
        total_samples = 0

        for X_batch, y_batch in train_loader:
            logits = model(X_batch)

            loss = loss_fn(
                logits,
                y_batch,
            )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            batch_size = y_batch.size(0)

            total_loss += (
                loss.item()
                * batch_size
            )

            total_samples += batch_size

        model.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for X_batch, y_batch in validation_loader:
                logits = model(X_batch)

                predictions = logits.argmax(
                    dim=1
                )

                correct += (
                    predictions == y_batch
                ).sum().item()

                total += y_batch.size(0)

        avg_loss = (
            total_loss
            / total_samples
        )

        accuracy = (
            correct
            / total
        )

        history["loss"].append(
            avg_loss
        )

        history["validation_accuracy"].append(
            accuracy
        )

        print(
            f"epoch={epoch + 1}, "
            f"loss={avg_loss:.4f}, "
            f"validation_accuracy={accuracy:.4f}"
        )

    model.eval()

    correct = 0
    total = 0

    test_loader = DataLoader(
        TensorDataset(
            X_test,
            y_test,
        ),
        batch_size=args.batch_size,
    )

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            predictions = model(
                X_batch
            ).argmax(dim=1)

            correct += (
                predictions == y_batch
            ).sum().item()

            total += y_batch.size(0)

    print(
        f"final_test_accuracy="
        f"{correct / total:.4f}"
    )

    args.model.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "architecture": architecture,
            "image_size": IMAGE_SIZE,
            "num_classes": num_classes,
            "seed": args.seed,
        },
        args.model,
    )

    print(
        f"Saved model to {args.model}"
    )

    progress_path = save_training_progress_image(
        history,
        args.output,
    )

    if progress_path is not None:
        print(
            f"Saved training progress chart "
            f"to {progress_path}"
        )


if __name__ == "__main__":
    main()