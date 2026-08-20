import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from train import ARCHITECTURE, CASE_SENSITIVE_CLASSES, IMAGE_SIZE, LetterMLP

LABELS = [chr(ord("A") + i) for i in range(26)]
CASE_SENSITIVE_LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="Predict a handwritten letter.")
    parser.add_argument("image", type=Path, help="Image file containing one handwritten letter.")
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "models/emnist_letters_model.pt")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs/prediction_preview.png")
    parser.add_argument("--show", action="store_true", help="Display the preview window after saving it.")
    return parser.parse_args()


def preprocess_image(path: Path) -> torch.Tensor:
    image = Image.open(path).convert("L")
    pixels = np.array(image, dtype=np.float32)

    if pixels.mean() > 127:
        pixels = 255.0 - pixels

    threshold = pixels.max() * 0.25
    mask = pixels > threshold
    if not mask.any():
        raise ValueError("No visible letter found in the image.")

    ys, xs = np.where(mask)
    cropped = pixels[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]


    cropped_image = Image.fromarray(cropped.astype(np.uint8))
    scale = 20 / max(cropped_image.size)
    new_size = (max(1, round(cropped_image.width * scale)), max(1, round(cropped_image.height * scale)))
    resized = cropped_image.resize(new_size, Image.Resampling.LANCZOS)

    canvas = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), 0)
    offset = ((IMAGE_SIZE - new_size[0]) // 2, (IMAGE_SIZE - new_size[1]) // 2)
    canvas.paste(resized, offset)

    pixels = torch.from_numpy(np.array(canvas, dtype=np.float32)).flatten() / 255.0
    return pixels


def main():
    args = parse_args()

    if not args.image.is_file():
        raise FileNotFoundError(f"Image not found: {args.image}")
    if not args.model.is_file():
        raise FileNotFoundError(f"Model not found: {args.model}")

    checkpoint = torch.load(args.model, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Invalid model checkpoint: missing model_state_dict.")
    architecture = checkpoint.get("architecture")
    supported_architectures = (
        ARCHITECTURE,
        [IMAGE_SIZE * IMAGE_SIZE, 256, 128, 64, CASE_SENSITIVE_CLASSES],
    )
    if architecture not in supported_architectures:
        raise ValueError(
            f"Unsupported model architecture: {architecture}."
        )

    labels = CASE_SENSITIVE_LABELS if architecture[-1] == CASE_SENSITIVE_CLASSES else LABELS
    model = LetterMLP(num_classes=architecture[-1])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    pixels = preprocess_image(args.image)

    with torch.no_grad():
        logits = model(pixels.unsqueeze(0))
        probabilities = torch.softmax(logits, dim=1)[0]
        predicted_index = int(probabilities.argmax().item())
        confidence = probabilities[predicted_index].item()

    predicted_letter = labels[predicted_index]
    print(f"Predicted letter: {predicted_letter} (confidence={confidence:.2%})")

    top_probabilities, top_indices = torch.topk(probabilities, k=3)
    print("Top 3 guesses:")
    for probability, index in zip(top_probabilities.tolist(), top_indices.tolist()):
        print(f"  {labels[index]}: {probability:.2%}")

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(pixels.reshape(IMAGE_SIZE, IMAGE_SIZE), cmap="gray")
    ax.set_title(f"Predicted letter: {predicted_letter}")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    print(f"Saved preview image: {output_path}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
