"""Download missing trained model weights from Google Drive."""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "app_config.json"
MIN_VALID_SIZE_BYTES = 200_000


def ensure_models_downloaded(config_path=CONFIG_PATH, verbose=True):
    """Ensure all configured model weights are available locally."""

    try:
        import gdown
    except ImportError as exc:
        raise ImportError(
            "gdown is required to download model weights. "
            "Install it using: pip install gdown"
        ) from exc

    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    missing_ids = []

    for model in cfg.get("models", []):
        label = model["label"]
        run_dir = HERE / model["run_dir"]
        weights_path = run_dir / "weights.pt"

        run_dir.mkdir(parents=True, exist_ok=True)

        # Skip existing valid-sized weights.
        if (
            weights_path.exists()
            and weights_path.stat().st_size >= MIN_VALID_SIZE_BYTES
        ):
            if verbose:
                size_mb = weights_path.stat().st_size / 1e6
                print(f"[models] {label}: already present ({size_mb:.1f} MB)")
            continue

        file_id = model.get("gdrive_id")

        if not file_id:
            missing_ids.append(label)
            continue

        if verbose:
            print(f"[models] {label}: downloading weights.pt ...")

        # Remove incomplete files before downloading.
        weights_path.unlink(missing_ok=True)

        try:
            result = gdown.download(
                id=file_id.strip(),
                output=str(weights_path),
                quiet=not verbose
            )

        except Exception as exc:
            weights_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"Failed to download weights for '{label}'. "
                "Check the Google Drive file ID and sharing permissions."
            ) from exc

        # Validate download result.
        if (
            result is None
            or not weights_path.exists()
            or weights_path.stat().st_size < MIN_VALID_SIZE_BYTES
        ):
            weights_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"Download for '{label}' failed or produced an invalid file. "
                "Check Google Drive permissions, file availability, "
                "and virus-scan confirmation handling."
            )

        if verbose:
            size_mb = weights_path.stat().st_size / 1e6
            print(f"[models] {label}: downloaded ({size_mb:.1f} MB)")

    if missing_ids:
        raise RuntimeError(
            "Missing gdrive_id for: " + ", ".join(missing_ids)
            + ". Add a Google Drive file ID for each model in app_config.json."
        )


if __name__ == "__main__":
    ensure_models_downloaded()
    print("All model weights are ready.")
