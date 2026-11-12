"""Download missing trained model weights and figure files from Google Drive."""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "app_config.json"
MIN_VALID_SIZE_BYTES = 200_000
DEFAULT_REPORT_FILE_COUNT = 17


def _import_gdown():
    try:
        import gdown
    except ImportError as exc:
        raise ImportError(
            "gdown is required to download files. "
            "Install it using: pip install gdown"
        ) from exc
    return gdown


def _count_files(directory):
    """Count all files (recursively) inside a directory."""
    if not directory.exists():
        return 0
    return sum(1 for p in directory.rglob("*") if p.is_file())


def ensure_figures_downloaded(config_path=CONFIG_PATH, verbose=True):
    """Ensure the figures folder is available locally in /figures."""

    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    figures_cfg = cfg.get("figures")

    if not figures_cfg or not figures_cfg.get("drive_id"):
        raise RuntimeError(
            "Missing 'figures' section or 'drive_id' in app_config.json."
        )

    label = figures_cfg.get("label", "figures")
    folder_id = figures_cfg["drive_id"].strip()
    figures_dir = HERE / figures_cfg.get("dir", "figures")
    expected = figures_cfg.get("expected_files", DEFAULT_REPORT_FILE_COUNT)

    # Skip if all expected files are already present.
    existing = _count_files(figures_dir)
    if existing >= expected:
        if verbose:
            print(f"[figures] {label}: already present ({existing} files)")
        return

    if verbose:
        print(
            f"[figures] {label}: found {existing}/{expected} files, "
            "downloading ..."
        )

    gdown = _import_gdown()
    figures_dir.mkdir(parents=True, exist_ok=True)

    url = f"https://drive.google.com/drive/folders/{folder_id}"

    try:
        try:
            gdown.download_folder(
                url=url,
                output=str(figures_dir),
                quiet=not verbose,
                remaining_ok=True,
            )
        except TypeError:
            # Older gdown versions don't support some arguments.
            gdown.download_folder(
                url=url,
                output=str(figures_dir),
                quiet=not verbose,
            )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download the '{label}' folder "
            f"({type(exc).__name__}: {exc}). "
            "Check that the ID is a FOLDER id and that sharing is set "
            "to 'Anyone with the link'."
        ) from exc

    downloaded = _count_files(figures_dir)
    if downloaded < expected:
        raise RuntimeError(
            f"Figures download incomplete: expected {expected} files, "
            f"found {downloaded} in '{figures_dir}'. "
            "Check Google Drive permissions and try again."
        )

    if verbose:
        print(f"[figures] {label}: downloaded ({downloaded} files)")

def ensure_models_downloaded(
    config_path=CONFIG_PATH, verbose=True, include_figures=True
):
    """Ensure all configured model weights (and figures) are available locally."""

    gdown = _import_gdown()

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
                quiet=not verbose,
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

    if include_figures:
        ensure_figures_downloaded(config_path=config_path, verbose=verbose)


if __name__ == "__main__":
    ensure_models_downloaded()
    print("All model weights and figures are ready.")