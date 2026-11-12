"""Downloads trained model weights from Google Drive into models/<run>/weights.pt.

Run once before `python app.py` (it is also called automatically at app startup),
whenever a weights.pt file is missing or looks incomplete.
"""
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "app_config.json"
MIN_VALID_SIZE_BYTES = 200_000   # a real weights.pt is at least a few hundred KB; smaller = failed/HTML page


def _extract_drive_id(url_or_id: str) -> str:
    """Accept a full Google Drive share link or a bare file ID; return the file ID."""
    s = url_or_id.strip()
    for pattern in (r"/file/d/([a-zA-Z0-9_-]{20,})",
                    r"[?&]id=([a-zA-Z0-9_-]{20,})",
                    r"^([a-zA-Z0-9_-]{20,})$"):
        m = re.search(pattern, s)
        if m:
            return m.group(1)
    raise ValueError(f"Could not find a Google Drive file ID in: {url_or_id!r}")


def ensure_models_downloaded(config_path=CONFIG_PATH, verbose=True):
    """Download any missing/incomplete weights.pt listed in app_config.json's models[].gdrive_url."""
    try:
        import gdown
    except ImportError as exc:
        raise ImportError("gdown is required to download model weights. "
                          "Install it with: pip install gdown") from exc

    cfg = json.loads(Path(config_path).read_text())
    missing_links = []
    for m in cfg.get("models", []):
        run_dir = HERE / m["run_dir"]
        weights_path = run_dir / "weights.pt"
        run_dir.mkdir(parents=True, exist_ok=True)

        if weights_path.exists() and weights_path.stat().st_size >= MIN_VALID_SIZE_BYTES:
            if verbose:
                print(f"[models] {m['label']}: already present ({weights_path.stat().st_size / 1e6:.1f} MB)")
            continue

        link = m.get("gdrive_url")
        if not link:
            missing_links.append(m["label"])
            continue

        if verbose:
            print(f"[models] {m['label']}: downloading weights.pt ...")
        file_id = _extract_drive_id(link)
        try:
            gdown.download(id=file_id, output=str(weights_path), quiet=not verbose, fuzzy=True)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download weights for '{m['label']}' from {link}. "
                "Check that the Drive file is shared as 'Anyone with the link'."
            ) from exc

        if not weights_path.exists() or weights_path.stat().st_size < MIN_VALID_SIZE_BYTES:
            weights_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"Download for '{m['label']}' looks wrong (file too small or missing). "
                "This usually means the Drive link is not shared as 'Anyone with the link', "
                "or Drive returned a virus-scan warning page instead of the file."
            )
        if verbose:
            print(f"[models] {m['label']}: downloaded ({weights_path.stat().st_size / 1e6:.1f} MB)")

    if missing_links:
        raise RuntimeError(
            "No gdrive_url set for: " + ", ".join(missing_links) +
            ". Add a Google Drive share link for each model in app_config.json before running the app."
        )


if __name__ == "__main__":
    ensure_models_downloaded()
    print("All model weights are ready.")
