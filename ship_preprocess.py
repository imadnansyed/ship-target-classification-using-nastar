mport json
from pathlib import Path

import cv2
import numpy as np

SUPPORTED_EXTENSIONS = (".tif", ".tiff", ".png", ".jpg", ".jpeg")
TRAIN_PATCH_SIZE = 512      # NovaSAR patches in NASTaR are 512 x 512 pixels
MIN_SIDE = 128


class InputError(Exception):
    """The input cannot be classified. The message is shown to the user."""


class SARPreprocessor:
    LOG_LO, LOG_HI = float(np.log1p(10.0)), float(np.log1p(10000.0))
    BG_SCALE = 6.0

    def __init__(self, target_size, crop_mode, intensity_mode, input_size, norm_mean, norm_std):
        self.target_size, self.crop_mode, self.intensity_mode = int(target_size), crop_mode, intensity_mode
        self.input_size = int(input_size)
        self.norm_mean, self.norm_std = float(norm_mean), float(norm_std)

    @classmethod
    def from_card(cls, card):
        pp = card["preprocessing"]
        return cls(pp["target_size"], pp["crop_mode"], pp["intensity_mode"], pp["input_size"],
                   pp["norm_mean"], pp["norm_std"])

    # ---- the same steps as Section 05 ----
    def resize(self, img):
        s = self.target_size
        if self.crop_mode == "resize":
            return cv2.resize(img, (s, s), interpolation=cv2.INTER_AREA)
        h, w = img.shape
        t, l = (h - s) // 2, (w - s) // 2
        return img[t:t + s, l:l + s].copy()

    def scale_intensity(self, img):
        y = np.log1p(img)
        if self.intensity_mode == "log":
            y = (y - self.LOG_LO) / (self.LOG_HI - self.LOG_LO)
            return np.clip(y, 0.0, 1.0).astype(np.float32)
        if self.intensity_mode == "percentile":
            lo, hi = np.percentile(y, (1.0, 99.5))
            y = (y - lo) / (hi - lo + 1e-6)
            return np.clip(y, 0.0, 1.0).astype(np.float32)
        y = (y - np.median(y)) / self.BG_SCALE                      # "bgnorm"
        return np.clip(y, -0.5, 1.0).astype(np.float32)

    def preprocess(self, raw_img):
        return self.scale_intensity(self.resize(np.asarray(raw_img, dtype=np.float32)))

    def _centre(self, x):
        h, s = x.shape[-1], self.input_size
        if s >= h:
            return x
        i = (h - s) // 2
        return x[..., i:i + s, i:i + s]

    def to_model_input(self, raw_img):
        """Same numerical path as training: float16 cache -> float32 -> normalise -> centre crop.
        Returns an array of shape (1, 1, input_size, input_size)."""
        p = self.preprocess(raw_img).astype(np.float16).astype(np.float32)
        x = self._centre((p - self.norm_mean) / self.norm_std)
        return np.ascontiguousarray(x[None, None]).astype(np.float32)

    # ---- display helpers (what the model sees) ----
    def model_view(self, raw_img):
        y = np.log1p(self.resize(np.asarray(raw_img, dtype=np.float32)))
        y = self._centre(y)
        lo, hi = np.percentile(y, (1.0, 99.5))
        return np.clip((y - lo) / (hi - lo + 1e-6), 0.0, 1.0).astype(np.float32)

    def model_view_uint8(self, raw_img):
        return (255 * self.model_view(raw_img)).astype(np.uint8)


def load_input_stats(path):
    """Brightness ranges of the training patches (written by Section 13), or None."""
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return None


def read_image(path):
    """Read a TIFF, PNG or JPG file. Returns (2-D float32 array, info dict, notes list)."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise InputError(f"Unsupported file format '{ext or 'unknown'}'. Please upload a TIFF, PNG or JPG image.")
    if not path.exists():
        raise InputError("The file could not be found.")
    try:
        if ext in (".tif", ".tiff"):
            import tifffile
            arr = tifffile.imread(str(path))
        else:
            arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if arr is None:
                raise ValueError("unreadable")
    except Exception as exc:
        raise InputError("The image file could not be read (it may be corrupted).") from exc
    notes = []
    arr = np.squeeze(arr)
    orig_dtype = arr.dtype
    if arr.ndim == 3:
        if arr.shape[-1] in (3, 4):
            arr = arr[..., :3].astype(np.float32).mean(axis=-1)
            notes.append("Colour image converted to grayscale.")
        else:
            arr = arr[0]
            notes.append("Multi-page TIFF: only the first page is used.")
    if arr.ndim != 2:
        raise InputError("Expected a single-channel (grayscale) image.")
    info = {"dtype": str(orig_dtype), "shape": tuple(int(s) for s in arr.shape)}
    return arr.astype(np.float32), info, notes


def validate_image(arr, info, stats=None):
    """Check a 2-D array. Returns (512 x 512 float32 array, list of warnings). Raises InputError if unusable."""
    warnings = []
    if not np.isfinite(arr).all():
        raise InputError("The image contains invalid values (NaN or infinity).")
    h, w = arr.shape
    if min(h, w) < MIN_SIDE:
        raise InputError(f"The image is too small ({h} x {w} pixels). SAR patches should be {TRAIN_PATCH_SIZE} x {TRAIN_PATCH_SIZE}.")
    if float(arr.max()) == float(arr.min()):
        raise InputError("The image is blank (all pixels have the same value).")
    if (h, w) != (TRAIN_PATCH_SIZE, TRAIN_PATCH_SIZE):
        interp = cv2.INTER_AREA if min(h, w) >= TRAIN_PATCH_SIZE else cv2.INTER_LINEAR
        arr = cv2.resize(arr, (TRAIN_PATCH_SIZE, TRAIN_PATCH_SIZE), interpolation=interp)
        warnings.append(f"Image resized from {h} x {w} to {TRAIN_PATCH_SIZE} x {TRAIN_PATCH_SIZE}. "
                        "The model was trained on 512 x 512 patches at 2.5 m per pixel, so results may be less reliable.")
    if info.get("dtype") == "uint8":
        warnings.append("This is an 8-bit image. The model was trained on 16-bit NovaSAR intensity patches, "
                        "so its brightness scale differs and the result may be unreliable.")
    if stats:
        med = float(np.median(arr[::4, ::4]))
        lo, hi = stats["median_range"]
        if med < lo * 0.5 or med > hi * 2.0:
            warnings.append(f"Brightness (median {med:.0f}) is outside the range of the training patches "
                            f"({lo:.0f} to {hi:.0f}). This may not be a NovaSAR intensity patch.")
    return arr.astype(np.float32), warnings
