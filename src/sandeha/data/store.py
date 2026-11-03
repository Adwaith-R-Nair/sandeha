"""The dataset on disk, and the lock on the test splits.

Layout under the data root:

    manifest.json      Written last. Without it, no split opens.
    normalizer.json    Fitted on train only.
    generation.json    The config and commit a run started with. Guards resuming.
    <split>/fields.npy   float16 (scene, frame, 3, H, W): u, v, density at the neural grid.
    <split>/statics.npz  Per scene, at both grids: obstacle, late obstacle, inflow, and so on.
    <split>/fine.npz     Reference-grid states at the frames listed in the config.
    <split>/scenes/      One finished scene per file, during generation only.

A frame's fields are zero in the solid that was in force on the way into that frame. The solid
in force from frame f to frame f+1 is ``obstacle(i, f)``, which includes a late obstacle once
``f >= late_frame``. So frame ``late_frame`` is the last one made under the old solid, and it may
hold smoke inside the new one. The model sees that combination when a person draws over smoke.
"""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from sandeha.physics.scenes import SPLITS, Scene
from sandeha.physics.solver import SmokeState

TEST_SPLITS = ("test_a", "test_b", "test_c")
SPLIT_FILES = ("fields.npy", "statics.npz", "fine.npz")
REPO = Path(__file__).resolve().parents[3]
DATA_ROOT = REPO / "data"  # The dataset the results come from. Tests use tiny roots elsewhere.
MANIFEST_COPY = REPO / "results" / "data_manifest.json"


class TamperedDataError(ValueError):
    """Data files don't match the checksums in the manifest, or the manifest its tracked copy."""


def sha256_file(path: Path) -> str:
    with open(path, "rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def split_digest(file_hashes: dict[str, str]) -> str:
    """One checksum for a split, covering each file's name and checksum."""
    text = "\n".join(f"{name} {file_hashes[name]}" for name in sorted(file_hashes))
    return hashlib.sha256(text.encode()).hexdigest()


def sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_atomic(path: Path, write) -> None:
    """Call ``write(file)`` on a temporary file, then move it into place in one step."""
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "wb") as file:
        write(file)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)
    sync_directory(path.parent)


def write_json_atomic(path: Path, value: dict) -> None:
    write_atomic(path, lambda file: file.write(json.dumps(value, indent=1).encode() + b"\n"))


class SplitStore:
    """Read-only access to one split. Use ``open_split``, which checks the lock and checksums."""

    def __init__(self, directory: Path, split: str):
        self.split = split
        self.directory = Path(directory)
        self.fields = np.load(self.directory / "fields.npy", mmap_mode="r")
        with np.load(self.directory / "statics.npz") as statics:
            self._statics = {name: statics[name] for name in statics.files}
        self._fine = None
        self.n_scenes, self.n_frames = self.fields.shape[:2]

    def __len__(self) -> int:
        return self.n_scenes

    def _check_scene(self, i: int) -> None:
        if not 0 <= i < self.n_scenes:
            raise IndexError(f"scene {i} of {self.n_scenes} in {self.split}")

    def _build_scene(self, i: int, prefix: str) -> Scene:
        self._check_scene(i)
        late_frame = int(self._statics["late_frame"][i])
        has_late = late_frame >= 0
        return Scene(
            obstacle=self._statics[prefix + "obstacle"][i],
            inflow=self._statics[prefix + "inflow"][i],
            buoyancy=float(self._statics["buoyancy"][i]),
            late_obstacle=self._statics[prefix + "late_obstacle"][i] if has_late else None,
            late_frame=late_frame if has_late else None,
        )

    def scene(self, i: int) -> Scene:
        """The scene at the neural grid. A coarse cell is solid if any fine cell in it is."""
        return self._build_scene(i, "")

    def fine_scene(self, i: int) -> Scene:
        """The scene at the reference grid, as it was sampled."""
        return self._build_scene(i, "fine_")

    def obstacle(self, i: int, frame: int) -> np.ndarray:
        """Neural-grid solid in force from ``frame`` to ``frame + 1``. Shape ``(H, W)``, bool."""
        self._check_scene(i)
        if not 0 <= frame < self.n_frames:
            raise IndexError(f"frame {frame} of {self.n_frames} in {self.split}")
        solid = self._statics["obstacle"][i]
        late_frame = int(self._statics["late_frame"][i])
        if 0 <= late_frame <= frame:
            return solid | self._statics["late_obstacle"][i]
        return solid

    def inflow(self, i: int) -> np.ndarray:
        """Neural-grid inflow rate, the average of the fine cells. Shape ``(H, W)``."""
        self._check_scene(i)
        return self._statics["inflow"][i]

    def buoyancy(self, i: int) -> float:
        self._check_scene(i)
        return float(self._statics["buoyancy"][i])

    def fine_state(self, i: int, frame: int) -> SmokeState:
        """The reference-grid state on the CPU. Raises ``KeyError`` if this frame wasn't stored.

        It is float32 and cell-centred, with no faces, so a solver started from it projects first.
        """
        self._check_scene(i)
        if self._fine is None:
            with np.load(self.directory / "fine.npz") as fine:
                self._fine = {name: fine[name] for name in fine.files}
        stored = self._fine["frames"].tolist()
        if frame not in stored:
            raise KeyError(f"frame {frame} is not stored at the reference grid, only {stored}")
        values = torch.from_numpy(self._fine["states"][i, stored.index(frame)])
        return SmokeState(values[None, :2], values[None, 2:])


def read_manifest(root: Path) -> dict:
    """The manifest under ``root``. The planned dataset's must also match its tracked copy.

    Checksums can't see a split replaced together with its manifest, because the new manifest
    vouches for the new file. The copy in git can. Other roots have no tracked copy to compare.
    """
    root = Path(root)
    path = root / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing, so generation has not finished")
    manifest = json.loads(path.read_text())
    if root.resolve() == Path(DATA_ROOT).resolve():
        if not Path(MANIFEST_COPY).exists():
            raise TamperedDataError(f"{MANIFEST_COPY}, the manifest's tracked copy, is missing")
        if json.loads(Path(MANIFEST_COPY).read_text()) != manifest:
            raise TamperedDataError(f"{path} does not match its tracked copy {MANIFEST_COPY}")
    return manifest


def open_split(root: Path, split: str, final_eval: bool = False) -> SplitStore:
    """Open a split for reading. Test splits need ``final_eval=True``.

    The lock is checked first, so a refused split isn't read at all. Every file's checksum is then
    compared with the manifest, which takes a few seconds on the largest split. See
    ``read_manifest`` for the manifest's own check.
    """
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}, expected one of {SPLITS}")
    if split in TEST_SPLITS and not final_eval:
        raise PermissionError(
            f"{split} is a locked test split, read only by the final evaluation (final_eval=True)"
        )
    root = Path(root)
    expected = read_manifest(root)["splits"][split]["files"]
    for name, checksum in expected.items():
        path = root / split / name
        if not path.exists():
            raise TamperedDataError(f"{path} is missing")
        if sha256_file(path) != checksum:
            raise TamperedDataError(f"{path} does not match its checksum in the manifest")
    return SplitStore(root / split, split)
