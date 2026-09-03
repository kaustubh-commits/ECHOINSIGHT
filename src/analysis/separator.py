"""Audio stem separation engine for EchoInsight."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import soundfile as sf

from src.core.stems import SeparatedTrack, StemPaths


class EchoSeparator:
    """Handles 4-stem audio separation using HTDemucs architectures.

    Supports both stock pretrained weights and custom fine-tuned checkpoints
    with automated disk-level caching and lazy PyTorch imports.
    """

    def __init__(
        self,
        model_name: str = "htdemucs",
        model_path: Optional[str | Path] = None,
        device: Optional[str] = None,
        cache_dir: Optional[str | Path] = None,
    ) -> None:
        self.model_name = model_name
        self.model_path = Path(model_path) if model_path else None
        self.device_name = device
        self.cache_dir = Path(cache_dir) if cache_dir else Path(".cache/stems")
        self._model = None
        self._device = None

    @property
    def is_custom_checkpoint(self) -> bool:
        """Indicates whether a custom fine-tuned checkpoint is configured."""
        return self.model_path is not None and self.model_path.is_file()

    def _load_model(self):
        """Lazy load PyTorch, Demucs architecture, and custom weights."""
        if self._model is not None:
            return self._model, self._device

        try:
            import torch
            from demucs.pretrained import get_model
        except ImportError as err:
            raise ImportError(
                "PyTorch and Demucs are required for audio separation. "
                "Install them via 'pip install torch torchaudio demucs'."
            ) from err

        if self.device_name:
            self._device = torch.device(self.device_name)
        else:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        base_bag = get_model(self.model_name)

        if self.is_custom_checkpoint:
            model = base_bag.models[0].to(self._device)
            state_dict = torch.load(str(self.model_path), map_location=self._device)
            model.load_state_dict(state_dict)
            model.eval()
            self._model = model
        else:
            base_bag.to(self._device)
            base_bag.eval()
            self._model = base_bag

        return self._model, self._device

    def separate(
        self,
        audio_path: str | Path,
        output_dir: Optional[str | Path] = None,
        force_recompute: bool = False,
    ) -> SeparatedTrack:
        """Separates an audio file into Drums, Bass, Other, and Vocals.

        Args:
            audio_path: Path to the input audio file (.wav, .mp3, etc.).
            output_dir: Target directory for exported stems. Defaults to cache_dir.
            force_recompute: If True, bypasses disk cache and re-runs inference.

        Returns:
            SeparatedTrack containing validated StemPaths.
        """
        input_file = Path(audio_path)
        if not input_file.is_file():
            raise FileNotFoundError(f"Input audio file not found: {input_file}")

        # Read basic metadata using soundfile without loading raw tensors
        try:
            info = sf.info(str(input_file))
            sample_rate = info.samplerate
            duration_seconds = float(info.duration)
        except Exception:
            sample_rate = 44100
            duration_seconds = 0.0

        track_id = input_file.stem
        target_dir = Path(output_dir) if output_dir else self.cache_dir / track_id
        target_dir.mkdir(parents=True, exist_ok=True)

        expected_stems = {
            "vocals": target_dir / "vocals.wav",
            "drums": target_dir / "drums.wav",
            "bass": target_dir / "bass.wav",
            "other": target_dir / "other.wav",
        }

        model_tag = f"{self.model_name}_custom" if self.is_custom_checkpoint else self.model_name

        # Check disk cache
        if not force_recompute and all(p.is_file() for p in expected_stems.values()):
            stem_paths = StemPaths(**expected_stems)
            return SeparatedTrack(
                track_id=track_id,
                original_audio=input_file,
                sample_rate=sample_rate,
                duration_seconds=duration_seconds,
                stems=stem_paths,
                model_name=model_tag,
            )

        # Lazy load model and dependencies
        model, device = self._load_model()
        import torch
        from demucs.apply import apply_model

        # Load and prepare audio
        data, sr = sf.read(str(input_file), dtype="float32", always_2d=True)
        audio_tensor = torch.from_numpy(data.T).unsqueeze(0).to(device)

        # Execute separation
        with torch.no_grad():
            estimates = apply_model(
                model,
                audio_tensor,
                shifts=1,
                split=True,
                overlap=0.25,
            )

        # Export individual stems
        estimates_np = estimates.squeeze(0).cpu().numpy()  # (4, channels, samples)
        source_names = ["drums", "bass", "other", "vocals"]

        for idx, src_name in enumerate(source_names):
            stem_audio = estimates_np[idx].T  # (samples, channels)
            sf.write(str(expected_stems[src_name]), stem_audio, sr)

        stem_paths = StemPaths(**expected_stems)
        return SeparatedTrack(
            track_id=track_id,
            original_audio=input_file,
            sample_rate=sr,
            duration_seconds=float(len(data) / sr),
            stems=stem_paths,
            model_name=model_tag,
        )

def load_pre_separated(
    audio_path: str | Path,
    stem_dir: Optional[str | Path] = None,
) -> Optional[SeparatedTrack]:
    """Locate a pre-separated 4-stem set for audio_path and return SeparatedTrack.

    Looks first in the directory containing the audio file (so loading
    ``.../heart_peripheral/Mixture.wav`` discovers its sibling stems), then
    falls back to the project-relative ``data/demo_stems/<name>`` directory.
    Stem filenames are matched case-insensitively; both ``others.wav`` and
    ``other.wav`` are accepted for the OTHER stem.

    Returns ``None`` when no complete stem set is found.
    """
    path = Path(audio_path)

    candidates = []
    if stem_dir is not None:
        candidates.append(Path(stem_dir).resolve())
    else:
        candidates.append(path.parent.resolve())
    project_root = Path(__file__).resolve().parents[2]
    name_token = "".join(ch if ch.isalnum() else "_" for ch in path.stem.lower())
    candidates.append(project_root / "data" / "demo_stems" / name_token)

    seen = set()
    for cand in candidates:
        if cand in seen:
            continue
        seen.add(cand)
        if not cand.is_dir():
            continue
        files = {f.name.lower(): f for f in cand.iterdir() if f.is_file()}

        def pick(*names):
            for n in names:
                if n in files:
                    return files[n]
            return None

        vocals = pick("vocals.wav")
        drums = pick("drums.wav")
        bass = pick("bass.wav")
        other = pick("others.wav", "other.wav")
        if vocals is None or drums is None or bass is None or other is None:
            continue

        mixture = pick("mixture.wav")
        info_source = vocals if vocals is not None else mixture
        sample_rate = 44100
        duration = 0.0
        if info_source is not None:
            try:
                info = sf.info(str(info_source))
                sample_rate = int(info.samplerate)
                duration = float(info.duration)
            except Exception:
                pass

        original = path if path.is_file() else (mixture if mixture is not None else vocals)
        return SeparatedTrack(
            track_id=cand.name,
            original_audio=original.resolve(),
            sample_rate=sample_rate,
            duration_seconds=duration,
            stems=StemPaths(vocals=vocals, drums=drums, bass=bass, other=other),
            model_name="pre-separated",
        )
    return None
