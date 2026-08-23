"""
Audio stem separation engine wrapping HTDemucs for EchoInsight.
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.core.stems import SeparatedTrack, StemPaths

logger = logging.getLogger("echoinsight.separator")


class EchoSeparator:
    """Encapsulates stem separation inference with lazy loading and disk caching."""

    def __init__(
        self,
        model_name: str = "htdemucs",
        device: str | None = None,
        shifts: int = 1,
        overlap: float = 0.25,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.shifts = shifts
        self.overlap = overlap
        self._model = None

    def _load_model(self) -> None:
        """Lazy-loads PyTorch and Demucs only when separation is invoked."""
        if self._model is None:
            import torch
            from demucs.pretrained import get_model

            if self.device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"

            logger.info("Loading separation model: %s on %s", self.model_name, self.device)
            self._model = get_model(self.model_name)
            self._model.to(self.device)
            self._model.eval()

    def separate(
        self,
        audio_path: str | Path,
        output_dir: str | Path,
        track_id: str,
    ) -> SeparatedTrack:
        """Separate an audio file into 4 stems and save them to output_dir."""
        import torch
        import torchaudio
        from demucs.apply import apply_model

        audio_path = Path(audio_path).resolve()
        output_dir = Path(output_dir).resolve() / track_id
        output_dir.mkdir(parents=True, exist_ok=True)

        expected_stems = {
            "vocals": output_dir / "vocals.wav",
            "drums": output_dir / "drums.wav",
            "bass": output_dir / "bass.wav",
            "other": output_dir / "other.wav",
        }

        # Check if already cached on disk
        if all(p.exists() for p in expected_stems.values()):
            logger.info("Using cached stems for track %s at %s", track_id, output_dir)
            mix_info = torchaudio.info(str(audio_path))
            return SeparatedTrack(
                track_id=track_id,
                original_audio=audio_path,
                sample_rate=mix_info.sample_rate,
                duration_seconds=mix_info.num_frames / mix_info.sample_rate,
                stems=StemPaths(**expected_stems),
                model_name=self.model_name,
            )

        self._load_model()

        mix, sr = torchaudio.load(str(audio_path))
        if sr != self._model.samplerate:
            mix = torchaudio.functional.resample(mix, sr, self._model.samplerate)
            sr = self._model.samplerate

        mix_tensor = mix.unsqueeze(0).to(self.device)

        logger.info("Running stem separation inference for %s...", track_id)
        with torch.no_grad():
            estimates = apply_model(
                self._model,
                mix_tensor,
                shifts=self.shifts,
                split=True,
                overlap=self.overlap,
            )

        estimates = estimates.squeeze(0).cpu()
        for idx, source in enumerate(self._model.sources):
            stem_path = expected_stems[source]
            torchaudio.save(str(stem_path), estimates[idx], sr)

        return SeparatedTrack(
            track_id=track_id,
            original_audio=audio_path,
            sample_rate=sr,
            duration_seconds=mix.shape[1] / sr,
            stems=StemPaths(**expected_stems),
            model_name=self.model_name,
        )