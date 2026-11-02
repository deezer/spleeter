"""PyTorch-based Spleeter 5-stem separator (4-stem also supported). No TensorFlow required.

Implements the exact same separation pipeline as Spleeter's librosa backend:
  1. STFT (n_fft=4096, hop=1024, periodic Hann window)
  2. Magnitude spectrogram → UNet per instrument → Wiener masks
  3. Masked complex STFT → iSTFT

Models are loaded from .pt files produced by convert_weights.py.
Default configuration: 5-stem (vocals, piano, drums, bass, other).
"""

import os
import sys
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from librosa.core import stft as librosa_stft, istft as librosa_istft
from scipy.signal.windows import hann as scipy_hann

sys.path.insert(0, os.path.dirname(__file__))
from unet import UNet

# Spleeter 4-stems parameters (from 4stems.json)
SAMPLE_RATE    = 44100
FRAME_LENGTH   = 4096    # n_fft
FRAME_STEP     = 1024    # hop_length
T              = 512     # time frames per UNet chunk
F_BINS         = 1024    # frequency bins fed to UNet (first F of n_fft//2+1=2049)
N_CHANNELS     = 2       # stereo
SEP_EXPONENT   = 2       # Wiener mask exponent
EPSILON        = 1e-10
WIN_COMP       = 2.0 / 3.0  # window compensation factor
INSTRUMENTS    = ['vocals', 'piano', 'drums', 'bass', 'other']  # 5-stem default


def _stft(waveform: np.ndarray) -> np.ndarray:
    """Compute STFT of a stereo waveform, matching Spleeter's librosa backend.

    Args:
        waveform: (n_samples, 2) float32 array.

    Returns:
        Complex spectrogram (n_frames, n_fft//2+1, 2) = (n_frames, 2049, 2).
    """
    N = FRAME_LENGTH
    H = FRAME_STEP
    win = scipy_hann(N, sym=False)
    out = []
    for c in range(N_CHANNELS):
        # Prepend and append N zeros (matching Spleeter's padding)
        d = np.concatenate([np.zeros(N, dtype=np.float32),
                            waveform[:, c].astype(np.float32),
                            np.zeros(N, dtype=np.float32)])
        s = librosa_stft(d, n_fft=N, hop_length=H, window=win, center=False)
        # s: (n_fft//2+1, n_frames) complex
        s = s.T  # (n_frames, 2049)
        out.append(s[:, :, np.newaxis])
    return np.concatenate(out, axis=2)  # (n_frames, 2049, 2)


def _istft(spec: np.ndarray, length: int) -> np.ndarray:
    """Inverse STFT of a complex stereo spectrogram.

    Args:
        spec: (n_frames, n_fft//2+1, 2) complex array.
        length: Original waveform length in samples.

    Returns:
        Waveform (length, 2) float32.
    """
    N = FRAME_LENGTH
    H = FRAME_STEP
    win = scipy_hann(N, sym=False)
    out = []
    for c in range(N_CHANNELS):
        s = spec[:, :, c].T  # (2049, n_frames)
        x = librosa_istft(s, hop_length=H, win_length=N, window=win,
                          center=False, length=length + N)
        # Trim pre-pended N zeros
        x = x[N : N + length]
        out.append(x[:, np.newaxis])
    waveform = np.concatenate(out, axis=1)  # (length, 2)
    return (waveform * WIN_COMP).astype(np.float32)


def _pad_and_chunk(spec_mag: np.ndarray) -> np.ndarray:
    """Pad time dimension to multiple of T, then chunk into (n_chunks, T, F, 2).

    Args:
        spec_mag: (n_frames, F_BINS, 2) magnitude spectrogram.

    Returns:
        (n_chunks, T, F_BINS, 2) array.
    """
    n_frames = spec_mag.shape[0]
    pad_len = (T - n_frames % T) % T
    if pad_len:
        spec_mag = np.pad(spec_mag, ((0, pad_len), (0, 0), (0, 0)))
    n_chunks = spec_mag.shape[0] // T
    return spec_mag.reshape(n_chunks, T, F_BINS, N_CHANNELS)


def _run_unets(chunks: np.ndarray, models: Dict[str, UNet],
               device: torch.device) -> Dict[str, np.ndarray]:
    """Run each instrument's UNet on the magnitude chunks.

    Args:
        chunks: (n_chunks, T, F_BINS, 2) magnitude array.
        models: dict instrument → UNet (eval mode).
        device: torch device.

    Returns:
        Dict instrument → (n_chunks, T, F_BINS, 2) output magnitude.
    """
    # PyTorch: (n_chunks, channels, T, F_BINS) = NCHW
    x = torch.from_numpy(chunks).permute(0, 3, 1, 2).to(device)
    outputs = {}
    with torch.no_grad():
        for name, model in models.items():
            y = model(x)  # (n_chunks, 2, T, F_BINS)
            outputs[name] = y.permute(0, 2, 3, 1).cpu().numpy()  # (n_chunks, T, F_BINS, 2)
    return outputs


class SpleeterPyTorch:
    """Spleeter 4-stem separator using PyTorch UNets (no TensorFlow)."""

    def __init__(self, model_dir: str, device: Optional[str] = None):
        """
        Args:
            model_dir: directory containing vocals.pt, drums.pt, bass.pt, other.pt.
            device: 'cpu', 'cuda', or None for auto-detect.
        """
        if device is None:
            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self._device = torch.device(device)

        # Detect which stems are available from .pt files
        available = [n for n in INSTRUMENTS
                     if os.path.exists(os.path.join(model_dir, f"{n}.pt"))]
        if not available:
            raise FileNotFoundError(
                f"No .pt model files found in {model_dir}. "
                "Run convert_weights.py first."
            )
        self._instruments = available

        self._models: Dict[str, UNet] = {}
        for name in self._instruments:
            pt_path = os.path.join(model_dir, f"{name}.pt")
            if not os.path.exists(pt_path):
                raise FileNotFoundError(
                    f"Model file not found: {pt_path}\n"
                    f"Run convert_weights.py first to generate .pt files."
                )
            model = UNet()
            model.load_state_dict(torch.load(pt_path, map_location='cpu'))
            model.eval()
            model.to(self._device)
            self._models[name] = model
        print(f"[Spleeter] Loaded {len(self._models)}-stem UNets ({', '.join(self._instruments)}) on {self._device}")

    def separate(self, waveform: np.ndarray) -> Dict[str, np.ndarray]:
        """Separate a stereo waveform into instrument stems.

        Args:
            waveform: (n_samples, 2) float32. Sample rate must be 44100 Hz.

        Returns:
            Dict instrument → (n_samples, 2) float32 separated waveform.
        """
        assert waveform.ndim == 2 and waveform.shape[1] == 2, \
            "waveform must be (n_samples, 2)"
        n_samples = waveform.shape[0]

        # 1. STFT → complex spec (n_frames, 2049, 2)
        mix_stft = _stft(waveform)
        n_frames = mix_stft.shape[0]

        # 2. Magnitude, slice to F_BINS
        mix_mag = np.abs(mix_stft[:, :F_BINS, :])  # (n_frames, 1024, 2)

        # 3. Pad and chunk
        chunks = _pad_and_chunk(mix_mag)  # (n_chunks, 512, 1024, 2)
        n_chunks = chunks.shape[0]

        # 4. Run UNets
        raw_outputs = _run_unets(chunks, self._models, self._device)
        n_instruments = len(raw_outputs)

        # 5. Wiener masking
        # Compute sum of outputs^SEP_EXPONENT for normalization
        output_sum = np.zeros_like(chunks) + EPSILON
        for out in raw_outputs.values():
            output_sum += out ** SEP_EXPONENT

        separated = {}
        for name, raw_out in raw_outputs.items():
            # Wiener mask (n_chunks, T, F_BINS, 2)
            mask = (raw_out ** SEP_EXPONENT + EPSILON / n_instruments) / output_sum

            # Reshape chunks back to frames: (n_chunks*T, F_BINS, 2)
            mask_frames = mask.reshape(n_chunks * T, F_BINS, N_CHANNELS)

            # Extend mask to full frequency range (n_chunks*T, 2049, 2)
            n_extra = FRAME_LENGTH // 2 + 1 - F_BINS  # = 1025
            mask_full = np.zeros((n_chunks * T, FRAME_LENGTH // 2 + 1, N_CHANNELS),
                                 dtype=np.float32)
            mask_full[:, :F_BINS, :] = mask_frames

            # Trim to actual n_frames
            mask_full = mask_full[:n_frames, :, :]

            # Apply mask to complex STFT
            masked_stft = mix_stft * mask_full.astype(mix_stft.dtype)

            # iSTFT → waveform
            separated[name] = _istft(masked_stft, n_samples)

        return separated
