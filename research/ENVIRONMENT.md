# EchoInsight Research Environment



**Owner:** Utsav

**Last verified:** 2026-10-13

**Platform:** Windows 11, Python 3.12.10



## 1. Hardware and runtime



- GPU: NVIDIA GeForce RTX 4060 Laptop GPU

- NVIDIA driver: 596.49

- PyTorch: 2.14.1+cu130

- PyTorch CUDA runtime: 13.0

- CUDA availability: `True`



The CUDA runtime bundled with PyTorch is distinct from the NVIDIA driver version.



## 2. Verified package versions



| Package | Version |

|---|---|

| Python | 3.12.10 |

| PyTorch | 2.14.1+cu130 |

| torchaudio | 2.11.0+cu130 |

| Demucs | 4.1.0 |

| librosa | 0.11.0 |

| NumPy | 2.5.4 |

| Matplotlib | 3.11.2 |

| PySide6 | 6.12.0 |

| pytest | 9.1.1 |



The complete installed-package snapshot is saved in `research/requirements-lock.txt`.



## 3. Environment setup

Run these commands from the project root in Windows Command Prompt.

Create and activate a fresh virtual environment. The following example uses `.venv-rebuild-test` so the original `.venv` remains untouched.

```bat
py -3.12 -m venv .venv-rebuild-test

.venv-rebuild-test\Scripts\activate.bat
```

Install the exact GPU-enabled PyTorch and TorchAudio versions used in the verified rebuild:

```bat
python -m pip install torch==2.14.1+cu130 torchaudio==2.11.0+cu130 --index-url https://download.pytorch.org/whl/cu130
```

Install the pinned packages from the environment snapshot without replacing their dependencies:

```bat
python -m pip install -r research\requirements-lock.txt --no-deps
```

Check package dependency consistency:

```bat
python -m pip check
```

**Important:** The lock file includes CUDA-specific PyTorch packages. Install those first from the CUDA package index as shown above. The `--no-deps` option prevents pip from trying to retrieve the CUDA-specific versions from the default package index.

## 4. Verification commands



Check the GPU runtime:



```bat

python -c "import torch; print('PyTorch:',torch.__version__); print('CUDA:',torch.cuda.is_available()); print('GPU:',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Not detected')"

```



Check Demucs:



```bat

python -c "import demucs; print(demucs.__file__)"

```



Check dependencies:



```bat

python -m pip check

```



Run the project test suite:



```bat

python -m pytest -v
```



## 5. Verified status

Verified on 2026-10-13 using the isolated `.venv-rebuild-test` environment.

- **Commit tested:** `aca3e172bedf7173e16da7a4e9dc16da01df6ae1`
- Python 3.12.10 was used.
- PyTorch 2.14.1+cu130 and TorchAudio 2.11.0+cu130 imported successfully.
- CUDA was available and the NVIDIA GeForce RTX 4060 Laptop GPU was detected.
- Demucs 4.1.0 and librosa 0.11.0 imported successfully.
- `python -m pip check` reported: `No broken requirements found.`
- Test command: `python -m pytest -v`
- Test result: **159 passed, 6 skipped, 0 failed, 4 warnings in 30.70 seconds.**

The six skipped tests and their reported reasons are:

- `tests\test_bootstrap.py:123` — No MP3 files in `data/raw/`; cannot test extraction.
- `tests\test_bootstrap.py:141` — No MP3 files in `data/raw/`; cannot test extraction.
- `tests\test_extraction_v2.py:225` — Got empty parameter set for `audio_file`.
- `tests\test_extraction_v2.py:280` — No audio fixtures in `tests/fixtures/`.
- `tests\test_frames.py:472` — No MP3 files in `data/raw/`.
- `tests\test_frames.py:552` — No MP3 files in `data/raw/`.

The four warnings did not cause test failures.

## 6. Reproducibility notes and limitations

- The pinned environment was installed successfully in an isolated rebuild environment, and the project test suite completed with no failures.
- The CUDA-enabled PyTorch and TorchAudio packages must be installed from the compatible CUDA package index before installing the lock file with `--no-deps`.
- Reproducibility depends on a compatible Python installation, NVIDIA driver, GPU, package availability and operating system.
- Six tests were skipped because the required local audio files or fixtures are unavailable. Their exact locations and reasons are listed in Section 5.
- This test result applies to commit `aca3e172bedf7173e16da7a4e9dc16da01df6ae1`. It does not verify the newer canonical branch commit `86c84ec`.
- The final work plan expects 296 passed and 2 skipped after cleanup PR #2 is merged. Rerun the tests against the latest `feature/stem-separation` after that merge; do not try to fix skipped tests as part of Step 0.
- Do not assume the existing E003 report has been reproduced on this machine.
- Keep research audio, stems, model checkpoints and generated outputs outside Git.
- Record input-file provenance, hashes, settings, package versions, hardware and runtime for every experiment.
- Do not overwrite `experiments/e003_baseline_report.json` during baseline reproduction.

**Status:** Isolated environment rebuild verified. The test result is documented against the tested commit. Audio-fixture-dependent tests remain unverified.
