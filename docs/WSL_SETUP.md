# WSL2 setup for the Essentia feature extraction half

`essentia-tensorflow` ships no Windows wheels, only Linux and macOS, so this
half of Step 9's DSP feature stack runs inside WSL2. The librosa half runs
natively on Windows. See `selector/audio/features_librosa.py`.

## Versions that worked here

| Component | Version |
|---|---|
| WSL distro | Ubuntu 24.04.1 LTS ("noble"), `wsl --install -d Ubuntu` |
| Python | 3.12.3 (system default on Ubuntu 24.04) |
| `essentia-tensorflow` | `2.1b6.dev1389` (`cp312`, manylinux 2.17/2014, x86_64) |
| TensorFlow (pulled in as a dependency) | bundled with the wheel above |

`essentia-tensorflow` publishing a `cp312` wheel at all was not a given,
older releases only shipped as far as `cp39`. If the exact version above is
no longer resolvable, run `pip index versions essentia-tensorflow` inside the
venv and take whatever the newest `cp312`-tagged release is.

## Setup, exactly as run

```bash
wsl --install -d Ubuntu        # if not already installed; reboot if prompted
wsl -d Ubuntu

# Inside Ubuntu: no apt packages were needed; Ubuntu 24.04 ships python3-venv
# by default. (If venv creation fails with "ensurepip is not available",
# `sudo apt-get install -y python3-venv` first.)
python3 -m venv ~/essentia_venv
source ~/essentia_venv/bin/activate
pip install --upgrade pip
pip install essentia-tensorflow
```

Verify the import (TensorFlow's CUDA warnings are expected and harmless on a
CPU-only box, they're informational, not errors):

```bash
python3 -c "import essentia; import essentia.standard as es; print(essentia.__version__)"
```

## Pretrained models

Cached to `data/essentia_models/` (gitignored, these are large binary
files, not source). Downloaded from the official Essentia models page:

| Model | File | Task |
|---|---|---|
| Danceability | `danceability-musicnn-msd-2.pb` | binary danceable/not |
| Mood: happy | `mood_happy-musicnn-msd-2.pb` | binary |
| Mood: sad | `mood_sad-musicnn-msd-2.pb` | binary |
| Mood: aggressive | `mood_aggressive-musicnn-msd-2.pb` | binary |
| Mood: party | `mood_party-musicnn-msd-2.pb` | binary |
| Mood: relaxed | `mood_relaxed-musicnn-msd-2.pb` | binary |

All six are the `musicnn-msd` variant (trained on the Million Song Dataset via
the MusiCNN embedding), pinned at the `-2` model revision published on
`essentia.upf.edu/models`. Each model's companion `.json` (also downloaded,
also cached alongside its `.pb`) matters just as much as the weights: it
gives the `classes` order for that model's softmax output, and that order is
**not consistent across models**, `danceability`'s positive class is index
0 (`"danceable"`) but `mood_party`'s positive class is index 1 (`"party"`).
`features_essentia.py` reads the label name out of the JSON at runtime rather
than hardcoding an index, specifically because of this inconsistency.

`scripts/wsl_fetch_essentia_models.sh` downloads all six with one command,
but run it from a **Windows** shell (`bash scripts/wsl_fetch_essentia_models.sh`
in Git Bash, not through `wsl.exe`). WSL2's own DNS resolver failed to
resolve `essentia.upf.edu` in this environment (`curl: (28) Resolving timed
out`) even though the same hostname resolved and downloaded fine from
Windows-side networking, and even though WSL had no trouble reaching other
hosts. This looks like a WSL2 resolver quirk rather than the host being
genuinely unreachable, Windows `Test-NetConnection` to the same host on
port 443 succeeded throughout. Re-running the script is a no-op for files
already present either way, since the model files land in `data/`, which
both Windows and WSL can see through the same mount.

The plan also calls for a Discogs genre classifier. As of this pass, Essentia's
model page serves the Discogs-EffNet embedding + genre head as a much larger
multi-file model bundle than the single-file mood/danceability classifiers,
and pulling it in wasn't done for the pilot scope, `features_essentia.py`
runs without it and records `genre_top: null` when the model files aren't
present, rather than failing. Add it the same way (download to
`data/essentia_models/`, point `features_essentia.py` at it) when scaling
past the pilot.

## Running it

From Windows, one command drives the whole WSL half:

```powershell
wsl -d Ubuntu -- bash /mnt/d/codeprojects/spotifyProject/scripts/run_essentia.sh
```

`scripts/run_essentia.sh` activates the venv and runs
`selector/audio/features_essentia.py`, which reads `data/audio/` through
`/mnt/d/...` (the Windows drive is already mounted at `/mnt/d`, no setup
needed) and writes `data/features_essentia.parquet`.

The Windows-side merge step (`selector/audio/merge.py`) degrades gracefully
if that file is absent: it sets `has_essentia = False` for every row and
proceeds with the librosa features alone, printing a message rather than
failing, so the pipeline is still runnable on a machine with no WSL at all.

## Gotchas hit along the way

- `wsl -d Ubuntu -- bash -lc "..."` invoked from a Git Bash / MSYS shell on
  Windows silently mangles `/mnt/d/...` paths (MSYS rewrites them as Windows
  paths before they ever reach `wsl.exe`). Two fixes: put the real logic in a
  `.sh` file and pass only its path as the argument (fewer characters to
  mangle), or set `MSYS_NO_PATHCONV=1` for the command.
- `sudo apt-get ...` on a fresh WSL distro image will block forever waiting
  for an interactive password if run non-interactively, Ubuntu 24.04 didn't
  actually need it for this setup (venv module is preinstalled), so the fix
  was simply not calling `apt-get` at all rather than working around sudo.
- **`local_path` values in `data/audio_matches.parquet` use Windows
  backslashes** (they're written by `pathlib` on Windows in `fetch.py`).
  Linux treats `\` as a literal filename character, not a separator, so
  `features_essentia.py` normalises to forward slashes before opening a
  clip. Easy to miss because the failure mode isn't a crash, `MonoLoader`
  just can't find a file with a backslash in its name and every clip
  silently comes back as a decode failure.
- **Reload TensorFlow graphs once, not per clip.** The first version of
  `features_essentia.py` called `es.TensorflowPredictMusiCNN(graphFilename=...)`
  fresh inside the per-clip loop, for all 6 models, on every one of 198
  clips, 1,188 graph loads instead of 6. Each load parses a multi-MB
  protobuf and spins up new TF session state that never gets released
  cleanly inside one long-running process. This ran the WSL2 VM's memory to
  the point that `wsl --shutdown` itself stopped responding, and recovering
  required force-killing the `vmmemWSL` process from Windows (`taskkill /F`,
  needs an elevated shell, it runs as SYSTEM). The fix is architectural, not
  a tuning knob: build every graph once up front, then loop clips against
  the already-built graphs.
- **A bus error (`Bus error (core dumped)`, exit code 135) reading a
  perfectly normal `.parquet` file was a corrupted `pyarrow`/`numpy`
  install in the venv, not a WSL/DrvFs mmap problem.** `pip list` showed
  `pandas None`, broken version metadata, after a rushed
  `pip install pandas pyarrow` on top of the numpy version
  `essentia-tensorflow` had already pulled in. It crashed identically on a
  brand-new local file created and read back in the same process, which is
  what ruled out `/mnt/d` (DrvFs) as the cause, the first, wrong hypothesis
  was that TensorFlow's mmap-based model loading was unreliable over the
  Windows-drive mount, and real effort went into copying files onto WSL's
  native filesystem before noticing the crash didn't need `/mnt/d` involved
  at all. The actual fix was blunter than any of that: delete the venv and
  reinstall `essentia-tensorflow`, `pandas`, and `pyarrow` together, fresh,
  in one `pip install` so the resolver picks mutually compatible versions
  instead of layering installs on top of each other.
