# Exact-commit source preparation and offline reproduction

Run the network preparation from the repository root with an exact published
main commit. Use new paths for the checkout and source archive:

```bash
python3 reproduce/prepare_published.py --commit FULL_40_CHARACTER_MAIN_HASH --checkout ../isolated-published-source --archive ../published-source.tar.gz
```

The script retrieves the four pinned source repositories and three nested header
submodules, checks `source-lock.json` and the complete source manifest, then
exports tracked sources. The archive excludes model build libraries, C++ objects,
device binaries, generated videos and previous run evidence. Its JSON sidecar
records the exact commits and archive SHA256.

Transfer the source archive and `reproduce/run_offline.sh` to the competition
filesystem. In an exclusive eight-card shell, activate the competition Python
and DTK environment, stage the declared offline dependencies, then set:

```bash
export SOURCE_ARCHIVE=/path/to/published-source.tar.gz
export SOURCE_ARCHIVE_SHA256=SHA256_FROM_PREPARATION_SIDECAR
export ISOLATED_ROOT=/path/to/new-source-build-directory
export MODEL_PATH=/path/to/competition/MiniMax-H3/diffusers
export FIRST_FRAME=/path/to/competition/first-frame.jpg
export REFERENCE_VIDEO=/path/to/fixed/official-reference.mp4
export SGCUP_DEPENDENCY_PYTHONPATH=/path/to/staged/offline-python-dependencies
export COMPETITION_TRITON_SOURCE=/path/to/unchanged-official-triton-package
export SGCUP_XMAKE=/current/node/path/to/official-xmake
export ISOLATED_BUILD_STATE=/current/node/path/to/new-xmake-state
# Optional: reuse only the validated Triton compiler cache.
export ISOLATED_TRITON_CACHE=/path/to/compiler-cache
bash /path/to/run_offline.sh
```

The runner checks archive and file hashes, prepares links to the official Triton
package, its sibling libraries and matching distribution metadata, and rebuilds
all model extensions and InfiniCore libraries from source. No old `_infinilm.so`,
`_infinicore.so`, dispatch library, AOT wrapper, C++ object, latent or generated
video is copied into the fresh build.

The source environment loads this submission's host-only `sitecustomize.py` for
AFS cross-device file publication. The runner verifies the exact startup module
before building and again before generation. It excludes the old task runtime's
startup patches and model-source overlays.

Generation uses eight cards, 124 frames, 50 configured scheduler points, seed 42
and the unmodified official FL2VA entry through the submitted adapter. The runner
calls the official comparison and checks media fields against the fixed
reference. Stage logs, exit codes, durations and source-build library provenance
are recorded under `outputs/reproduction/`; successful verification writes
`outputs/reproduction/reproduction.json`.
