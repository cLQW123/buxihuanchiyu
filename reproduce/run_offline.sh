#!/usr/bin/env bash
# Run inside a fresh exclusive eight-card competition shell, after environment activation.
set -euo pipefail
: "${SOURCE_ARCHIVE:?Set the exact-commit source .tar.gz path}"
: "${SOURCE_ARCHIVE_SHA256:?Set its locally verified SHA256}"
: "${ISOLATED_ROOT:?Set a new, nonexistent source directory}"
: "${MODEL_PATH:?Set the competition MiniMax-H3 diffusers weights path}"
: "${FIRST_FRAME:?Set the competition first-frame path}"
: "${REFERENCE_VIDEO:?Set the fixed official reference video path}"
: "${SGCUP_DEPENDENCY_PYTHONPATH:?Set vendor/offline dependency paths, without previous model-source overlays}"
: "${COMPETITION_TRITON_SOURCE:?Set the unchanged official Triton package directory containing __init__.py}"
: "${SGCUP_XMAKE:?Set the current-node official xmake executable path}"

python - <<'PY'
import hashlib, json, os
from pathlib import Path
import tarfile
archive = Path(os.environ['SOURCE_ARCHIVE'])
root = Path(os.environ['ISOLATED_ROOT'])
assert not root.exists(), 'Isolation requires a new source directory'
assert hashlib.sha256(archive.read_bytes()).hexdigest() == os.environ['SOURCE_ARCHIVE_SHA256']
with tarfile.open(archive) as package:
    for item in package.getmembers():
        path = Path(item.name)
        assert not path.is_absolute() and '..' not in path.parts, item.name
        if item.issym() or item.islnk():
            target = Path(item.linkname)
            assert not target.is_absolute() and '..' not in target.parts, item.name
    root.mkdir(parents=True)
    package.extractall(root)
proof = json.loads((root/'SOURCE_PROVENANCE.json').read_text())
for relative, sha in proof['files'].items():
    assert hashlib.sha256((root/relative).read_bytes()).hexdigest() == sha, relative
print('ISOLATED_PUBLISHED_SOURCE_VERIFIED', proof['integration_commit'], len(proof['files']), flush=True)
PY

export PYTHONPATH="$SGCUP_DEPENDENCY_PYTHONPATH"
export LD_LIBRARY_PATH="/opt/dtk/lib:/opt/dtk/hip/lib:/opt/dtk/cuda/cuda/lib64:/opt/hyhal/lib:/opt/dtk/.hyhal/rocm_smi/lib:/opt/dtk/.hyhal/rocm_smi${SGCUP_VENDOR_LIBRARY_PATH:+:$SGCUP_VENDOR_LIBRARY_PATH}"
unset SGCUP_PREFIX SGCUP_ROOT INFINI_ROOT SGCUP_TP_MATMUL_KERNEL_DIR
export SGCUP_BUILD_STATE="${ISOLATED_BUILD_STATE:-$ISOLATED_ROOT/build}"
# Optional Triton compiler cache only. No old model extension or InfiniCore
# library is copied; all C++ modules, AOT wrappers and dispatch libraries rebuild.
export SGCUP_CACHE_DIR="${ISOLATED_TRITON_CACHE:-$ISOLATED_ROOT/build/cache/triton}"
export SGCUP_DIAGNOSTIC_LATENTS=0
unset SGCUP_TRACE_TIMEOUT
cd "$ISOLATED_ROOT"
mkdir -p outputs/reproduction
export SGCUP_COMPILER_PYTHONPATH="$ISOLATED_ROOT/build/competition-python"
python scripts/prepare_compiler_dependency.py --triton-source "$COMPETITION_TRITON_SOURCE" \
    > outputs/reproduction/compiler-dependency.log 2>&1
source scripts/env.sh
python - <<'PY'
import hashlib, json, os
from pathlib import Path
import sitecustomize
root = Path(os.environ['SGCUP_ROOT']).resolve()
module = Path(sitecustomize.__file__).resolve()
assert module == root/'scripts/sitecustomize.py', module
assert os.replace is sitecustomize._replace
receipt = {'scope':'submitted host-file publication startup shim',
           'path':str(module), 'sha256':hashlib.sha256(module.read_bytes()).hexdigest(),
           'os_replace_wrapper_active':True}
(root/'outputs/reproduction/startup-shim.json').write_text(json.dumps(receipt,indent=2))
print('SUBMITTED_STARTUP_SHIM_ACTIVE',json.dumps(receipt),flush=True)
PY
run_stage() {
    local stage=$1
    shift
    local started status
    started=$(date +%s)
    set +e
    "$@" > "outputs/reproduction/$stage.log" 2>&1
    status=$?
    set -e
    printf '%s\n' "$status" > "outputs/reproduction/$stage.exit"
    python - "$stage" "$status" "$started" <<'PY'
import json, sys, time
from pathlib import Path
stage, status, started = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
Path('outputs/reproduction/' + stage + '-timing.json').write_text(json.dumps({
    'exit': status, 'wall_seconds': time.time()-started}, indent=2))
PY
    return "$status"
}
run_stage environment python scripts/check_environment.py
run_stage build bash scripts/build.sh
source scripts/env.sh
python - <<'PY'
import os
from pathlib import Path
import sitecustomize
assert Path(sitecustomize.__file__).resolve() == Path(os.environ['SGCUP_ROOT']).resolve()/'scripts/sitecustomize.py'
assert os.replace is sitecustomize._replace
print('SUBMITTED_STARTUP_SHIM_STILL_ACTIVE_BEFORE_GENERATION',flush=True)
PY
run_stage runtime-policy python scripts/validate_runtime_policy.py --output outputs/reproduction/runtime-policy.json
export SGCUP_RUN_EVIDENCE="$ISOLATED_ROOT/outputs/reproduction/seed42"
run_stage generation bash scripts/run.sh fl2va --model-path "$MODEL_PATH" --first-frame "$FIRST_FRAME" \
    --text 'A dog falls asleep' --gpus 0,1,2,3,4,5,6,7 --num-frames 124 --steps 50 \
    --seed 42 --output outputs/reproduction/generated.mp4
run_stage comparison python Projects/InfiniLM/examples/minimax_h3_compare.py --tasks fl2va \
    --model-path "$MODEL_PATH" --first-frame "$FIRST_FRAME" \
    --reference-fl2va-video "$REFERENCE_VIDEO" \
    --infinilm-fl2va-video outputs/reproduction/generated.mp4 \
    --output-dir outputs/reproduction/comparison
run_stage media python scripts/verify_media.py --reference "$REFERENCE_VIDEO" \
    --video outputs/reproduction/generated.mp4 --output outputs/reproduction/media.json
python - <<'PY'
import hashlib, json, math, re
from pathlib import Path
folder = Path('outputs/reproduction')
for stage in ('environment','build','runtime-policy','generation','comparison','media'):
    assert (folder/(stage+'.exit')).read_text().strip() == '0', stage
report = (folder/'comparison.log').read_text().rsplit('[result] MiniMax-H3 video comparison',1)
assert len(report)==2 and 'FL2VA' in report[1]
assert re.search(r'\bSize:\s+PASS\b',report[1]) and re.search(r'\bLength:\s+PASS\b',report[1])
ssim = re.search(r'Average SSIM:\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))%',report[1])
assert ssim and math.isfinite(float(ssim.group(1)))
media = json.loads((folder/'media.json').read_text())
assert media['pass'] is True and media['actual']['frames']==124
assert media['actual']==media['reference']
video_sha = hashlib.sha256((folder/'generated.mp4').read_bytes()).hexdigest()
assert media['video_sha256']==video_sha
proof = {'pass': True, 'integration_commit':json.loads(Path('SOURCE_PROVENANCE.json').read_text())['integration_commit'],
         'video_sha256': video_sha, 'ssim_match_percent':float(ssim.group(1)),
         'scope':'fresh published exact-commit offline build and full eight-card FL2VA generation'}
(folder/'reproduction.json').write_text(json.dumps(proof,indent=2))
print('ISOLATED_REPRODUCTION_VERIFIED',json.dumps(proof),flush=True)
PY
printf '%s\n' ISOLATED_EXACT_COMMIT_BUILD_AND_GENERATION_COMPLETED
