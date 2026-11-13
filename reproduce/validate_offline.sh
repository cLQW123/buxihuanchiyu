#!/usr/bin/env bash
# Execute from a prepared fresh source checkout on one exclusive eight-card node.
set -euo pipefail
task_repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$task_repo"
: "${MODEL_PATH:?Set the original competition diffusers model path}"
: "${FIRST_FRAME:?Set the unchanged competition first-frame path}"
: "${REFERENCE_VIDEO:?Set the reference video path}"
: "${REFERENCE_PROVENANCE:?Set the reference provenance JSON path}"
source build/activate.sh
task_output="$task_repo/outputs/offline-validation"
mkdir -p "$task_output"
if test -e "$task_output/stages.jsonl"; then
    printf '%s\n' 'Use a fresh output/source directory; previous validation exists.' >&2
    exit 1
fi
export SGCUP_VALIDATION_OUTPUT="$task_output"
python - <<'PY'
import hashlib, json, os
from pathlib import Path
root=Path(os.environ['SGCUP_ROOT']).resolve()
output=Path(os.environ['SGCUP_VALIDATION_OUTPUT'])
reference=Path(os.environ['REFERENCE_VIDEO']).resolve(strict=True)
source=Path(os.environ['REFERENCE_PROVENANCE']).resolve(strict=True)
record=json.loads(source.read_text())
digest=hashlib.file_digest(reference.open('rb'),'sha256').hexdigest()
expected=record.get('video_sha256',record.get('reference_video_sha256'))
assert digest == expected, 'Reference video differs from its provenance record'
assert record.get('kind') in ('organizer-standard', 'compatibility-reference'), 'Explicit reference kind required'
assert record.get('description'), 'Reference source description required'
proof=json.loads((root/'SOURCE_PROVENANCE.json').read_text())
for relative, expected in proof['files'].items():
    assert hashlib.file_digest((root/relative).open('rb'),'sha256').hexdigest()==expected, relative
assert not (Path.home()/'.ninetoothed').resolve() == Path(os.environ['NINETOOTHED_CACHE_DIR']).resolve()
assert Path(os.environ['TRITON_CACHE_DIR']).resolve().is_relative_to(root/'build')
assert Path(os.environ['NINETOOTHED_CACHE_DIR']).resolve().is_relative_to(root/'build')
receipt={'source':str(source),'video':str(reference),'video_sha256':digest,
         'kind':record['kind'],'description':record['description'],
         'formal_reference_accepted':record['kind']=='organizer-standard',
         'provenance':record}
(output/'reference.json').write_text(json.dumps(receipt,indent=2)+'\n')
print('SOURCE_AND_REFERENCE_PROVENANCE_RECORDED',receipt['kind'],len(proof['files']),flush=True)
PY
run_stage() {
    local task_stage=$1
    shift
    python - "$task_stage" "$@" <<'PY'
import json, os, subprocess, sys, time
from pathlib import Path
stage,command=sys.argv[1],sys.argv[2:]
output=Path(os.environ['SGCUP_VALIDATION_OUTPUT'])
started=time.time()
record={'stage':stage,'command':command,'started_epoch':started}
try:
    with (output/(stage+'.log')).open('w') as log:
        process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
        (output/(stage+'.pid')).write_text(str(process.pid)+'\n')
        status=process.wait()
    record['exit']=status
except BaseException as error:
    record['error']=repr(error)
    record['exit']=status=1
finally:
    record['wall_seconds']=time.time()-started
    (output/(stage+'.exit')).write_text(str(record['exit'])+'\n')
    with (output/'stages.jsonl').open('a') as stream:
        stream.write(json.dumps(record)+'\n')
    print('OFFLINE_VALIDATION_STAGE',json.dumps(record),flush=True)
raise SystemExit(status)
PY
}
run_stage environment python scripts/check_environment.py
run_stage build bash scripts/build.sh
run_stage runtime-policy python scripts/validate_runtime_policy.py --output "$task_output/runtime-policy.json"
run_stage continuous python scripts/continuous.py --model-path "$MODEL_PATH" \
    --first-frame "$FIRST_FRAME" --output-dir "$task_output/continuous" --seeds 42,43
task_video="$task_output/continuous/case-00-seed-42/generated.mp4"
run_stage comparison python Projects/InfiniLM/examples/minimax_h3_compare.py --tasks fl2va \
    --model-path "$MODEL_PATH" --first-frame "$FIRST_FRAME" \
    --reference-fl2va-video "$REFERENCE_VIDEO" --infinilm-fl2va-video "$task_video" \
    --output-dir "$task_output/comparison"
run_stage media python scripts/verify_media.py --reference "$REFERENCE_VIDEO" \
    --video "$task_video" --output "$task_output/media.json"
run_stage media-seed43 python scripts/verify_media.py --reference "$REFERENCE_VIDEO" \
    --video "$task_output/continuous/case-01-seed-43/generated.mp4" --output "$task_output/media-seed43.json"
python - <<'PY'
import hashlib, json, math, os, re
from pathlib import Path
output=Path(os.environ['SGCUP_VALIDATION_OUTPUT'])
stages=[json.loads(line) for line in (output/'stages.jsonl').read_text().splitlines()]
assert [item['stage'] for item in stages] == ['environment','build','runtime-policy','continuous','comparison','media','media-seed43']
assert all(item['exit']==0 for item in stages)
cases=json.loads((output/'continuous/cases.json').read_text())
assert [item['seed'] for item in cases]==[42,43] and all(item['exit']==0 for item in cases)
report=(output/'comparison.log').read_text().rsplit('[result] MiniMax-H3 video comparison',1)
assert len(report)==2
assert re.search(r'\bSize:\s+PASS\b',report[1]) and re.search(r'\bLength:\s+PASS\b',report[1])
match=re.search(r'Average SSIM:\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))%',report[1])
assert match and math.isfinite(float(match.group(1)))
media=json.loads((output/'media.json').read_text())
assert media['pass'] and media['actual']==media['reference']
reference=json.loads((output/'reference.json').read_text())
receipt={'technical_validation_pass':True,'reference_kind':reference['kind'],
         'formal_reference_accepted':reference['formal_reference_accepted'],
         'ssim_percent':float(match.group(1)),'cases':cases,'stages':stages,
         'scope':'fresh offline source build and consecutive full FL2VA runs; does not certify organizer acceptance or all operator boundaries'}
(output/'validation.json').write_text(json.dumps(receipt,indent=2)+'\n')
print('ISOLATED_OFFLINE_TECHNICAL_VALIDATION_COMPLETED',json.dumps(receipt),flush=True)
PY
