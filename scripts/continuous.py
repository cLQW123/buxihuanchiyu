"""Run successive full FL2VA cases and preserve every exit, log and duration."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

parser=argparse.ArgumentParser()
parser.add_argument('--model-path',required=True)
parser.add_argument('--first-frame',required=True)
parser.add_argument('--output-dir',required=True)
parser.add_argument('--seeds',default='42,43')
parser.add_argument('--text',default='A dog falls asleep')
args=parser.parse_args()
root=Path(os.environ['SGCUP_ROOT'])
output=Path(args.output_dir).resolve()
output.mkdir(parents=True,exist_ok=True)
records=[]
for index,seed in enumerate(map(int,args.seeds.split(','))):
    folder=output/f'case-{index:02d}-seed-{seed}'
    folder.mkdir(exist_ok=False)
    video=folder/'generated.mp4'
    command=[sys.executable,str(root/'scripts/run.py'),'fl2va','--model-path',args.model_path,
             '--first-frame',args.first_frame,'--text',args.text,'--gpus','0,1,2,3,4,5,6,7',
             '--num-frames','124','--steps','50','--seed',str(seed),'--output',str(video)]
    env=os.environ.copy();env['SGCUP_RUN_EVIDENCE']=str(folder)
    started=time.time()
    with (folder/'generation.log').open('w') as log:
        process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
        (folder/'process.pid').write_text(str(process.pid))
        status=process.wait()
    (folder/'generation.exit').write_text(str(status)+'\n')
    record={'seed':seed,'command':command,'exit':status,'wall_seconds':time.time()-started}
    if video.is_file():
        record['video_sha256']=hashlib.sha256(video.read_bytes()).hexdigest()
        record['video_bytes']=video.stat().st_size
    elif status == 0:
        record['validation_error']='The generator returned success without a video'
        status=record['exit']=1
    records.append(record)
    (output/'cases.json').write_text(json.dumps(records,indent=2))
    print(json.dumps(record),flush=True)
    if status:
        raise SystemExit(status)
print('SUCCESSIVE_FULL_FL2VA_CASES_COMPLETED',len(records),flush=True)
