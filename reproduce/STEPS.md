# Final publication and exact-commit reproduction

These commands are prepared only. Do not publish until the complete technical
acceptance evidence has been checked by the root agent. Local preparation commits
do not establish technical acceptance or publication.

## Freeze the final source state

1. Finish any source/script/README edits, regenerate the source manifest, and
   verify it remotely. Update `runtime/delivery-source-commits.json` to reference
   that exact manifest. Source branch HEADs must still match each lock record.
2. Run `python3 runtime/delivery-prepare.py`. It refuses any mismatched source or
   package file, pins the gitlinks, copies scripts/dependency records, and imports
   the three complete source histories into the integration repository locally.
3. Add the actual final generation/comparison/continuous/build/policy receipts,
   logs and example video under `submission/delivery/evidence/` and `examples/`.
   Keep failed experiments clearly identified. Add no weights or fixed data.
4. Update the README only to match verified evidence; if it changes, regenerate
   the manifest and repeat the previous verification/preparation step.

## Commit and publish, after technical acceptance

```bash
git -C submission/delivery add README.md source-lock.json .gitmodules .gitignore Projects scripts dependencies patches evidence examples
git -C submission/delivery diff --cached --check -- . ':(exclude)patches/*.patch'
git -C submission/delivery -c user.name='SGCUP submission' -c user.email='sgcup-submission@users.noreply.github.com' commit -m 'Submit reproducible eight-card MiniMax-H3 FL2VA adaptation'
git -C submission/delivery rev-parse HEAD
git -c http.proxy=http://127.0.0.1:7897 -C submission/delivery push --atomic origin main:main sgcup-infinicore-20261111:sgcup-infinicore-20261111 sgcup-infinilm-20261111:sgcup-infinilm-20261111 sgcup-ninetoothed-20261111:sgcup-ninetoothed-20261111
GIT_TERMINAL_PROMPT=0 git -c credential.helper= -c http.proxy=http://127.0.0.1:7897 ls-remote https://github.com/cLQW123/buxihuanchiyu.git refs/heads/main refs/heads/sgcup-infinicore-20261111 refs/heads/sgcup-infinilm-20261111 refs/heads/sgcup-ninetoothed-20261111
```

The push is atomic and does not force or rewrite public history. Compare all
advertised hashes with local hashes and the source lock. The public repository
was checked as PUBLIC with ADMIN access; anonymous ls-remote proves read access
to the actual published references.

## Network preparation from the exact published main commit

Use a new directory. Replace the hash with the full main commit printed above.

```bash
HTTPS_PROXY=http://127.0.0.1:7897 HTTP_PROXY=http://127.0.0.1:7897 python3 runtime/delivery-isolated-prepare.py --commit MAIN_FULL_40_CHARACTER_HASH --checkout runtime/delivery-isolated-published --archive runtime/delivery-published-source.tar.gz
```

The script fetches main and its pinned four source repositories plus the three
header submodules, verifies the complete source manifest, and exports tracked
sources only. Its JSON sidecar records the tarball SHA256, commits and file count.
No `.git`, model libraries, object files, device binaries, build trees or generated
outputs enter the archive. Transfer this source archive and the run script to the
competition AFS, then use a fresh source directory for offline reproduction.

## Offline build and full generation

Activate the competition Python environment and DTK, stage the fixed offline
dependencies, and set these variables before running the prepared shell script:

```bash
export SOURCE_ARCHIVE=/afs/path/delivery-published-source.tar.gz
export SOURCE_ARCHIVE_SHA256=SHA256_FROM_LOCAL_SIDECAR
export ISOLATED_ROOT=/afs/path/new-published-source-build
export MODEL_PATH=/public/home/aclsylqidf/shared/MiniMax-H3/diffusers
export FIRST_FRAME="$IFAB_AFS_HOME/Golde33443.jpg"
export REFERENCE_VIDEO="$IFAB_AFS_HOME/outputs/full-validation-v157/reference-v157.mp4"
export SGCUP_DEPENDENCY_PYTHONPATH="$SGCUP_NODE_PYTHONPATH:$IFAB_AFS_HOME/python-deps-vendor-flash-v43"
export COMPETITION_TRITON_SOURCE=/afs/path/to/unchanged-official-triton-package
export SGCUP_XMAKE=/current/node/path/to/official-xmake
export ISOLATED_BUILD_STATE=/current/node/path/to/new-xmake-state
export ISOLATED_TRITON_CACHE="$IFAB_AFS_HOME/compiler-cache-v310"
bash /afs/path/delivery-isolated-run.sh
```

The optional Triton cache accelerates compilation only. No old `_infinilm.so`,
`_infinicore.so`, InfiniCore dispatch library, AOT wrapper, C++ object, generated
latent or output video is copied into the fresh build. The script records every
stage exit and duration, checks eight-card source-build library provenance,
executes 124 frames/50 configured steps, calls the unchanged official comparison,
and verifies the actual media against the fixed reference. Only the final actual
`outputs/reproduction/reproduction.json` with `pass: true` proves this reproduction.

The offline runner first calls the submitted `prepare_compiler_dependency.py`.
It exposes only the original official `triton`, `triton.libs` and matching
distribution metadata through three links in this fresh build's
`build/competition-python`. It does not add their old parent `runtime/` directory
to `PYTHONPATH`, and does not copy or modify the official compiler package.

The submitted `scripts/sitecustomize.py` provides host-only cross-device file
publication for the AFS compiler cache. `source scripts/env.sh` puts only this
submission's scripts directory on Python's startup path, before build and run.
The isolated runner verifies the exact startup module and `os.replace` wrapper
before building and again before generation; it does not load the old runtime's
numerical patches.
