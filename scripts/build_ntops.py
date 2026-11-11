import concurrent.futures
import importlib
import os
import pathlib
import sys

CURRENT_FILE_PATH = pathlib.Path(__file__).resolve()
PROJECTS_DIR_PATH = CURRENT_FILE_PATH.parent.parent.parent

SRC_DIR_PATH = CURRENT_FILE_PATH.parent.parent / "src"
NTOPS_SRC_DIR_PATH = PROJECTS_DIR_PATH / "ntops" / "src"
NINETOOTHED_SRC_DIR_PATH = PROJECTS_DIR_PATH / "ninetoothed" / "src"

for source_dir in (NTOPS_SRC_DIR_PATH, SRC_DIR_PATH, NINETOOTHED_SRC_DIR_PATH):
    if source_dir.is_dir() and str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))

BUILD_DIRECTORY_PATH = importlib.import_module(
    "infiniop.ninetoothed.build"
).BUILD_DIRECTORY_PATH


def _find_and_build_ops():
    ops_path = SRC_DIR_PATH / "infiniop" / "ops"
    requested_ops = {
        name.strip()
        for name in os.getenv("INFINICORE_NINETOOTHED_OPS", "").split(",")
        if name.strip()
    }
    selected_ops = sorted(
        op_dir.name
        for op_dir in ops_path.iterdir()
        if (op_dir / "ninetoothed" / "build.py").is_file()
        and (not requested_ops or op_dir.name in requested_ops)
    )
    missing_ops = requested_ops - set(selected_ops)
    if missing_ops:
        raise ValueError(f"Unknown or non-NineToothed ops: {sorted(missing_ops)}")

    max_workers = int(os.getenv("INFINICORE_NINETOOTHED_WORKERS", "4"))
    if max_workers < 1:
        raise ValueError("INFINICORE_NINETOOTHED_WORKERS must be positive")

    print(f"NineToothed ops: {', '.join(selected_ops)}", flush=True)
    print(f"NineToothed build workers: {max_workers}", flush=True)
    for op_name in selected_ops:
        print(f"Starting NineToothed op: {op_name}", flush=True)

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_build, ops_path / op_name / "ninetoothed"): op_name
            for op_name in selected_ops
        }

        for future in concurrent.futures.as_completed(futures):
            op_name = futures[future]
            future.result()
            print(f"Built NineToothed op: {op_name}", flush=True)


def _build(ninetoothed_path):
    module_path = ninetoothed_path / "build"
    relative_path = module_path.relative_to(SRC_DIR_PATH)
    import_name = ".".join(relative_path.parts)
    module = importlib.import_module(import_name)

    module.build()


if __name__ == "__main__":
    BUILD_DIRECTORY_PATH.mkdir(parents=True, exist_ok=True)

    _find_and_build_ops()
