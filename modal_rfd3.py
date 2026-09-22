# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "modal>=1.0",
# ]
# ///
"""RFdiffusion3 binder / motif design on Modal.

RFdiffusion3: https://github.com/RosettaCommons/foundry/tree/production/models/rfd3
Input spec: https://github.com/RosettaCommons/foundry/blob/production/models/rfd3/docs/input.md

Pass a local YAML (or JSON) whose `input:` fields point at motif PDB/CIF files.
Those structure files are resolved relative to the YAML and uploaded with it.

Weights live on the Modal Volume `rfd3-weights`. Designs go to Volume `rfd3`.
After each backbone, SolubleMPNN designs sequences with motif/target residues
held fixed from that design's `diffused_index_map`. Download checkpoints once,
then run with --detach:

    uv run --with modal modal run modal_rfd3.py --install-weights

    # or upload a checkpoint you already have:
    uv run --with modal modal run modal_rfd3.py \\
      --upload-weights /path/to/rfd3_latest.ckpt

    GPU=A100 uv run --with modal modal run --detach modal_rfd3.py \\
      --input-yaml pdl1.yaml --n-batches 1 --diffusion-batch-size 8 \\
      --num-mpnn 8

    # later: modal volume get rfd3 <run_name> ./out/rfd3/
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from modal import App, Image, Volume

GPU = os.environ.get("GPU", "L40S")
TIMEOUT = int(os.environ.get("TIMEOUT", 12)) * 60 * 60
print(f"Using GPU {GPU}; TIMEOUT {TIMEOUT}")

WEIGHTS_VOLUME_NAME = os.environ.get("RFD3_WEIGHTS_VOLUME", "rfd3-weights")
WEIGHTS_VOLUME = Volume.from_name(WEIGHTS_VOLUME_NAME, create_if_missing=True)
WEIGHTS_MOUNT = "/rfd3-weights"

OUT_VOLUME_NAME = os.environ.get("RFD3_OUT_VOLUME", "rfd3")
OUT_VOLUME = Volume.from_name(OUT_VOLUME_NAME, create_if_missing=True)
OUT_MOUNT = f"/{OUT_VOLUME_NAME}"

WORK_DIR = "/tmp/rfd3/work"
FOUNDRY_VERSION = "0.2.0"
SOLUBLE_MPNN_NAME = "solublempnn_v_48_020.pt"
SOLUBLE_MPNN_URL = (
    "https://files.ipd.uw.edu/pub/ligandmpnn/solublempnn_v_48_020.pt"
)
MPNN_MAX_BATCH = 8

WEIGHTS_HELP = (
    f"No RFdiffusion3 checkpoint found on volume '{WEIGHTS_VOLUME_NAME}'. "
    "Download once:\n"
    "  uv run --with modal modal run modal_rfd3.py --install-weights\n"
    "Or upload a local checkpoint:\n"
    "  uv run --with modal modal run modal_rfd3.py "
    "--upload-weights /path/to/rfd3_latest.ckpt"
)

# YAML: input: path  or  input: "path"
_YAML_INPUT = re.compile(
    r"^(?P<prefix>[ \t]*input[ \t]*:[ \t]*)"
    r"(?P<q>[\"']?)"
    r"(?P<path>[^\"'#\n]+?)"
    r"(?P=q)"
    r"(?P<suffix>[ \t]*(?:#.*)?)?$",
    re.MULTILINE | re.IGNORECASE,
)
# JSON: "input": "path"
_JSON_INPUT = re.compile(
    r'(?P<prefix>"input"\s*:\s*")(?P<path>[^"]+)(?P<suffix>")',
    re.IGNORECASE,
)

image = (
    Image.debian_slim(python_version="3.12")
    .apt_install(
        "git",
        "wget",
        "ca-certificates",
        "build-essential",
        "cmake",
        "libgomp1",
    )
    # CUDA torch first so rc-foundry does not pull a CPU wheel.
    .run_commands(
        "pip install torch --index-url https://download.pytorch.org/whl/cu126"
    )
    .run_commands(f"pip install 'rc-foundry[rfd3]=={FOUNDRY_VERSION}'")
    .env({"FOUNDRY_CHECKPOINT_DIRS": WEIGHTS_MOUNT})
)

app = App("rfd3", image=image)


def _is_missing_input(path: str) -> bool:
    return path.strip().lower() in {"", "null", "none", "~", "false"}


def _unique_name(path: Path, used: set[str]) -> str:
    name = path.name
    if name not in used:
        used.add(name)
        return name
    stem = f"{path.parent.name}_{path.name}" if path.parent.name else f"dup_{path.name}"
    candidate = stem
    n = 2
    while candidate in used:
        candidate = f"{path.stem}_{n}{path.suffix}"
        n += 1
    used.add(candidate)
    return candidate


def collect_and_rewrite_inputs(
    config_text: str, config_dir: Path
) -> tuple[str, dict[str, bytes]]:
    """Resolve YAML/JSON `input:` PDB/CIF paths and rewrite them for the container.

    Paths are tried relative to the config file, then the current working
    directory. Host paths are replaced with `/tmp/rfd3/work/<filename>` so
    RFD3 can find the uploaded motif files.
    """
    matches = list(_JSON_INPUT.finditer(config_text)) or list(
        _YAML_INPUT.finditer(config_text)
    )
    additional_files: dict[str, bytes] = {}
    path_to_remote: dict[str, str] = {}
    used_names: set[str] = set()
    rewritten = config_text

    for match in matches:
        raw = match.group("path").strip()
        if _is_missing_input(raw):
            continue
        if raw in path_to_remote:
            continue

        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            rel = config_dir / candidate
            cwd = Path.cwd() / candidate
            if rel.exists():
                candidate = rel
            elif cwd.exists():
                candidate = cwd
            else:
                raise FileNotFoundError(
                    f"Structure referenced by input: {raw!r} not found at "
                    f"{rel} (relative to the YAML) or {cwd}"
                )
        elif not candidate.exists():
            raise FileNotFoundError(f"Structure referenced by input: {candidate}")

        remote_name = _unique_name(candidate.resolve(), used_names)
        additional_files[remote_name] = candidate.read_bytes()
        path_to_remote[raw] = f"{WORK_DIR}/{remote_name}"
        print(f"Including referenced structure: {raw} -> {remote_name}")

    # Replace last-to-first so spans stay valid.
    for match in reversed(matches):
        raw = match.group("path").strip()
        if raw not in path_to_remote:
            continue
        groups = match.groupdict()
        quote = groups.get("q") or ""
        suffix = groups.get("suffix") or ""
        rewritten = (
            rewritten[: match.start()]
            + match.group("prefix")
            + quote
            + path_to_remote[raw]
            + quote
            + suffix
            + rewritten[match.end() :]
        )
    return rewritten, additional_files


def _find_checkpoint(weights_dir: Path) -> Path:
    if not weights_dir.exists():
        raise FileNotFoundError(WEIGHTS_HELP)
    ckpts = sorted(p for p in weights_dir.rglob("*.ckpt") if p.is_file())
    if not ckpts:
        raise FileNotFoundError(WEIGHTS_HELP)
    rfd3 = [p for p in ckpts if "rfd3" in p.name.lower()]
    return rfd3[0] if rfd3 else ckpts[0]


def _find_soluble_mpnn(weights_dir: Path) -> Path | None:
    named = weights_dir / SOLUBLE_MPNN_NAME
    if named.is_file():
        return named
    hits = sorted(p for p in weights_dir.rglob("*solublempnn*.pt") if p.is_file())
    return hits[0] if hits else None


def _ensure_soluble_mpnn(weights_dir: Path) -> Path:
    """Return SolubleMPNN weights, downloading onto the volume if needed."""
    import subprocess

    existing = _find_soluble_mpnn(weights_dir)
    if existing is not None:
        return existing
    weights_dir.mkdir(parents=True, exist_ok=True)
    dest = weights_dir / SOLUBLE_MPNN_NAME
    print(f"Downloading SolubleMPNN weights to {dest}")
    subprocess.run(["wget", "-q", "-O", str(dest), SOLUBLE_MPNN_URL], check=True)
    WEIGHTS_VOLUME.commit()
    return dest


def _mpnn_batching(n_sequences: int, max_batch: int = MPNN_MAX_BATCH) -> tuple[int, int]:
    """Return (batch_size, number_of_batches) that yield exactly n_sequences."""
    if n_sequences < 1:
        raise ValueError("n_sequences must be >= 1")
    for batch_size in range(min(max_batch, n_sequences), 0, -1):
        if n_sequences % batch_size == 0:
            return batch_size, n_sequences // batch_size
    return n_sequences, 1


def _fixed_residues_from_index_map(index_map: dict) -> list[str]:
    """Output CIF residue IDs that originated in the RFD3 input (motif + target).

    ``diffused_index_map`` is input residue id → output residue id, e.g.
    ``{"A16": "A4", "B1": "B1"}``. ProteinMPNN sees the output numbering, so
    the values are the positions to keep fixed.
    """
    seen: set[str] = set()
    fixed: list[str] = []
    for out_id in index_map.values():
        if out_id is None:
            continue
        token = str(out_id).strip()
        if not token or token in seen:
            continue
        seen.add(token)
        fixed.append(token)
    return fixed


def _structure_for_json(json_path: Path) -> Path | None:
    stem = json_path.with_suffix("")
    for ext in (".cif.gz", ".cif", ".pdb"):
        candidate = Path(str(stem) + ext)
        if candidate.is_file():
            return candidate
    return None


def _iter_rfd3_designs(out_dir: Path):
    """Yield (json_path, structure_path, fixed_residues) for each RFD3 design."""
    for json_path in sorted(out_dir.rglob("*.json")):
        if "mpnn" in json_path.relative_to(out_dir).parts:
            continue
        try:
            meta = json.loads(json_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(meta, dict) or "diffused_index_map" not in meta:
            continue
        structure = _structure_for_json(json_path)
        if structure is None:
            print(f"Skipping {json_path.name}: no sibling CIF/PDB")
            continue
        index_map = meta.get("diffused_index_map") or {}
        if not isinstance(index_map, dict):
            index_map = {}
        yield json_path, structure, _fixed_residues_from_index_map(index_map)


def _unzip_cif(structure: Path, dest_dir: Path) -> Path:
    import gzip
    import shutil

    if not structure.name.endswith(".gz"):
        return structure
    dest = dest_dir / structure.name[: -len(".gz")]
    dest.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(structure, "rb") as src, dest.open("wb") as out:
        shutil.copyfileobj(src, out)
    return dest


def _run_soluble_mpnn(
    out_dir: Path,
    num_sequences: int,
    temperature: float,
) -> int:
    """Design sequences for every RFD3 CIF using SolubleMPNN.

    Motif and target residues listed as values in ``diffused_index_map`` are
    passed as ``fixed_residues``; remaining (diffused) residues are designed.
    """
    import subprocess

    designs = list(_iter_rfd3_designs(out_dir))
    if not designs:
        print("No RFD3 designs with diffused_index_map found; skipping MPNN")
        return 0

    mpnn_ckpt = _ensure_soluble_mpnn(Path(WEIGHTS_MOUNT))
    batch_size, n_batches = _mpnn_batching(num_sequences)
    mpnn_dir = out_dir / "mpnn"
    unzip_dir = Path(WORK_DIR) / "mpnn_cifs"
    unzip_dir.mkdir(parents=True, exist_ok=True)
    mpnn_dir.mkdir(parents=True, exist_ok=True)

    inputs = []
    for json_path, structure, fixed in designs:
        cif_path = _unzip_cif(structure, unzip_dir)
        entry = {
            "structure_path": str(cif_path),
            "name": json_path.stem,
            "batch_size": batch_size,
            "number_of_batches": n_batches,
            "temperature": temperature,
        }
        if fixed:
            entry["fixed_residues"] = fixed
            print(
                f"MPNN {json_path.stem}: {len(fixed)} fixed residues "
                f"from diffused_index_map, {num_sequences} sequences"
            )
        else:
            print(
                f"MPNN {json_path.stem}: empty diffused_index_map, "
                f"designing all residues, {num_sequences} sequences"
            )
        inputs.append(entry)

    config = {
        "model_type": "protein_mpnn",
        "checkpoint_path": str(mpnn_ckpt),
        "is_legacy_weights": True,
        "out_directory": str(mpnn_dir),
        "write_fasta": True,
        "write_structures": True,
        "inputs": inputs,
    }
    config_path = mpnn_dir / "mpnn_config.json"
    config_path.write_text(json.dumps(config, indent=2))

    cmd = ["mpnn", "--config_json", str(config_path)]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False, cwd=WORK_DIR)
    if result.returncode != 0:
        raise RuntimeError(f"mpnn exited with code {result.returncode}")
    n_out = sum(1 for p in mpnn_dir.rglob("*") if p.is_file())
    print(f"SolubleMPNN wrote {n_out} files under {mpnn_dir}")
    return n_out


@app.function(
    timeout=60 * 60,
    volumes={WEIGHTS_MOUNT: WEIGHTS_VOLUME},
)
def install_rfd3_weights() -> dict:
    """Download RFD3 and SolubleMPNN checkpoints onto the rfd3-weights volume."""
    import subprocess

    Path(WEIGHTS_MOUNT).mkdir(parents=True, exist_ok=True)
    print(f"foundry install rfd3 --checkpoint-dir {WEIGHTS_MOUNT}")
    subprocess.run(
        ["foundry", "install", "rfd3", "--checkpoint-dir", WEIGHTS_MOUNT],
        check=True,
    )
    mpnn_ckpt = _ensure_soluble_mpnn(Path(WEIGHTS_MOUNT))
    WEIGHTS_VOLUME.commit()
    ckpt = _find_checkpoint(Path(WEIGHTS_MOUNT))
    print(f"RFD3 checkpoint: {ckpt} ({ckpt.stat().st_size / 1e9:.2f} GB)")
    print(f"SolubleMPNN:     {mpnn_ckpt} ({mpnn_ckpt.stat().st_size / 1e6:.1f} MB)")
    return {
        "volume": WEIGHTS_VOLUME_NAME,
        "checkpoint": str(ckpt),
        "soluble_mpnn": str(mpnn_ckpt),
    }


@app.function(
    image=image,
    gpu=GPU,
    timeout=TIMEOUT,
    volumes={WEIGHTS_MOUNT: WEIGHTS_VOLUME, OUT_MOUNT: OUT_VOLUME},
)
def rfd3_design(
    yaml_str: str,
    yaml_name: str,
    additional_files: dict[str, bytes],
    design_path: str,
    n_batches: int = 1,
    diffusion_batch_size: int = 8,
    dump_trajectories: bool = False,
    low_memory_mode: bool = False,
    prevalidate_inputs: bool = True,
    extra_args: str | None = None,
    run_mpnn: bool = True,
    num_sequences: int = 8,
    mpnn_temperature: float = 0.1,
) -> dict:
    """Run RFdiffusion3, then SolubleMPNN sequence design on each backbone."""
    import os
    import subprocess
    import threading
    import time

    os.environ["FOUNDRY_CHECKPOINT_DIRS"] = WEIGHTS_MOUNT
    ckpt = _find_checkpoint(Path(WEIGHTS_MOUNT))

    work = Path(WORK_DIR)
    work.mkdir(parents=True, exist_ok=True)
    yaml_path = work / yaml_name
    yaml_path.write_text(yaml_str)
    for rel_path, content in additional_files.items():
        file_path = work / rel_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(content)

    out_dir = Path(design_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / yaml_name).write_text(yaml_str)

    cmd = [
        "rfd3",
        "design",
        f"out_dir={out_dir}",
        f"inputs={yaml_path}",
        f"ckpt_path={ckpt}",
        f"n_batches={n_batches}",
        f"diffusion_batch_size={diffusion_batch_size}",
        f"dump_trajectories={dump_trajectories}",
        f"low_memory_mode={low_memory_mode}",
        f"prevalidate_inputs={prevalidate_inputs}",
        "skip_existing=False",
    ]
    if extra_args:
        cmd.extend(extra_args.split())

    print(f"Checkpoint: {ckpt}")
    print(f"Running: {' '.join(cmd)}")
    started = time.time()
    stop_commit = threading.Event()

    def _commit_loop():
        while not stop_commit.wait(60):
            OUT_VOLUME.commit()

    commit_thread = threading.Thread(target=_commit_loop, daemon=True)
    commit_thread.start()

    n_mpnn = 0
    n_out = 0
    try:
        result = subprocess.run(cmd, check=False, cwd=WORK_DIR)
        if result.returncode != 0:
            raise RuntimeError(f"rfd3 design exited with code {result.returncode}")
        OUT_VOLUME.commit()
        if run_mpnn and num_sequences > 0:
            n_mpnn = _run_soluble_mpnn(
                out_dir, num_sequences, mpnn_temperature
            )
        n_out = sum(1 for p in out_dir.rglob("*") if p.is_file())
    finally:
        stop_commit.set()
        OUT_VOLUME.commit()

    print(f"Results committed to volume '{OUT_VOLUME_NAME}' at {design_path}")
    return {
        "volume": OUT_VOLUME_NAME,
        "design_path": design_path,
        "output_files": n_out,
        "mpnn_files": n_mpnn,
        "elapsed_s": round(time.time() - started),
    }


def _upload_weights(local_path: str) -> None:
    path = Path(local_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)

    files: list[tuple[Path, str]] = []
    if path.is_file():
        files.append((path, f"/{path.name}"))
    else:
        for f in sorted(path.rglob("*")):
            if f.is_file():
                files.append((f, f"/{f.relative_to(path).as_posix()}"))
        if not files:
            raise FileNotFoundError(f"No files in {path}")

    total = sum(src.stat().st_size for src, _ in files)
    print(
        f"Uploading {len(files)} file(s) ({total / 1e9:.2f} GB) "
        f"to Modal Volume '{WEIGHTS_VOLUME_NAME}' ..."
    )
    with WEIGHTS_VOLUME.batch_upload() as batch:
        for src, dest in files:
            print(f"  {src} -> {dest}")
            batch.put_file(src, dest)
    print("Upload complete. You can now run design.")


@app.local_entrypoint()
def main(
    input_yaml: str | None = None,
    n_batches: int = 1,
    diffusion_batch_size: int = 8,
    dump_trajectories: bool = False,
    low_memory_mode: bool = False,
    prevalidate_inputs: bool = True,
    extra_args: str | None = None,
    run_name: str | None = None,
    install_weights: bool = False,
    upload_weights: str | None = None,
    mpnn: bool = True,
    num_mpnn: int | None = None,
    num_sequences: int | None = None,
    mpnn_temperature: float = 0.1,
):
    """Run RFdiffusion3 from a local YAML/JSON spec, then SolubleMPNN.

    PDB/CIF paths in each spec's `input:` field are collected automatically
    (relative to the YAML file) and uploaded with the job. After backbones
    are generated, SolubleMPNN designs `--num-mpnn` sequences per
    structure. Residues listed as values in each design's
    `diffused_index_map` (motif + target) are held fixed.

    Args:
        input_yaml: Path to RFD3 YAML or JSON (motif PDB/CIF via `input:`).
        n_batches: Batches per YAML key (diversity). Default 1.
        diffusion_batch_size: Designs per batch (default 8). Total designs
            are n_batches * diffusion_batch_size per spec.
        dump_trajectories: Also save diffusion trajectories (large).
        low_memory_mode: Memory-efficient tokenization if GPU RAM is tight.
        prevalidate_inputs: Validate the YAML before loading weights.
        extra_args: Extra Hydra CLI args, e.g.
            "inference_sampler.num_timesteps=50 inference_sampler.step_scale=3".
        run_name: Subdirectory on the rfd3 volume (default: timestamp).
        install_weights: Download RFD3 + SolubleMPNN checkpoints onto rfd3-weights.
        upload_weights: Local .ckpt file or directory to put on rfd3-weights.
        mpnn: Run SolubleMPNN after RFD3 (default True). Pass --no-mpnn to skip.
        num_mpnn: SolubleMPNN sequences per backbone (default 8).
        num_sequences: Alias for --num-mpnn.
        mpnn_temperature: SolubleMPNN sampling temperature (default 0.1).
    """
    from datetime import datetime

    if upload_weights:
        _upload_weights(upload_weights)

    if install_weights:
        print(f"Downloading RFD3 + SolubleMPNN weights onto volume '{WEIGHTS_VOLUME_NAME}' ...")
        info = install_rfd3_weights.remote()
        print(f"Weights ready: {info}")

    if not input_yaml:
        if install_weights or upload_weights:
            return
        raise ValueError("Provide --input-yaml, or --install-weights / --upload-weights")

    yaml_path = Path(input_yaml).expanduser().resolve()
    if not yaml_path.exists():
        raise FileNotFoundError(yaml_path)

    yaml_str, additional_files = collect_and_rewrite_inputs(
        yaml_path.read_text(), yaml_path.parent
    )
    if not additional_files:
        print(
            "Warning: no PDB/CIF `input:` files found in the YAML. "
            "That is only valid for fully de novo jobs with no motif."
        )

    if num_mpnn is not None and num_sequences is not None and num_mpnn != num_sequences:
        raise ValueError("Pass only one of --num-mpnn or --num-sequences")
    mpnn_count = (
        num_mpnn if num_mpnn is not None
        else num_sequences if num_sequences is not None
        else 8
    )
    if mpnn_count < 0:
        raise ValueError("--num-mpnn must be >= 0")

    today = datetime.now().strftime("%Y%m%d%H%M")[2:]
    run_subdir = run_name or today
    design_path = f"{OUT_MOUNT}/{run_subdir}"

    print(f"Weights volume: '{WEIGHTS_VOLUME_NAME}' (mounted at {WEIGHTS_MOUNT})")
    print(f"Results will be written to volume '{OUT_VOLUME_NAME}' at {design_path}")
    if mpnn and mpnn_count > 0:
        print(f"SolubleMPNN: {mpnn_count} sequences/backbone, T={mpnn_temperature}")
    else:
        print("SolubleMPNN skipped (--no-mpnn or --num-mpnn 0)")
    print("Run with --detach; you can close the terminal and the job will keep going.")
    print(f"Download later: modal volume get {OUT_VOLUME_NAME} {run_subdir} ./out/rfd3/")
    print(f"List volume:     modal volume ls {OUT_VOLUME_NAME} {run_subdir}")

    # spawn().get() (not bare spawn, not remote) is required for long --detach
    # jobs: bare spawn returns immediately and the ephemeral app shuts down;
    # remote() FunctionCalls expire after 24h.
    result = rfd3_design.spawn(
        yaml_str=yaml_str,
        yaml_name=yaml_path.name,
        additional_files=additional_files,
        design_path=design_path,
        n_batches=n_batches,
        diffusion_batch_size=diffusion_batch_size,
        dump_trajectories=dump_trajectories,
        low_memory_mode=low_memory_mode,
        prevalidate_inputs=prevalidate_inputs,
        extra_args=extra_args,
        run_mpnn=mpnn,
        num_sequences=mpnn_count,
        mpnn_temperature=mpnn_temperature,
    ).get()
    print(f"RFdiffusion3 finished: {result}")
