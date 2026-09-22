# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "modal>=1.0",
# ]
# ///
"""Runs BindCraft2 (BC2) binder design on Modal.

Upstream: https://github.com/PacesaLab/BindCraft2

BC2 designs de novo miniproteins, scaffolded binders (VHH, ARP, scFv, Fab),
peptides, and multistate campaigns from one JSON file. Sequence optimisation
uses AlphaFold 2, then ProteinMPNN redesign, then a separate AF2 ensemble
for ranking.

Results are written to the Modal Volume named "bindcraft2". Always use
--detach so you can close the terminal; the job keeps running on Modal:

    GPU=A100 uv run --with modal modal run --detach modal_bindcraft2.py \\
      --input-pdb PDL1.pdb --number-of-final-designs 1

    # shipped PD-L1 target, VHH + humanization
    GPU=A100 uv run --with modal modal run --detach modal_bindcraft2.py \\
      --target hPDL1 --modality VHH --humanize --number-of-final-designs 1

    # native BC2 campaign JSON (local target_path / scaffold files are uploaded)
    GPU=A100 uv run --with modal modal run --detach modal_bindcraft2.py \\
      --input-json design.json

Download results later:

    modal volume get bindcraft2 <run_name> ./out/bindcraft2/
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from modal import App, Image, Volume

GPU = os.environ.get("GPU", "L40S")
TIMEOUT = int(os.environ.get("TIMEOUT", 24)) * 60 * 60
print(f"Using GPU {GPU}; TIMEOUT {TIMEOUT}")

VOLUME_NAME = "bindcraft2"
VOLUME = Volume.from_name(VOLUME_NAME, create_if_missing=True)
VOLUME_MOUNT = f"/{VOLUME_NAME}"

BINDCRAFT2_ROOT = "/opt/bindcraft"
# Pin so Modal rebuilds after upstream BindCraft2 changes.
BINDCRAFT2_COMMIT = "18a9042fbe9a5373a5b7d98fe82335127c2fd70d"
AF2_PARAMS = "/opt/bindcraft-weights/alphafold"
WORK_DIR = "/tmp/bindcraft2"

PATH_KEYS = {"target_path", "binder_scaffold", "scaffold_path"}

image = (
    Image.debian_slim(python_version="3.12")
    .apt_install(
        "git",
        "wget",
        "aria2",
        "ca-certificates",
        "build-essential",
        "python3-dev",
    )
    .run_commands(
        "aria2c -q -x 16 https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar"
        f" && mkdir -p {AF2_PARAMS}"
        f" && tar -xf alphafold_params_2022-12-06.tar -C {AF2_PARAMS}"
        " && rm -f alphafold_params_2022-12-06.tar"
    )
    .run_commands(
        f"git clone https://github.com/PacesaLab/BindCraft2.git {BINDCRAFT2_ROOT}"
        f" && cd {BINDCRAFT2_ROOT} && git checkout {BINDCRAFT2_COMMIT}"
    )
    .env(
        {
            "PIP_BREAK_SYSTEM_PACKAGES": "1",
            "PIP_NO_CACHE_DIR": "1",
            "PYTHONUNBUFFERED": "1",
            "NVIDIA_DRIVER_CAPABILITIES": "compute,utility",
            "BINDCRAFT_AF2_PARAMS": AF2_PARAMS,
        }
    )
    # Editable install: settings/ and scaffolds/ live at the repo root.
    .run_commands(f"python3 -m pip install -e '{BINDCRAFT2_ROOT}[cuda13]'")
    # jax[cuda13] wheels keep CUDA libs under nvidia/*/lib; the loader
    # does not look there unless ldconfig is told.
    .run_commands(
        "python3 -c 'import nvidia, pathlib; "
        "paths=sorted({str(p) for r in nvidia.__path__ for p in pathlib.Path(r).glob(\"*/lib\")}); "
        "open(\"/etc/ld.so.conf.d/bindcraft-cuda.conf\",\"w\").write(chr(10).join(paths)+chr(10))' "
        "&& ldconfig"
    )
    .run_commands("python3 -m bindcraft.selfcheck cuda13")
)

app = App("bindcraft2", image=image)


def _is_path_key(key: str) -> bool:
    return key in PATH_KEYS or key.endswith("_path")


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


def _maybe_upload_path(
    raw: str, json_dir: Path, extra: dict[str, bytes], used: set[str]
) -> str:
    """Rewrite a host path that exists locally; leave repo-relative paths alone."""
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        rel = json_dir / candidate
        cwd = Path.cwd() / candidate
        if rel.exists():
            candidate = rel
        elif cwd.exists():
            candidate = cwd
        else:
            return raw
    elif not candidate.exists():
        raise FileNotFoundError(f"Path referenced in campaign JSON not found: {candidate}")

    remote_name = _unique_name(candidate.resolve(), used)
    extra[remote_name] = candidate.read_bytes()
    remote = f"{WORK_DIR}/{remote_name}"
    print(f"Including referenced file: {raw} -> {remote_name}")
    return remote


def collect_and_rewrite_paths(
    campaign: dict, json_dir: Path
) -> tuple[dict, dict[str, bytes]]:
    """Upload local target/scaffold files and rewrite their paths for the container."""
    extra: dict[str, bytes] = {}
    used: set[str] = set()

    def walk(obj, key: str | None = None):
        if isinstance(obj, dict):
            return {k: walk(v, k) for k, v in obj.items()}
        if isinstance(obj, list):
            return [walk(v, key) for v in obj]
        if isinstance(obj, str) and key and _is_path_key(key):
            return _maybe_upload_path(obj, json_dir, extra, used)
        return obj

    return walk(campaign), extra


def _parse_set_values(set_values: str | None) -> list[str]:
    if not set_values:
        return []
    return [part.strip() for part in set_values.split(";") if part.strip()]


def _parse_lengths(lengths: str) -> list[int]:
    values = [int(part.strip()) for part in lengths.split(",") if part.strip()]
    if not values:
        raise ValueError("binder_lengths must contain at least one integer")
    if len(values) == 1:
        return [values[0], values[0]]
    return [values[0], values[-1]]


def _count_ranked(design_path: str) -> int:
    ranked = Path(design_path) / "3_Ranked"
    if not ranked.is_dir():
        return 0
    csv_path = ranked / "!_Ranked.csv"
    if csv_path.is_file():
        import csv

        with csv_path.open(newline="") as handle:
            return sum(1 for _ in csv.DictReader(handle))
    return sum(
        1
        for path in ranked.iterdir()
        if path.suffix.lower() in {".cif", ".pdb"} and not path.name.startswith(".")
    )


def _add_nvidia_libs() -> None:
    """Ensure jax CUDA wheel libraries are on LD_LIBRARY_PATH."""
    try:
        import nvidia
        from pathlib import Path as P
    except ImportError:
        return
    libs = sorted(
        {str(path) for root in nvidia.__path__ for path in P(root).glob("*/lib")}
    )
    if not libs:
        return
    current = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = ":".join(libs + ([current] if current else []))


@app.function(
    image=image,
    gpu=GPU,
    timeout=TIMEOUT,
    volumes={VOLUME_MOUNT: VOLUME},
)
def bindcraft2(
    campaign_json: str,
    extra_files: dict[str, bytes],
    design_path: str,
    cli_args: list[str],
):
    """Run a BindCraft2 campaign and commit results to the bindcraft2 volume."""
    import subprocess
    import sys
    import threading
    import time

    _add_nvidia_libs()

    import jax

    print(f"BindCraft2 {BINDCRAFT2_COMMIT}")
    print("jax devices:", jax.devices())
    if jax.default_backend() != "gpu":
        raise RuntimeError(
            "JAX is not using a GPU. BindCraft2 cannot run on CPU. "
            f"backend={jax.default_backend()!r} devices={jax.devices()!r}"
        )

    work = Path(WORK_DIR)
    work.mkdir(parents=True, exist_ok=True)
    for name, content in extra_files.items():
        dest = work / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)

    settings_path = work / "campaign.json"
    settings_path.write_text(campaign_json)
    print("campaign", campaign_json)
    print("cli", cli_args)

    Path(design_path).mkdir(parents=True, exist_ok=True)
    started = time.time()
    stop_commit = threading.Event()

    def _commit_loop():
        while not stop_commit.wait(60):
            VOLUME.commit()

    commit_thread = threading.Thread(target=_commit_loop, daemon=True)
    commit_thread.start()

    cmd = [
        sys.executable,
        "-u",
        "-m",
        "bindcraft.cli",
        "design",
        str(settings_path),
        *cli_args,
    ]
    try:
        result = subprocess.run(cmd, cwd=BINDCRAFT2_ROOT, check=False)
        if result.returncode != 0:
            raise RuntimeError(
                f"bindcraft design exited with code {result.returncode}"
            )
        accepted = _count_ranked(design_path)
    finally:
        stop_commit.set()
        VOLUME.commit()

    print(f"Results committed to volume '{VOLUME_NAME}' at {design_path}")
    return {
        "volume": VOLUME_NAME,
        "design_path": design_path,
        "accepted_designs": accepted,
        "elapsed_s": round(time.time() - started),
    }


@app.local_entrypoint()
def main(
    input_pdb: str | None = None,
    input_json: str | None = None,
    target: str | None = None,
    target_chains: str = "A",
    hotspots: str = "",
    modality: str | None = None,
    binder_lengths: str | None = None,
    number_of_final_designs: int | None = None,
    max_trajectories: int | None = None,
    campaign_name: str | None = None,
    run_name: str | None = None,
    core: str | None = None,
    set_values: str | None = None,
    humanize: bool = False,
    forced_targeting: bool = False,
    protease_stable: bool = False,
    disulfide_staple: bool = False,
    mixed_topology: bool = False,
    termini_together: bool = False,
    termini_accessible: bool = False,
    initial_guess: bool = False,
    bigbang: bool = False,
):
    """Local entrypoint to run BindCraft2.

    Uses spawn().get() (required for long detached jobs). Run with
    `modal run --detach` so closing the terminal does not kill the job.
    Results are stored on the Modal Volume named "bindcraft2".

    Provide one of --input-pdb, --input-json, or --target (shipped name
    such as hPDL1). Extra BC2 settings: --set-values 'key=value;key=value'.
    """
    from datetime import datetime

    sources = [flag for flag, value in (
        ("--input-pdb", input_pdb),
        ("--input-json", input_json),
        ("--target", target),
    ) if value]
    if not sources:
        raise ValueError("Provide --input-pdb, --input-json, or --target")
    if len(sources) > 1:
        raise ValueError(f"Provide only one of {', '.join(sources)}")

    extra_files: dict[str, bytes] = {}
    campaign: dict
    name_hint: str

    if input_json:
        json_path = Path(input_json).expanduser().resolve()
        if not json_path.exists():
            raise FileNotFoundError(json_path)
        campaign, extra_files = collect_and_rewrite_paths(
            json.loads(json_path.read_text()), json_path.parent
        )
        name_hint = (
            campaign_name
            or campaign.get("campaign_name")
            or campaign.get("binder_name")
            or json_path.stem
        )
    elif input_pdb:
        pdb_path = Path(input_pdb).expanduser().resolve()
        if not pdb_path.exists():
            raise FileNotFoundError(pdb_path)
        remote_name = pdb_path.name
        extra_files[remote_name] = pdb_path.read_bytes()
        name_hint = campaign_name or pdb_path.stem
        target_entry = {
            "name": name_hint,
            "target_path": f"{WORK_DIR}/{remote_name}",
            "chains": target_chains,
        }
        if hotspots:
            target_entry["hotspots"] = hotspots
        campaign = {
            "targets": [target_entry],
            "modality": modality or "binder",
            "campaign_name": name_hint,
            "number_of_final_designs": number_of_final_designs or 1,
        }
    else:
        name_hint = campaign_name or target
        campaign = {
            "target": target,
            "modality": modality or "binder",
            "campaign_name": name_hint,
            "number_of_final_designs": number_of_final_designs or 1,
        }

    today = datetime.now().strftime("%Y%m%d%H%M")[2:]
    run_subdir = run_name or today
    design_path = f"{VOLUME_MOUNT}/{run_subdir}/{name_hint}"
    campaign["project_folder"] = design_path
    campaign.setdefault("campaign_name", name_hint)

    if input_json:
        if number_of_final_designs is not None:
            campaign["number_of_final_designs"] = number_of_final_designs
        if modality:
            campaign["modality"] = modality

    if binder_lengths:
        campaign["binder_lengths"] = _parse_lengths(binder_lengths)
    if max_trajectories is not None:
        campaign["max_trajectories"] = max_trajectories

    cli_args: list[str] = []
    if core:
        cli_args += ["--core", core]
    if modality and input_json:
        cli_args += ["--modality", modality]
    properties = {
        "humanize": humanize,
        "forced_targeting": forced_targeting,
        "protease_stable": protease_stable,
        "disulfide_staple": disulfide_staple,
        "mixed_topology": mixed_topology,
        "termini_together": termini_together,
        "termini_accessible": termini_accessible,
        "initial_guess": initial_guess,
        "bigbang": bigbang,
    }
    for flag_name, enabled in properties.items():
        if enabled:
            cli_args.append("--" + flag_name.replace("_", "-"))
    for assignment in _parse_set_values(set_values):
        cli_args += ["--set", assignment]
    cli_args += ["--set", f"project_folder={design_path}"]

    print(f"Results will be written to volume '{VOLUME_NAME}' at {design_path}")
    print(
        "Run with --detach; you can close the terminal and the job will keep going."
    )
    print(
        f"Download later: modal volume get {VOLUME_NAME} {run_subdir} ./out/bindcraft2/"
    )
    print(f"List volume:     modal volume ls {VOLUME_NAME} {run_subdir}")
    print("Accepted designs: 3_Ranked/!_Ranked.csv and 3_Ranked/*.cif")

    result = bindcraft2.spawn(
        campaign_json=json.dumps(campaign, indent=2),
        extra_files=extra_files,
        design_path=design_path,
        cli_args=cli_args,
    ).get()
    print(f"BindCraft2 finished: {result}")
