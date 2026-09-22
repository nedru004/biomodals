# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "modal>=1.0",
# ]
# ///
"""Runs BindCraft binder design on Modal, using the PyRosetta-free fork.

Fork: https://github.com/nedru004/BindCraft
Upstream: https://github.com/martinpacesa/BindCraft

The fork skips PyRosetta FastRelax and Rosetta interface scores, so trajectories
finish faster. Scoring uses AlphaFold2 confidence plus BioPython interface stats.

Results are written to the Modal Volume named "bindcraft". Always use --detach
so you can close the terminal; the job keeps running on Modal:

    GPU=A100 uv run --with modal modal run --detach modal_bindcraft.py \\
      --input-pdb PDL1.pdb --number-of-final-designs 1

Download results later:

    modal volume get bindcraft <run_name> ./out/bindcraft/
"""

import os
from pathlib import Path

from modal import App, Image, Volume

GPU = os.environ.get("GPU", "L40S")
TIMEOUT = int(os.environ.get("TIMEOUT", 24)) * 60 * 60
print(f"Using GPU {GPU}; TIMEOUT {TIMEOUT}")

VOLUME_NAME = "bindcraft"
VOLUME = Volume.from_name(VOLUME_NAME)
VOLUME_MOUNT = f"/{VOLUME_NAME}"

BINDCRAFT_ROOT = "/root/bindcraft"
# Pin so Modal rebuilds after fork changes (PyRosetta-free + starting_binder_seq).
BINDCRAFT_COMMIT = "aa2e0a8bb0f4dd2051ba00c4e3c5f7c5d3265605"

image = (
    Image.debian_slim(python_version="3.11")
    .apt_install("git", "wget", "aria2", "ffmpeg")
    .uv_pip_install("numpy<2.0")
    .uv_pip_install(
        "pdb-tools==2.4.8",
        "ffmpeg-python==0.2.0",
        "plotly==5.18.0",
        "kaleido==0.2.1",
        "biopython",
        "scipy",
        "pandas",
    )
    .uv_pip_install("git+https://github.com/sokrypton/ColabDesign.git")
    .run_commands(
        f"git clone https://github.com/nedru004/BindCraft.git {BINDCRAFT_ROOT}"
        f" && cd {BINDCRAFT_ROOT} && git checkout {BINDCRAFT_COMMIT}"
        f" && chmod +x {BINDCRAFT_ROOT}/functions/dssp"
    )
    .run_commands(
        "ln -s /usr/local/lib/python3.*/dist-packages/colabdesign colabdesign && mkdir /params"
    )
    .run_commands(
        "aria2c -q -x 16 https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar"
        f" && mkdir -p {BINDCRAFT_ROOT}/params"
        f" && tar -xf alphafold_params_2022-12-06.tar -C {BINDCRAFT_ROOT}/params"
    )
    .uv_pip_install(
        "numpy<2.0",
        "jax[cuda]<0.7.0",  # Pin to avoid 'wraps' removal in JAX 0.7.0
        "matplotlib==3.8.1",  # https://github.com/martinpacesa/BindCraft/issues/4
    )
)

app = App("bindcraft", image=image)


def _advanced_settings_path(
    design_protocol: str,
    interface_protocol: str,
    template_protocol: str,
) -> str:
    if design_protocol == "Default":
        design_protocol_tag = "default_4stage_multimer"
    elif design_protocol == "Beta-sheet":
        design_protocol_tag = "betasheet_4stage_multimer"
    elif design_protocol == "Peptide":
        design_protocol_tag = "peptide_3stage_multimer"
    else:
        raise ValueError("Unsupported design protocol")

    if interface_protocol == "AlphaFold2":
        interface_protocol_tag = ""
    elif interface_protocol == "MPNN":
        interface_protocol_tag = "_mpnn"
    else:
        raise ValueError("Unsupported interface protocol")

    if template_protocol == "Default":
        template_protocol_tag = ""
    elif template_protocol == "Masked":
        template_protocol_tag = "_flexible"
    else:
        raise ValueError("Unsupported template protocol")

    return (
        f"{BINDCRAFT_ROOT}/settings_advanced/"
        + design_protocol_tag
        + interface_protocol_tag
        + template_protocol_tag
        + ".json"
    )


def _filter_settings_path(filter_option: str) -> str:
    if filter_option == "Default":
        return f"{BINDCRAFT_ROOT}/settings_filters/default_filters.json"
    if filter_option == "Peptide":
        return f"{BINDCRAFT_ROOT}/settings_filters/peptide_filters.json"
    if filter_option == "Relaxed":
        return f"{BINDCRAFT_ROOT}/settings_filters/relaxed_filters.json"
    if filter_option == "Peptide_Relaxed":
        return f"{BINDCRAFT_ROOT}/settings_filters/peptide_relaxed_filters.json"
    if filter_option == "None":
        return f"{BINDCRAFT_ROOT}/settings_filters/no_filters.json"
    raise ValueError("Unsupported filter type")


def _rank_accepted_designs(design_path: str) -> int:
    """Rank Accepted PDBs by Average_i_pTM. BindCraft also ranks when the
    target count is reached; this covers max-trajectory early stops."""
    import shutil
    import sys

    import pandas as pd

    sys.path.insert(0, BINDCRAFT_ROOT)
    from functions.generic_utils import (  # type: ignore
        generate_dataframe_labels,
        generate_directories,
    )

    design_paths = generate_directories(design_path)
    _, design_labels, final_labels = generate_dataframe_labels()
    mpnn_csv = os.path.join(design_path, "mpnn_design_stats.csv")
    final_csv = os.path.join(design_path, "final_design_stats.csv")

    accepted_binders = [
        f
        for f in os.listdir(design_paths["Accepted"])
        if f.endswith(".pdb") and not f.startswith(".")
    ]
    for f in os.listdir(design_paths["Accepted/Ranked"]):
        os.remove(os.path.join(design_paths["Accepted/Ranked"], f))

    if not accepted_binders or not os.path.exists(mpnn_csv):
        return len(accepted_binders)

    design_df = pd.read_csv(mpnn_csv)
    design_df = design_df.sort_values("Average_i_pTM", ascending=False)
    final_df = pd.DataFrame(columns=final_labels)

    rank = 1
    for _, row in design_df.iterrows():
        for binder in accepted_binders:
            binder_name, model = binder.rsplit("_model", 1)
            if binder_name == row["Design"]:
                row_data = {
                    "Rank": rank,
                    **{label: row[label] for label in design_labels},
                }
                final_df = pd.concat(
                    [final_df, pd.DataFrame([row_data])], ignore_index=True
                )
                old_path = os.path.join(design_paths["Accepted"], binder)
                new_path = os.path.join(
                    design_paths["Accepted/Ranked"],
                    f"{rank}_{binder_name}_model{model.rsplit('.', 1)[0]}.pdb",
                )
                shutil.copyfile(old_path, new_path)
                rank += 1
                break

    final_df.to_csv(final_csv, index=False)
    return len(accepted_binders)


@app.function(
    image=image,
    gpu=GPU,
    timeout=TIMEOUT,
    volumes={VOLUME_MOUNT: VOLUME},
)
def bindcraft(
    design_path,
    binder_name,
    pdb_str,
    chains,
    target_hotspot_residues,
    lengths,
    number_of_final_designs,
    design_protocol="Default",
    interface_protocol="AlphaFold2",
    template_protocol="Default",
    filter_option="Default",
    max_trajectories: int | None = None,
    starting_binder_seq: str | None = None,
):
    """Executes the BindCraft pipeline to design protein binders against a target structure.

    Args:
        design_path (str): Path for design outputs on the bindcraft Volume.
        binder_name (str): Name for the binder design project.
        pdb_str (str): PDB file content as a string.
        chains (str): Target chain(s) in the PDB.
        target_hotspot_residues (str): Hotspot residues on the target.
        lengths (list[int]): Range of lengths for the binder.
        number_of_final_designs (int): Desired number of final designs.
        design_protocol (str): Design protocol to use (e.g., "Default", "Beta-sheet").
        interface_protocol (str): Interface protocol (e.g., "AlphaFold2", "MPNN").
        template_protocol (str): Template protocol (e.g., "Default", "Masked").
        filter_option (str): Filter settings to apply (e.g., "Default", "Peptide").
        max_trajectories (int | None): Maximum number of design trajectories to run.
        starting_binder_seq (str | None): Optional amino acid sequence to seed
            hallucination instead of a random binder. Binder length is taken from
            this sequence when set.

    Returns:
        dict: Summary with volume name, design path, and accepted design count.
    """
    import json
    import subprocess
    import sys
    import threading
    import time

    starting_pdb = f"/tmp/bindcraft/{binder_name}.pdb"
    Path(starting_pdb).parent.mkdir(parents=True, exist_ok=True)
    Path(starting_pdb).write_text(pdb_str)

    settings = {
        "design_path": design_path,
        "binder_name": binder_name,
        "starting_pdb": starting_pdb,
        "chains": chains,
        "target_hotspot_residues": target_hotspot_residues,
        "lengths": lengths,
        "number_of_final_designs": number_of_final_designs,
    }
    target_settings_path = f"/tmp/bindcraft/{binder_name}.json"
    Path(target_settings_path).write_text(json.dumps(settings, indent=4))

    advanced_settings_path = _advanced_settings_path(
        design_protocol, interface_protocol, template_protocol
    )
    if max_trajectories is not None:
        with open(advanced_settings_path) as f:
            advanced_settings = json.load(f)
        advanced_settings["max_trajectories"] = max_trajectories
        advanced_settings_path = f"/tmp/bindcraft/{binder_name}_advanced.json"
        Path(advanced_settings_path).write_text(
            json.dumps(advanced_settings, indent=4)
        )

    filter_settings_path = _filter_settings_path(filter_option)

    cmd = [
        sys.executable,
        "-u",
        "bindcraft.py",
        "--settings",
        target_settings_path,
        "--filters",
        filter_settings_path,
        "--advanced",
        advanced_settings_path,
    ]
    if starting_binder_seq:
        cmd += ["--starting_binder_seq", starting_binder_seq]
        print(
            f"Seeding hallucination with starting_binder_seq "
            f"({len(starting_binder_seq)} aa)"
        )

    print(f"BindCraft fork {BINDCRAFT_COMMIT} (no PyRosetta)")
    print("target_settings", settings)
    print("advanced", advanced_settings_path)
    print("filters", filter_settings_path)

    Path(design_path).mkdir(parents=True, exist_ok=True)
    started = time.time()
    stop_commit = threading.Event()

    def _commit_loop():
        while not stop_commit.wait(60):
            VOLUME.commit()

    commit_thread = threading.Thread(target=_commit_loop, daemon=True)
    commit_thread.start()

    try:
        result = subprocess.run(
            cmd,
            cwd=BINDCRAFT_ROOT,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"bindcraft.py exited with code {result.returncode}"
            )
        accepted = _rank_accepted_designs(design_path)
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
    input_pdb: str,
    target_chains: str = "A",
    target_hotspot_residues: str = "",
    lengths: str = "50,130",
    number_of_final_designs: int = 1,
    max_trajectories: int | None = None,
    binder_name: str | None = None,
    run_name: str | None = None,
    starting_binder_seq: str | None = None,
):
    """Local entrypoint to run BindCraft binder design.

    Uses spawn().get() (required for long detached jobs). Run with
    `modal run --detach` so closing the terminal does not kill the job.
    Results are stored on the Modal Volume named "bindcraft".

    Args:
        input_pdb (str): Path to the input PDB file.
        target_chains (str, optional): Target chain(s) in the PDB. Defaults to "A".
        target_hotspot_residues (str, optional): Hotspot residues on the target.
            For example "1,2-10" or chain specific "A1-10,B1-20" or entire chains "A".
            If left blank, an appropriate site will be selected by the pipeline.
            Defaults to "".
        lengths (str, optional): Comma-separated string defining the range of lengths for the binder
                                 (e.g., "50,130"). Defaults to "50,130".
        number_of_final_designs (int, optional): Desired number of final designs. Defaults to 1.
        max_trajectories (int | None, optional): Maximum number of design trajectories to run.
                                                 Defaults to None.
        binder_name (str | None, optional): Name for the binder design project. If None, it's derived
                                            from the input PDB filename. Defaults to None.
        run_name (str | None, optional): Optional name for the run subdirectory on the volume.
                                         If None, a timestamp-based name is used. Defaults to None.
        starting_binder_seq (str | None, optional): Amino acid sequence to seed hallucination
            instead of a random binder. When set, binder length is taken from this sequence
            (``--lengths`` is ignored for trajectory length). Defaults to None.

    Returns:
        None
    """
    from datetime import datetime

    today = datetime.now().strftime("%Y%m%d%H%M")[2:]
    run_subdir = run_name or today

    pdb_str = open(input_pdb).read()
    binder_name = binder_name or Path(input_pdb).stem
    design_path = f"{VOLUME_MOUNT}/{run_subdir}/{binder_name}/"
    lengths_list = [int(i) for i in lengths.split(",")]

    print(f"Results will be written to volume '{VOLUME_NAME}' at {design_path}")
    print(
        "Run with --detach; you can close the terminal and the job will keep going."
    )
    print(f"Download later: modal volume get {VOLUME_NAME} {run_subdir} ./out/bindcraft/")
    print(f"List volume:     modal volume ls {VOLUME_NAME} {run_subdir}")

    # spawn().get() (not bare spawn, not remote) is required for long --detach
    # jobs: bare spawn returns immediately and the ephemeral app shuts down;
    # remote() FunctionCalls expire after 24h.
    result = bindcraft.spawn(
        design_path=design_path,
        binder_name=binder_name,
        pdb_str=pdb_str,
        chains=target_chains,
        target_hotspot_residues=target_hotspot_residues,
        lengths=lengths_list,
        number_of_final_designs=number_of_final_designs,
        max_trajectories=max_trajectories,
        starting_binder_seq=starting_binder_seq,
    ).get()
    print(f"BindCraft finished: {result}")
