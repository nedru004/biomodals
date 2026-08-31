# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "modal>=1.0",
# ]
# ///
"""Runs Protein Hunter (Boltz edition) binder design on Modal.

Protein Hunter: https://github.com/yehlincho/Protein-Hunter
Paper: https://www.biorxiv.org/content/10.1101/2025.10.10.681530

Designs de novo protein binders with Boltz-2 hallucination + LigandMPNN
sequence design. AlphaFold3 cross-validation is not included.

Results are written to the Modal Volume named "proteinhunter". Create it
once, then always use --detach so you can close the terminal:

    modal volume create proteinhunter

    GPU=A100 uv run --with modal modal run --detach modal_proteinhunter.py \\
      --protein-seqs AFTVTVPKDLYVVEYGSNMTIECKFPVEKQLDLAALIVYWEMEDKNIIQFVHGEEDLKVQHSSYRQRARLLKDQLSLGNAALQITDVKLQDAGVYRCMISYGGADYKRITVKVNAPYAAALE \\
      --num-designs 1

    # or from a PDB, like BindCraft:
    GPU=A100 uv run --with modal modal run --detach modal_proteinhunter.py \\
      --input-pdb PDL1.pdb --target-chains A --num-designs 1

Download results later:

    modal volume get proteinhunter <run_name> ./out/proteinhunter/

~7-10 min per design on an H100; expect ~20-25 GB VRAM.
"""

import os
from pathlib import Path

from modal import App, Image, Volume

GPU = os.environ.get("GPU", "L40S")
TIMEOUT = int(os.environ.get("TIMEOUT", 12)) * 60 * 60
print(f"Using GPU {GPU}; TIMEOUT {TIMEOUT}")

VOLUME_NAME = "proteinhunter"
VOLUME = Volume.from_name(VOLUME_NAME)
VOLUME_MOUNT = f"/{VOLUME_NAME}"

PH_ROOT = "/root/Protein-Hunter"

AA3TO1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "MSE": "M",
}


def download_boltz_weights():
    """Download Boltz-2 checkpoints and CCD library into ~/.boltz."""
    from pathlib import Path as P

    from boltz.main import download_boltz2

    cache = P("/root/.boltz")
    cache.mkdir(parents=True, exist_ok=True)
    download_boltz2(cache)
    print(f"Boltz-2 weights downloaded to {cache}")


image = (
    Image.debian_slim(python_version="3.11")
    .apt_install("git", "wget", "gcc", "g++", "build-essential")
    .run_commands(
        f"git clone https://github.com/yehlincho/Protein-Hunter.git {PH_ROOT}"
    )
    # CUDA torch first so boltz_ph does not pull a CPU wheel
    .run_commands(
        "pip install torch --index-url https://download.pytorch.org/whl/cu124"
    )
    .run_commands(f"cd {PH_ROOT}/boltz_ph && pip install -e .")
    .uv_pip_install(
        "matplotlib",
        "seaborn",
        "prody",
        "tqdm",
        "pyyaml",
        "requests",
        "pypdb",
        "py3Dmol",
        "py2Dmol",
        "logmd==0.1.45",
        "ml_collections",
    )
    .run_function(download_boltz_weights)
    .run_commands(
        f"mkdir -p {PH_ROOT}/LigandMPNN/model_params"
        f" && cd {PH_ROOT}/LigandMPNN/model_params"
        " && wget -q https://files.ipd.uw.edu/pub/ligandmpnn/solublempnn_v_48_020.pt"
        " && wget -q https://files.ipd.uw.edu/pub/ligandmpnn/ligandmpnn_v_32_010_25.pt"
        " && wget -q https://files.ipd.uw.edu/pub/ligandmpnn/proteinmpnn_v_48_020.pt"
        f" && chmod +x {PH_ROOT}/utils/DAlphaBall.gcc"
    )
)

app = App("proteinhunter", image=image)


def _chain_ids(chains: str) -> list[str]:
    """Parse BindCraft-style chain selectors ('A', 'A,B', 'AB')."""
    if "," in chains:
        return [c.strip() for c in chains.split(",") if c.strip()]
    if ":" in chains:
        return [c.strip() for c in chains.split(":") if c.strip()]
    return [c for c in chains.replace(" ", "") if c]


def pdb_chains_to_seqs(pdb_str: str, chains: str) -> str:
    """Extract one-letter sequences for the given chains, colon-separated.

    Uses CA atoms so each residue is counted once. MSE is mapped to M.
    """
    wanted = _chain_ids(chains)
    seqs: dict[str, list[str]] = {c: [] for c in wanted}
    seen: dict[str, set[int]] = {c: set() for c in wanted}
    for line in pdb_str.splitlines():
        if not line.startswith("ATOM") and not (
            line.startswith("HETATM") and len(line) >= 26 and line[17:20] == "MSE"
        ):
            continue
        if len(line) < 26:
            continue
        chain = line[21]
        if chain not in seqs:
            continue
        atom = line[12:16].strip()
        if atom != "CA":
            continue
        try:
            resnum = int(line[22:26])
        except ValueError:
            continue
        if resnum in seen[chain]:
            continue
        aa = AA3TO1.get(line[17:20].strip())
        if aa is None:
            continue
        seen[chain].add(resnum)
        seqs[chain].append(aa)
    missing = [c for c in wanted if not seqs[c]]
    if missing:
        raise ValueError(f"No protein sequence found for chain(s): {', '.join(missing)}")
    return ":".join("".join(seqs[c]) for c in wanted)


def _patch_colabfold_msa() -> None:
    """Work around flaky / migrated ColabFold MSA downloads.

    api.colabfold.com often reports COMPLETE then returns JSON 404 for
    GET /result/download/{id}. Their gateway rewrites that to
    /compute/v1/msa/result/download/{id}, which is not a registered route.
    Try alternate paths and hosts, then fall back to a single-sequence MSA.
    """
    import shutil
    import tarfile
    import time
    from urllib.parse import urlparse

    import requests

    orig_get = requests.get
    download_suffixes = (
        "/result/download/{id}",
        "/api/result/download/{id}",
        "/ticket/{id}/download",
    )
    extra_hosts = ("https://api-105.colabfold.com",)

    def get_with_long_download(*args, **kwargs):
        url = args[0] if args else kwargs.get("url", "")
        if not (isinstance(url, str) and "download" in url):
            return orig_get(*args, **kwargs)

        extra = dict(kwargs)
        extra.pop("url", None)
        extra["timeout"] = (6.02, 600)
        rest = args[1:] if args else ()
        job_id = url.rstrip("/").rsplit("/", 1)[-1]
        parsed = urlparse(url)
        bases = [f"{parsed.scheme}://{parsed.netloc}"]
        for host in extra_hosts:
            if host not in bases:
                bases.append(host)

        last_preview = b""
        last_status = None
        for base in bases:
            for suffix in download_suffixes:
                candidate = f"{base}{suffix.format(id=job_id)}"
                print(f"MSA download trying {candidate}")
                response = orig_get(candidate, *rest, **extra)
                payload = response.content or b""
                if len(payload) >= 100 and payload[:2] == b"\x1f\x8b":
                    print(f"MSA download ok ({len(payload)} bytes)")
                    return response
                last_status = response.status_code
                last_preview = payload[:200]
                print(
                    f"  not gzip (status={last_status} "
                    f"len={len(payload)} preview={last_preview!r})"
                )
        raise requests.RequestException(
            f"MSA download 404/invalid for job {job_id} "
            f"(status={last_status} preview={last_preview!r})"
        )

    requests.get = get_with_long_download

    import model_utils
    import pipeline as pipeline_mod

    orig_process_msa = pipeline_mod.process_msa

    def process_msa_robust(chain_id, sequence, msa_dir):
        last_err = None
        for attempt in range(1, 4):
            try:
                return orig_process_msa(chain_id, sequence, msa_dir)
            except (tarfile.ReadError, Exception) as err:
                last_err = err
                print(f"WARNING: ColabFold MSA attempt {attempt}/3 failed: {err}")
                for leftover in Path(msa_dir).glob(f"{chain_id}*"):
                    if leftover.is_dir():
                        shutil.rmtree(leftover, ignore_errors=True)
                    else:
                        leftover.unlink(missing_ok=True)
                time.sleep(5 * attempt)
        print(
            "WARNING: ColabFold MSA download is a JSON 404 from their API "
            f"gateway ({last_err}). Continuing with single-sequence MSA "
            "(equivalent to --msa-mode single)."
        )
        return "empty"

    pipeline_mod.process_msa = process_msa_robust
    model_utils.process_msa = process_msa_robust
    print(
        "ColabFold MSA: try alternate download routes/hosts, then single-seq fallback"
    )


@app.function(
    image=image,
    gpu=GPU,
    timeout=TIMEOUT,
    volumes={VOLUME_MOUNT: VOLUME},
)
def proteinhunter(
    save_dir: str,
    name: str,
    protein_seqs: str = "",
    num_designs: int = 1,
    num_cycles: int = 5,
    min_protein_length: int = 90,
    max_protein_length: int = 150,
    percent_x: int = 100,
    msa_mode: str = "mmseqs",
    high_iptm_threshold: float = 0.7,
    contact_residues: str = "",
    cyclic: bool = False,
    alanine_bias: bool = False,
    ligand_ccd: str = "",
    ligand_smiles: str = "",
    nucleic_seq: str = "",
    nucleic_type: str = "dna",
    seq: str = "",
    omit_aa: str = "C",
    plot: bool = True,
    template_path: str = "",
    template_cif_chain_id: str = "",
) -> dict:
    """Run Protein Hunter binder design and persist outputs on the volume.

    Args:
        save_dir: Absolute path on the proteinhunter Volume for this run.
        name: Design job name (used in output filenames).
        protein_seqs: Target protein sequence(s); colon-separated for multimers.
        num_designs: Number of independent design trajectories.
        num_cycles: Hallucination / MPNN cycles per trajectory.
        min_protein_length: Minimum designed binder length.
        max_protein_length: Maximum designed binder length.
        percent_x: Percent unknown (X) residues in the initial binder sequence.
        msa_mode: "mmseqs" (ColabFold server) or "single" (no MSA).
        high_iptm_threshold: ipTM cutoff for the high-confidence subset.
        contact_residues: Target residues that must contact the binder
            (e.g. "2,3,10"; multi-chain as "1,2 | 3,5").
        cyclic: Design a cyclic peptide binder.
        alanine_bias: Discourage alanine during sequence design.
        ligand_ccd: Optional CCD ligand code (e.g. "SAM").
        ligand_smiles: Optional ligand SMILES.
        nucleic_seq: Optional DNA/RNA target sequence.
        nucleic_type: "dna" or "rna".
        seq: Existing binder sequence to refine (empty = de novo).
        omit_aa: Amino acids to omit during MPNN (default cysteine).
        plot: Write per-run metric plots.
        template_path: PDB code or path to a template structure.
        template_cif_chain_id: Chain ID in a template mmCIF.

    Returns:
        Summary with volume name, save path, and high-ipTM count.
    """
    import os
    import sys

    os.environ.setdefault("MPLBACKEND", "Agg")
    os.chdir(PH_ROOT)
    sys.path.insert(0, PH_ROOT)
    sys.path.insert(0, f"{PH_ROOT}/boltz_ph")

    import matplotlib

    matplotlib.use("Agg")

    # ColabFold MSA server can trip SSL verify on Modal
    import requests.adapters
    import urllib3

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    _orig_send = requests.adapters.HTTPAdapter.send
    requests.adapters.HTTPAdapter.send = lambda self, request, **kw: _orig_send(
        self, request, **{**kw, "verify": False}
    )

    argv = [
        "design.py",
        "--gpu_id",
        "0",
        "--name",
        name,
        "--num_designs",
        str(num_designs),
        "--num_cycles",
        str(num_cycles),
        "--min_protein_length",
        str(min_protein_length),
        "--max_protein_length",
        str(max_protein_length),
        "--percent_X",
        str(percent_x),
        "--msa_mode",
        msa_mode,
        "--high_iptm_threshold",
        str(high_iptm_threshold),
        "--omit_AA",
        omit_aa,
        "--save_dir",
        save_dir,
        "--work_dir",
        PH_ROOT,
        "--boltz_model_path",
        "/root/.boltz/boltz2_conf.ckpt",
        "--ccd_path",
        "/root/.boltz/mols",
    ]
    if protein_seqs:
        argv += ["--protein_seqs", protein_seqs]
    if contact_residues:
        argv += ["--contact_residues", contact_residues]
    if seq:
        argv += ["--seq", seq]
    if ligand_ccd:
        argv += ["--ligand_ccd", ligand_ccd]
    if ligand_smiles:
        argv += ["--ligand_smiles", ligand_smiles]
    if nucleic_seq:
        argv += ["--nucleic_seq", nucleic_seq, "--nucleic_type", nucleic_type]
    if template_path:
        argv += ["--template_path", template_path]
        if template_cif_chain_id:
            argv += ["--template_cif_chain_id", template_cif_chain_id]
    if cyclic:
        argv.append("--cyclic")
    if alanine_bias:
        argv.append("--alanine_bias")
    if plot:
        argv.append("--plot")

    sys.argv = argv
    from design import parse_args, print_args
    from pipeline import ProteinHunter_Boltz

    _patch_colabfold_msa()
    args = parse_args()
    print_args(args)
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    class ProteinHunterBoltzModal(ProteinHunter_Boltz):
        """Commit the volume after each trajectory so --detach jobs keep partial results."""

        def _run_design_cycle(self, *cycle_args, **cycle_kwargs):
            try:
                return super()._run_design_cycle(*cycle_args, **cycle_kwargs)
            finally:
                VOLUME.commit()

    hunter = ProteinHunterBoltzModal(args)
    hunter.run_pipeline()
    VOLUME.commit()

    high_iptm_pdb = Path(save_dir) / "high_iptm_pdb"
    n_high = (
        len(list(high_iptm_pdb.glob("*.pdb"))) if high_iptm_pdb.exists() else 0
    )
    print(f"Results committed to volume '{VOLUME_NAME}' at {save_dir}")
    return {
        "volume": VOLUME_NAME,
        "save_dir": save_dir,
        "name": name,
        "high_iptm_pdbs": n_high,
    }


@app.local_entrypoint()
def main(
    protein_seqs: str | None = None,
    input_pdb: str | None = None,
    target_chains: str = "A",
    contact_residues: str = "",
    lengths: str = "90,150",
    num_designs: int = 1,
    num_cycles: int = 5,
    percent_x: int = 100,
    msa_mode: str = "mmseqs",
    high_iptm_threshold: float = 0.7,
    cyclic: bool = False,
    alanine_bias: bool = False,
    ligand_ccd: str = "",
    ligand_smiles: str = "",
    nucleic_seq: str = "",
    nucleic_type: str = "dna",
    seq: str = "",
    omit_aa: str = "C",
    plot: bool = True,
    template_path: str = "",
    template_cif_chain_id: str = "",
    binder_name: str | None = None,
    run_name: str | None = None,
):
    """Local entrypoint to run Protein Hunter binder design.

    Uses spawn().get() (required for long detached jobs). Run with
    `modal run --detach` so closing the terminal does not kill the job.
    Results are stored on the Modal Volume named "proteinhunter".

    Provide `--protein-seqs` and/or `--input-pdb`. A PDB is converted to
    sequences for the chains in `--target-chains`. Ligand/nucleic-only
    design needs `--ligand-ccd`, `--ligand-smiles`, or `--nucleic-seq`.

    Args:
        protein_seqs: Target amino-acid sequence(s). Colon-separated for multimers.
        input_pdb: Path to a target PDB; sequences are extracted from target_chains.
        target_chains: Chain(s) to read from input_pdb. Defaults to "A".
        contact_residues: Hotspot-like contact residues on the target (e.g. "2,3,10").
        lengths: Comma-separated min,max binder length (e.g. "90,150").
        num_designs: Number of design trajectories. Defaults to 1.
        num_cycles: Optimization cycles per trajectory. Defaults to 5.
        percent_x: Percent X in the initial binder sequence. Defaults to 100.
        msa_mode: "mmseqs" or "single". Defaults to "mmseqs".
        high_iptm_threshold: ipTM filter for high-confidence designs. Defaults to 0.7.
        cyclic: If True, design a cyclic peptide.
        alanine_bias: If True, penalize alanine during MPNN.
        ligand_ccd: Optional small-molecule CCD code.
        ligand_smiles: Optional small-molecule SMILES.
        nucleic_seq: Optional DNA/RNA sequence to bind.
        nucleic_type: "dna" or "rna". Defaults to "dna".
        seq: Optional binder sequence to refine rather than design de novo.
        omit_aa: Amino acids omitted by MPNN. Defaults to "C".
        plot: Write cycle plots. Defaults to True.
        template_path: Optional PDB code or template file path.
        template_cif_chain_id: Chain ID for a template mmCIF.
        binder_name: Job name. Defaults to the PDB stem or "binder".
        run_name: Volume subdirectory. Defaults to a timestamp.
    """
    from datetime import datetime

    if not protein_seqs and input_pdb:
        protein_seqs = pdb_chains_to_seqs(open(input_pdb).read(), target_chains)
        print(f"Extracted target sequence(s) from {input_pdb} chains {target_chains}:")
        print(protein_seqs)

    if not protein_seqs and not ligand_ccd and not ligand_smiles and not nucleic_seq:
        raise SystemExit(
            "Provide --protein-seqs, --input-pdb, --ligand-ccd, --ligand-smiles, or --nucleic-seq"
        )

    parts = [int(x) for x in lengths.split(",")]
    if len(parts) == 1:
        min_len = max_len = parts[0]
    else:
        min_len, max_len = parts[0], parts[1]

    today = datetime.now().strftime("%Y%m%d%H%M")[2:]
    run_subdir = run_name or today
    binder_name = binder_name or (Path(input_pdb).stem if input_pdb else "binder")
    save_dir = f"{VOLUME_MOUNT}/{run_subdir}/{binder_name}"

    print(f"Results will be written to volume '{VOLUME_NAME}' at {save_dir}")
    print("Run with --detach; you can close the terminal and the job will keep going.")
    print(
        f"Download later: modal volume get {VOLUME_NAME} {run_subdir} ./out/proteinhunter/"
    )
    print(f"List volume:     modal volume ls {VOLUME_NAME} {run_subdir}")

    result = proteinhunter.spawn(
        save_dir=save_dir,
        name=binder_name,
        protein_seqs=protein_seqs or "",
        num_designs=num_designs,
        num_cycles=num_cycles,
        min_protein_length=min_len,
        max_protein_length=max_len,
        percent_x=percent_x,
        msa_mode=msa_mode,
        high_iptm_threshold=high_iptm_threshold,
        contact_residues=contact_residues,
        cyclic=cyclic,
        alanine_bias=alanine_bias,
        ligand_ccd=ligand_ccd,
        ligand_smiles=ligand_smiles,
        nucleic_seq=nucleic_seq,
        nucleic_type=nucleic_type,
        seq=seq,
        omit_aa=omit_aa,
        plot=plot,
        template_path=template_path,
        template_cif_chain_id=template_cif_chain_id,
    ).get()
    print(f"Protein Hunter finished: {result}")
