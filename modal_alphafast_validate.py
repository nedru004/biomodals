# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "modal>=1.0",
# ]
# ///
"""Validate designed protein binders with AlphaFast (AlphaFold 3).

Uses the published AlphaFast container (`romerolabduke/alphafast:latest`)
instead of building AlphaFold 3 from source:
https://github.com/RomeroLab/alphafast#modal-setup

Upload weights once onto AlphaFast's `af3-weights` volume:

    uv run --with modal modal run modal_alphafast_validate.py \\
      --upload-weights /path/to/af3.bin.zst

From a BindCraft volume folder (chain A = target, B = binder):

    GPU=A100-80GB uv run --with modal modal run --detach modal_alphafast_validate.py \\
      --volume-name bindcraft --input-dir <run>/<target>/Accepted \\
      --binder-chain B --target-chains A

Download later:

    modal volume get alphafast-validate <run_name> ./out/alphafast_validate/
"""

from __future__ import annotations

import os
from pathlib import Path

from modal import App, Image, Volume

GPU = os.environ.get("GPU", "A100-80GB")
TIMEOUT = int(os.environ.get("TIMEOUT", 90))
ORCH_TIMEOUT = int(os.environ.get("ORCH_TIMEOUT", 24)) * 60 * 60
MAX_CONTAINERS = int(os.environ.get("MAX_CONTAINERS", 8))
print(f"Using GPU {GPU}; TIMEOUT {TIMEOUT} min/design; MAX_CONTAINERS {MAX_CONTAINERS}")

# Match RomeroLab/alphafast modal/config.py
AF3_REPO = "/app/alphafold"
AF3_VENV = "/alphafold3_venv"
MODEL_VOLUME_NAME = "af3-weights"
MODEL_VOLUME = Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=True)
MODEL_MOUNT = "/weights"

KNOWN_VOLUME_NAMES = ("bindcraft", "proteinhunter")
EXTRA_VOLUME_NAME = os.environ.get("DESIGN_VOLUME")

OUT_VOLUME_NAME = "alphafast-validate"
OUT_VOLUME = Volume.from_name(OUT_VOLUME_NAME, create_if_missing=True)
OUT_MOUNT = f"/{OUT_VOLUME_NAME}"

_volume_mounts: dict[str, Volume] = {
    MODEL_MOUNT: MODEL_VOLUME,
    OUT_MOUNT: OUT_VOLUME,
}
for _name in KNOWN_VOLUME_NAMES:
    _volume_mounts[f"/vol/{_name}"] = Volume.from_name(_name, create_if_missing=True)
if EXTRA_VOLUME_NAME and EXTRA_VOLUME_NAME not in KNOWN_VOLUME_NAMES:
    _volume_mounts[f"/vol/{EXTRA_VOLUME_NAME}"] = Volume.from_name(
        EXTRA_VOLUME_NAME, create_if_missing=True
    )

STRUCTURE_SUFFIXES = {".pdb", ".cif", ".mmcif"}
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
    "SEC": "C",
    "PYL": "K",
}

WEIGHTS_HELP = (
    f"No AlphaFold 3 weights found on volume '{MODEL_VOLUME_NAME}'. "
    "Upload once (AlphaFast convention):\n"
    "  uv run --with modal modal run modal_alphafast_validate.py "
    "--upload-weights /path/to/af3.bin.zst\n"
    f"The volume is mounted at {MODEL_MOUNT} and passed as --model_dir."
)

# Inference-only XLA flags from alphafast modal/af3_predict.py
_XLA_INFERENCE_ENV = {
    "XLA_FLAGS": "--xla_gpu_enable_triton_gemm=false",
    "XLA_PYTHON_CLIENT_PREALLOCATE": "true",
    "XLA_CLIENT_MEM_FRACTION": "0.95",
}

# Prebuilt AlphaFast image — no local AF3 compile.
# https://github.com/RomeroLab/alphafast#modal-setup
# gemmi is for this script's PDB/CIF parsing (Modal function Python),
# not for the AlphaFast venv (which has no pip).
image = (
    Image.from_registry("romerolabduke/alphafast:latest")
    .entrypoint([])
    .uv_pip_install("gemmi")
)

app = App("alphafast-validate", image=image)


def _disable_ssl_verify() -> None:
    """ColabFold MSA server can fail Modal's default SSL verify."""
    import requests.adapters
    import urllib3

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    _orig_send = requests.adapters.HTTPAdapter.send
    requests.adapters.HTTPAdapter.send = lambda self, request, **kw: _orig_send(
        self, request, **{**kw, "verify": False}
    )


def _chain_ids(chains: str) -> list[str]:
    """Parse 'A', 'A,B', 'A:B', or 'AB' into chain IDs."""
    if not chains:
        return []
    if "," in chains:
        return [c.strip() for c in chains.split(",") if c.strip()]
    if ":" in chains:
        return [c.strip() for c in chains.split(":") if c.strip()]
    if len(chains) > 1 and chains.isalnum() and chains.isupper():
        return list(chains.replace(" ", ""))
    return [chains.strip()]


def _read_structure(name: str, content: str):
    """Parse PDB or mmCIF text into a gemmi Structure."""
    import gemmi

    lower = name.lower()
    if lower.endswith((".cif", ".mmcif")):
        doc = gemmi.cif.read_string(content)
        return gemmi.make_structure_from_block(doc.sole_block())
    return gemmi.read_pdb_string(content)


def extract_chains(name: str, content: str) -> dict[str, dict]:
    """Extract per-chain protein sequence and CA coordinates from a structure.

    Returns:
        Mapping of chain ID to dict with keys sequence (str), ca (N,3 float array),
        plddt (N float array from B-factors).
    """
    import numpy as np

    st = _read_structure(name, content)
    if len(st) == 0:
        raise ValueError(f"No models in {name}")
    try:
        st.merge_chain_parts()
    except Exception:
        pass

    chains: dict[str, dict] = {}
    for chain in st[0]:
        seq: list[str] = []
        ca: list[list[float]] = []
        plddt: list[float] = []
        for residue in chain:
            aa = AA3TO1.get(residue.name)
            if aa is None:
                continue
            atom = residue.find_atom("CA", "*")
            if atom is None:
                continue
            seq.append(aa)
            ca.append([atom.pos.x, atom.pos.y, atom.pos.z])
            plddt.append(float(atom.b_iso))
        if seq:
            chains[chain.name] = {
                "sequence": "".join(seq),
                "ca": np.asarray(ca, dtype=float),
                "plddt": np.asarray(plddt, dtype=float),
            }
    if not chains:
        raise ValueError(f"No protein CA atoms in {name}")
    return chains


def resolve_roles(
    chain_ids: list[str],
    binder_chain: str | None,
    target_chains: list[str],
) -> tuple[str, list[str]]:
    """Decide binder vs target chain IDs, with shortest-chain fallback."""
    if binder_chain and binder_chain in chain_ids:
        binder = binder_chain
    elif len(chain_ids) == 1:
        binder = chain_ids[0]
    else:
        binder = None

    targets = [c for c in target_chains if c in chain_ids and c != binder]
    if not targets and binder is None and len(chain_ids) >= 2:
        binder = min(chain_ids, key=len)
        targets = [c for c in chain_ids if c != binder]
    elif not targets:
        targets = [c for c in chain_ids if c != binder]

    if binder is None:
        raise ValueError(
            f"Could not identify binder chain from {chain_ids}; "
            "pass --binder-chain"
        )
    return binder, targets


def _kabsch(P, Q):
    """Rigid superposition mapping P onto Q.

    Returns:
        R, t, rmsd, aligned P coordinates.
    """
    import numpy as np

    if len(P) != len(Q) or len(P) < 3:
        return None, None, None, None
    Pc = P.mean(axis=0)
    Qc = Q.mean(axis=0)
    P0 = P - Pc
    Q0 = Q - Qc
    H = P0.T @ Q0
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    t = Qc - R @ Pc
    aligned = (R @ P.T).T + t
    rmsd = float(np.sqrt(((aligned - Q) ** 2).sum(axis=1).mean()))
    return R, t, rmsd, aligned


def _tm_score(aligned, reference) -> float | None:
    """TM-score after coordinates are already superimposed."""
    import numpy as np

    n = len(reference)
    if n < 3:
        return None
    d0 = max(0.5, 1.24 * (max(n, 27) - 15) ** (1.0 / 3.0) - 1.8)
    d = np.linalg.norm(aligned - reference, axis=1)
    return float(np.mean(1.0 / (1.0 + (d / d0) ** 2)))


def _apply_rt(coords, R, t):
    return (R @ coords.T).T + t


def _match_ca(pred: dict, ref: dict):
    """Trim to the shared prefix length when residue counts differ."""
    n = min(len(pred["ca"]), len(ref["ca"]))
    if n < 3:
        return None, None
    return pred["ca"][:n], ref["ca"][:n]


def compute_rmsds(
    pred_chains: dict[str, dict],
    ref_chains: dict[str, dict],
    binder: str,
    targets: list[str],
) -> dict[str, float | None]:
    """RMSD / TM-score of a prediction against designed coordinates."""
    import numpy as np

    def matched(ids: list[str]):
        ps, rs = [], []
        for c in ids:
            if c not in pred_chains or c not in ref_chains:
                continue
            a, b = _match_ca(pred_chains[c], ref_chains[c])
            if a is None:
                continue
            ps.append(a)
            rs.append(b)
        if not ps:
            return None, None
        return np.concatenate(ps), np.concatenate(rs)

    out = {
        "rmsd_complex": None,
        "tm_complex": None,
        "rmsd_binder_on_target": None,
        "tm_binder_on_target": None,
        "rmsd_binder_fold": None,
        "tm_binder_fold": None,
    }
    shared = [c for c in pred_chains if c in ref_chains]
    pred_all, ref_all = matched(shared)
    if pred_all is not None:
        _, _, rmsd, aligned = _kabsch(pred_all, ref_all)
        out["rmsd_complex"] = rmsd
        out["tm_complex"] = _tm_score(aligned, ref_all) if aligned is not None else None

    pb, rb = None, None
    if binder in pred_chains and binder in ref_chains:
        pb, rb = _match_ca(pred_chains[binder], ref_chains[binder])
        if pb is not None:
            _, _, rmsd, aligned = _kabsch(pb, rb)
            out["rmsd_binder_fold"] = rmsd
            out["tm_binder_fold"] = _tm_score(aligned, rb) if aligned is not None else None

    pred_tgt, ref_tgt = matched([t for t in targets])
    if pred_tgt is not None and pb is not None:
        R, t, _, _ = _kabsch(pred_tgt, ref_tgt)
        if R is not None:
            aligned_b = _apply_rt(pb, R, t)
            out["rmsd_binder_on_target"] = float(
                np.sqrt(((aligned_b - rb) ** 2).sum(axis=1).mean())
            )
            out["tm_binder_on_target"] = _tm_score(aligned_b, rb)
    return out


def _mean_plddt(chain: dict | None) -> float | None:
    if not chain:
        return None
    p = chain.get("plddt")
    if p is None or len(p) == 0:
        return None
    vals = [x for x in p if x == x]
    if not vals:
        return None
    mean = sum(vals) / len(vals)
    return float(mean / 100.0) if mean > 1.5 else float(mean)


def _free_chain_id(taken: set[str]) -> str:
    for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        if c not in taken:
            return c
    raise ValueError(f"No free chain ID left; taken={taken}")


def _af3_job_name(name: str) -> str:
    import re

    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-")
    return cleaned or "design"


def _has_weights(directory: Path) -> bool:
    if not directory.is_dir():
        return False
    for f in directory.iterdir():
        if not f.is_file():
            continue
        n = f.name.lower()
        if n.endswith(".bin.zst") or n.endswith(".bin"):
            return True
        if ".bin.zst." in n or n.endswith(".bin.zst"):
            return True
    return False


def resolve_model_dir() -> Path:
    """Directory that contains af3.bin.zst / af3.bin on the models volume."""
    root = Path(MODEL_MOUNT)
    if _has_weights(root):
        return root
    if root.is_dir():
        for sub in sorted(root.iterdir()):
            if sub.is_dir() and _has_weights(sub):
                return sub
    raise FileNotFoundError(WEIGHTS_HELP)


def _empty_protein(chain_id: str, sequence: str, unpaired_msa: str | None) -> dict:
    """AF3 protein entity. Empty MSA/templates skips the data pipeline."""
    msa = unpaired_msa if unpaired_msa is not None else ""
    return {
        "protein": {
            "id": chain_id,
            "sequence": sequence,
            "unpairedMsa": msa,
            "pairedMsa": "",
            "templates": [],
        }
    }


def build_af3_json(
    name: str,
    target_seqs: list[tuple[str, str]],
    binder_id: str,
    binder_seq: str,
    seed: int,
    target_msas: dict[str, str] | None,
) -> dict:
    """AF3 input JSON: ColabFold MSA on targets (if provided), empty binder MSA."""
    sequences = []
    for chain_id, seq in target_seqs:
        msa = (target_msas or {}).get(chain_id) or ""
        sequences.append(_empty_protein(chain_id, seq, msa))
    sequences.append(_empty_protein(binder_id, binder_seq, ""))
    return {
        "name": _af3_job_name(name),
        "sequences": sequences,
        "modelSeeds": [int(seed)],
        "dialect": "alphafold3",
        "version": 1,
    }


def _a3m_from_colabfold_tar(tar_bytes: bytes, query_seq: str) -> str:
    """Merge ColabFold A3M files into a single AF3 unpairedMsa string."""
    import io
    import tarfile

    texts: list[str] = []
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
        for member in tar.getmembers():
            name = member.name.lower()
            if not name.endswith(".a3m") or "pair" in name:
                continue
            fh = tar.extractfile(member)
            if fh is None:
                continue
            texts.append(fh.read().decode("utf-8", errors="replace"))

    records: list[tuple[str, str]] = []
    seen: set[str] = set()
    for text in texts:
        header = None
        seq_parts: list[str] = []
        for line in text.splitlines():
            if line.startswith("#"):
                continue
            if line.startswith(">"):
                if header is not None:
                    s = "".join(seq_parts).replace("\x00", "")
                    key = header.split()[0]
                    if s and key not in seen:
                        records.append((header, s))
                        seen.add(key)
                header = line
                seq_parts = []
            elif header is not None:
                seq_parts.append(line.strip())
        if header is not None:
            s = "".join(seq_parts).replace("\x00", "")
            key = header.split()[0]
            if s and key not in seen:
                records.append((header, s))
                seen.add(key)

    lines = [">query", query_seq]
    q = query_seq.upper()
    for header, seq in records:
        raw = seq.replace("-", "").replace(".", "").upper()
        if raw == q:
            continue
        lines.append(header)
        lines.append(seq)
    return "\n".join(lines) + "\n"


def fetch_colabfold_msa(sequence: str, host: str = "https://api.colabfold.com") -> str:
    """A3M from the ColabFold MMseqs2 server (UniRef + environmental)."""
    import random
    import time

    import requests

    headers = {"User-Agent": "biomodals-alphafast-validate/1.0"}
    query = f">101\n{sequence}\n"
    mode = "env"

    def submit():
        res = requests.post(
            f"{host}/ticket/msa",
            data={"q": query, "mode": mode},
            timeout=30,
            headers=headers,
            verify=False,
        )
        res.raise_for_status()
        return res.json()

    out = submit()
    while out.get("status") in ("UNKNOWN", "RATELIMIT"):
        time.sleep(5 + random.random() * 5)
        out = submit()
    if out.get("status") == "ERROR" or "id" not in out:
        raise RuntimeError(f"ColabFold MSA submit failed: {out}")

    ticket = out["id"]
    while True:
        st = requests.get(
            f"{host}/ticket/{ticket}", timeout=30, headers=headers, verify=False
        )
        st.raise_for_status()
        payload = st.json()
        status = payload.get("status")
        if status == "COMPLETE":
            break
        if status == "ERROR":
            raise RuntimeError(f"ColabFold MSA ticket {ticket} failed: {payload}")
        time.sleep(5 + random.random() * 5)

    raw = requests.get(
        f"{host}/result/download/{ticket}",
        timeout=120,
        headers=headers,
        verify=False,
    )
    raw.raise_for_status()
    return _a3m_from_colabfold_tar(raw.content, sequence)


def parse_af3_summary(
    path: Path, binder_id: str, target_ids: list[str]
) -> dict:
    """Pull pTM / ipTM and binder–target pair metrics from AF3 summary JSON."""
    import json

    data = json.loads(path.read_text())
    chain_ids = [str(c) for c in (data.get("chain_ids") or [])]
    out = {
        "ptm": data.get("ptm"),
        "iptm": data.get("iptm"),
        "ranking_score": data.get("ranking_score"),
        "fraction_disordered": data.get("fraction_disordered"),
        "has_clash": data.get("has_clash"),
        "binder_target_iptm": None,
        "binder_ptm": None,
        "chain_pair_pae_min": None,
    }
    if binder_id in chain_ids:
        b = chain_ids.index(binder_id)
        chain_ptm = data.get("chain_ptm") or []
        if b < len(chain_ptm):
            out["binder_ptm"] = chain_ptm[b]
        pair_iptm = data.get("chain_pair_iptm") or []
        pair_pae = data.get("chain_pair_pae_min") or []
        iptm_vals = []
        pae_vals = []
        for t_id in target_ids:
            if t_id not in chain_ids:
                continue
            t = chain_ids.index(t_id)
            for i, j in ((b, t), (t, b)):
                try:
                    v = pair_iptm[i][j]
                except (IndexError, TypeError):
                    v = None
                if v is not None:
                    iptm_vals.append(float(v))
                try:
                    p = pair_pae[i][j]
                except (IndexError, TypeError):
                    p = None
                if p is not None:
                    pae_vals.append(float(p))
        if iptm_vals:
            out["binder_target_iptm"] = max(iptm_vals)
        if pae_vals:
            out["chain_pair_pae_min"] = min(pae_vals)
    return out


def parse_interface_pae(
    conf_path: Path, binder_id: str, target_ids: list[str]
) -> dict[str, float | None]:
    """Mean / min PAE between binder and target tokens (lower is better)."""
    import json

    import numpy as np

    data = json.loads(conf_path.read_text())
    pae = np.asarray(data["pae"], dtype=float)
    token_chains = [str(c) for c in data.get("token_chain_ids") or []]
    if len(token_chains) != pae.shape[0]:
        return {"interface_pae_mean": None, "interface_pae_min": None}
    binder_mask = np.array([c == binder_id for c in token_chains])
    target_set = set(target_ids)
    target_mask = np.array([c in target_set for c in token_chains])
    if not binder_mask.any() or not target_mask.any():
        return {"interface_pae_mean": None, "interface_pae_min": None}
    blocks = [
        pae[np.ix_(binder_mask, target_mask)],
        pae[np.ix_(target_mask, binder_mask)],
    ]
    cat = np.concatenate([b.ravel() for b in blocks])
    return {
        "interface_pae_mean": float(cat.mean()),
        "interface_pae_min": float(cat.min()),
    }


def _run_af3(json_obj: dict, out_dir: Path, extra_flags: str) -> None:
    import json
    from subprocess import run

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "input.json"
    json_path.write_text(json.dumps(json_obj, indent=2))
    model_dir = resolve_model_dir()
    jax_cache = Path(MODEL_MOUNT) / "jax_cache"
    jax_cache.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, **_XLA_INFERENCE_ENV}
    cmd = (
        f'"{AF3_VENV}/bin/python" run_alphafold.py'
        f' --json_path="{json_path}"'
        f' --model_dir="{model_dir}"'
        f' --output_dir="{out_dir}"'
        " --norun_data_pipeline"
        " --force_output_dir"
        f' --jax_compilation_cache_dir="{jax_cache}"'
        f" {extra_flags}"
    )
    run(cmd, shell=True, check=True, cwd=AF3_REPO, env=env)


def _prediction_paths(pred_root: Path, job_name: str) -> tuple[Path | None, Path | None, Path | None]:
    """Return (structure cif, summary json, confidences json) for the top-ranked sample."""
    stem = _af3_job_name(job_name)
    job_dir = pred_root / stem
    if not job_dir.is_dir():
        dirs = [p for p in pred_root.iterdir() if p.is_dir()]
        job_dir = dirs[0] if dirs else pred_root
    struct = job_dir / f"{job_dir.name}_model.cif"
    summary = job_dir / f"{job_dir.name}_summary_confidences.json"
    conf = job_dir / f"{job_dir.name}_confidences.json"
    if not struct.exists():
        cifs = sorted(pred_root.glob("**/*_model.cif"))
        struct = cifs[0] if cifs else None
    else:
        struct = struct if struct.exists() else None
    if not summary.exists():
        summaries = sorted(pred_root.glob("**/*_summary_confidences.json"))
        summary = summaries[0] if summaries else None
    else:
        summary = summary if summary.exists() else None
    if not conf.exists():
        confs = sorted(pred_root.glob("**/*_confidences.json"))
        # Prefer the top-level file over per-sample copies.
        top = [p for p in confs if p.parent == (struct.parent if struct else job_dir)]
        conf = (top[0] if top else confs[0]) if confs else None
    else:
        conf = conf if conf.exists() else None
    return struct, summary, conf


def _copy_tree(src: Path, dest: Path) -> None:
    import shutil

    dest.mkdir(parents=True, exist_ok=True)
    for f in src.glob("**/*"):
        if f.is_file():
            target = dest / f.relative_to(src)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, target)


def _iter_structure_files(root: Path, recursive: bool) -> list[Path]:
    files = root.rglob("*") if recursive else root.glob("*")
    return sorted(
        f
        for f in files
        if f.is_file()
        and f.suffix.lower() in STRUCTURE_SUFFIXES
        and "Ranked" not in f.parts
    )


def seq_identity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    return sum(x == y for x, y in zip(a[:n], b[:n])) / max(len(a), len(b))


def _load_msa_map(paths: dict[str, str] | None) -> dict[str, str]:
    if not paths:
        return {}
    out = {}
    for chain_id, path in paths.items():
        if not path:
            continue
        p = Path(path)
        if p.exists():
            out[chain_id] = p.read_text()
    return out


@app.function(
    timeout=TIMEOUT * 60,
    gpu=GPU,
    volumes=_volume_mounts,
    max_containers=MAX_CONTAINERS,
)
def validate_one(job: dict) -> dict:
    """AF3 complex + optional monomer prediction for one design."""
    from tempfile import TemporaryDirectory

    name = job["name"]
    print(f"=== {name} ===")
    _disable_ssl_verify()

    row = {
        "design": name,
        "binder_chain": job["binder_chain"],
        "target_chains": ",".join(job["target_chains"]),
        "binder_seq": job["binder_seq"],
        "target_seq": job["target_seq"],
        "binder_len": len(job["binder_seq"]),
        "target_len": len((job.get("target_seq") or "").replace(":", "")),
    }

    try:
        resolve_model_dir()
        ref_chains = extract_chains(job["filename"], job["content"])
        binder = job["binder_chain"]
        targets = list(job["target_chains"])
        if binder not in ref_chains:
            raise ValueError(f"{name}: binder chain {binder} not in structure")

        target_seqs = [(c, ref_chains[c]["sequence"]) for c in targets if c in ref_chains]
        if job.get("target_content") and not target_seqs:
            tgt_chains = extract_chains(
                job.get("target_filename") or "target.pdb", job["target_content"]
            )
            t_ids = [c for c in (job.get("target_file_chains") or []) if c in tgt_chains]
            if not t_ids:
                t_ids = list(tgt_chains)
            target_seqs = [(c, tgt_chains[c]["sequence"]) for c in t_ids]
            targets = t_ids

        if not target_seqs:
            raise ValueError(
                f"{name}: no target sequences (missing --target-chains or --target-pdb)"
            )

        taken = {c for c, _ in target_seqs}
        binder_json = binder if binder not in taken else _free_chain_id(taken)
        target_msas = _load_msa_map(job.get("target_msa_paths"))
        extra = job.get("params_str") or ""
        seed = int(job.get("seed") or 1)

        json_complex = build_af3_json(
            name,
            target_seqs,
            binder_json,
            job["binder_seq"],
            seed,
            target_msas,
        )
        json_mono = build_af3_json(
            f"{name}_monomer",
            [],
            binder,
            job["binder_seq"],
            seed,
            None,
        )

        target_json_ids = [c for c, _ in target_seqs]

        with TemporaryDirectory() as td:
            td_path = Path(td)
            complex_dir = td_path / "complex"
            _run_af3(json_complex, complex_dir, extra)
            struct, summary, conf = _prediction_paths(complex_dir, json_complex["name"])
            if summary:
                row.update(parse_af3_summary(summary, binder_json, target_json_ids))
            if conf:
                row.update(parse_interface_pae(conf, binder_json, target_json_ids))
            pred_chains = (
                extract_chains(struct.name, struct.read_text()) if struct else {}
            )
            if pred_chains:
                row["binder_plddt_complex"] = _mean_plddt(pred_chains.get(binder_json))
                tgt_plddts = [
                    p
                    for c in target_json_ids
                    if (p := _mean_plddt(pred_chains.get(c))) is not None
                ]
                row["target_plddt_complex"] = (
                    sum(tgt_plddts) / len(tgt_plddts) if tgt_plddts else None
                )
                all_plddts = [
                    p
                    for ch in pred_chains.values()
                    if (p := _mean_plddt(ch)) is not None
                ]
                row["complex_plddt"] = (
                    sum(all_plddts) / len(all_plddts) if all_plddts else None
                )
                pred_for_rmsd = dict(pred_chains)
                if binder_json != binder and binder_json in pred_for_rmsd:
                    pred_for_rmsd[binder] = pred_for_rmsd[binder_json]
                row.update(compute_rmsds(pred_for_rmsd, ref_chains, binder, targets))

            if job.get("run_monomer", True):
                mono_dir = td_path / "monomer"
                _run_af3(json_mono, mono_dir, extra)
                m_struct, m_summary, _ = _prediction_paths(
                    mono_dir, json_mono["name"]
                )
                if m_summary:
                    m = parse_af3_summary(m_summary, binder, [])
                    row["monomer_ptm"] = m.get("ptm")
                    row["monomer_ranking_score"] = m.get("ranking_score")
                if m_struct and binder in ref_chains:
                    m_chains = extract_chains(m_struct.name, m_struct.read_text())
                    m_id = binder if binder in m_chains else next(iter(m_chains))
                    row["monomer_plddt"] = _mean_plddt(m_chains.get(m_id))
                    pb, rb = _match_ca(m_chains[m_id], ref_chains[binder])
                    if pb is not None:
                        _, _, rmsd, aligned = _kabsch(pb, rb)
                        row["rmsd_binder_monomer"] = rmsd
                        row["tm_binder_monomer"] = (
                            _tm_score(aligned, rb) if aligned is not None else None
                        )
                    if row.get("binder_plddt_complex") is not None and row.get(
                        "monomer_plddt"
                    ) is not None:
                        row["delta_plddt_binder"] = (
                            row["binder_plddt_complex"] - row["monomer_plddt"]
                        )

            dest = Path(job["out_dir"]) / "predictions" / name
            _copy_tree(complex_dir, dest / "complex")
            if job.get("run_monomer", True):
                _copy_tree(td_path / "monomer", dest / "monomer")

        try:
            MODEL_VOLUME.commit()
        except Exception:
            pass
        OUT_VOLUME.commit()
    except Exception as exc:
        print(f"FAILED {name}: {exc}")
        row["error"] = str(exc)
        return row

    row["error"] = ""
    return row


@app.function(
    timeout=ORCH_TIMEOUT,
    volumes=_volume_mounts,
)
def run_validation(
    volume_name: str | None,
    input_dir: str,
    designs: list[dict] | None,
    binder_chain: str,
    target_chains: str,
    target_filename: str | None,
    target_content: str | None,
    target_file_chains: list[str],
    target_pdb_path: str | None,
    run_name: str,
    params_str: str,
    use_msa: bool,
    run_monomer: bool,
    recursive: bool,
    seed: int,
) -> dict:
    """Load designs, run AF3 per design, write rankings.csv."""
    import csv
    import hashlib
    from collections import Counter
    from datetime import datetime

    resolve_model_dir()

    out_dir = Path(OUT_MOUNT) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if target_content is None and target_pdb_path:
        tpath = Path(target_pdb_path)
        vol_path = Path(f"/vol/{volume_name}") / target_pdb_path if volume_name else None
        if tpath.exists():
            target_content = tpath.read_text()
            target_filename = target_filename or tpath.name
        elif vol_path is not None and vol_path.exists():
            target_content = vol_path.read_text()
            target_filename = target_filename or vol_path.name
        else:
            raise FileNotFoundError(
                f"target-pdb not found locally or on volume: {target_pdb_path}"
            )

    loaded: list[dict] = []
    if designs:
        loaded = designs
    else:
        if not volume_name:
            raise ValueError("volume_name or local designs required")
        mount = Path(f"/vol/{volume_name}") / input_dir
        if not mount.exists():
            raise FileNotFoundError(
                f"No such path on volume '{volume_name}': {input_dir} "
                f"(looked at {mount})"
            )
        files = _iter_structure_files(mount, recursive)
        if not files:
            raise FileNotFoundError(f"No PDB/CIF files under {mount}")
        for f in files:
            loaded.append(
                {"name": f.stem, "filename": f.name, "content": f.read_text()}
            )

    t_chains = _chain_ids(target_chains)
    jobs = []
    seq_rows = []
    parse_errors = []
    for item in loaded:
        try:
            chains = extract_chains(item["filename"], item["content"])
        except Exception as exc:
            parse_errors.append({"design": item["name"], "error": str(exc)})
            continue
        try:
            b_id, tgt_ids = resolve_roles(list(chains), binder_chain, t_chains)
        except Exception as exc:
            parse_errors.append({"design": item["name"], "error": str(exc)})
            continue

        if (not tgt_ids or b_id in tgt_ids and len(chains) == 1) and target_content:
            tgt_struct = extract_chains(target_filename or "target.pdb", target_content)
            use_ids = [c for c in (target_file_chains or t_chains) if c in tgt_struct]
            if not use_ids:
                use_ids = list(tgt_struct)
            tgt_ids = use_ids
            target_seq = ":".join(tgt_struct[c]["sequence"] for c in tgt_ids)
            target_chain_seqs = {c: tgt_struct[c]["sequence"] for c in tgt_ids}
        else:
            target_seq = ":".join(chains[c]["sequence"] for c in tgt_ids if c in chains)
            target_chain_seqs = {c: chains[c]["sequence"] for c in tgt_ids if c in chains}

        binder_seq = chains[b_id]["sequence"]
        for cid, data in chains.items():
            role = "binder" if cid == b_id else ("target" if cid in tgt_ids else "other")
            seq_rows.append(
                {
                    "design": item["name"],
                    "chain": cid,
                    "role": role,
                    "length": len(data["sequence"]),
                    "sequence": data["sequence"],
                }
            )
        jobs.append(
            {
                "name": item["name"],
                "filename": item["filename"],
                "content": item["content"],
                "binder_chain": b_id,
                "target_chains": tgt_ids,
                "binder_seq": binder_seq,
                "target_seq": target_seq,
                "target_chain_seqs": target_chain_seqs,
                "target_filename": target_filename,
                "target_content": target_content,
                "target_file_chains": target_file_chains,
                "params_str": params_str,
                "use_msa": use_msa,
                "run_monomer": run_monomer,
                "seed": seed,
                "out_dir": str(out_dir),
            }
        )

    target_counts = Counter(j["target_seq"] for j in jobs)
    print(f"Parsed {len(jobs)} designs; {len(parse_errors)} parse failures")
    print(f"Unique target sequences: {len(target_counts)}")
    if len(target_counts) > 1:
        print("WARNING: target sequences differ across designs:")
        for seq, n in target_counts.most_common():
            print(
                f"  n={n} len={len(seq.split(':')[0])} "
                f"hash={hashlib.md5(seq.encode()).hexdigest()[:8]}"
            )

    binder_seqs = [j["binder_seq"] for j in jobs]
    n_unique_binders = len(set(binder_seqs))
    print(f"Unique binder sequences: {n_unique_binders}/{len(jobs)}")
    near = []
    for i, a in enumerate(binder_seqs):
        for j, b in enumerate(binder_seqs[i + 1 :], start=i + 1):
            ident = seq_identity(a, b)
            if ident >= 0.9 and a != b:
                near.append((jobs[i]["name"], jobs[j]["name"], round(ident, 3)))
    if near:
        print(f"Near-duplicate binders (identity >= 0.9): {near[:20]}")

    seq_to_msa_path: dict[str, str] = {}
    if use_msa and jobs:
        _disable_ssl_verify()
        unique_seqs = sorted(
            {
                seq
                for job in jobs
                for seq in (job.get("target_chain_seqs") or {}).values()
            }
        )
        msa_dir = out_dir / "msas"
        msa_dir.mkdir(parents=True, exist_ok=True)
        print(f"Fetching ColabFold MSAs for {len(unique_seqs)} unique target chain(s)...")
        for seq in unique_seqs:
            digest = hashlib.md5(seq.encode()).hexdigest()
            dest = msa_dir / f"{digest}.a3m"
            try:
                dest.write_text(fetch_colabfold_msa(seq))
                seq_to_msa_path[seq] = str(dest)
                print(f"  MSA ok len={len(seq)} hash={digest[:8]}")
            except Exception as exc:
                print(f"  MSA failed len={len(seq)} ({exc}); using empty MSA")
        OUT_VOLUME.commit()

    for job in jobs:
        paths = {}
        for cid, seq in (job.get("target_chain_seqs") or {}).items():
            if seq in seq_to_msa_path:
                paths[cid] = seq_to_msa_path[seq]
        job["target_msa_paths"] = paths
        job.pop("target_chain_seqs", None)

    seq_csv = out_dir / "sequences.csv"
    if seq_rows:
        with seq_csv.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(seq_rows[0].keys()))
            w.writeheader()
            w.writerows(seq_rows)

    OUT_VOLUME.commit()

    if not jobs:
        raise RuntimeError(f"No valid designs to predict. Parse errors: {parse_errors}")

    print(
        f"Running AlphaFold 3 on {len(jobs)} design(s) "
        f"({'complex+monomer' if run_monomer else 'complex only'})..."
    )
    results = list(validate_one.map(jobs, order_outputs=True))
    for err in parse_errors:
        results.append(
            {
                "design": err["design"],
                "error": err["error"],
                "binder_seq": "",
                "target_seq": "",
            }
        )

    for r in results:
        ident_max = 0.0
        seq = r.get("binder_seq") or ""
        for other in results:
            if other.get("design") == r.get("design"):
                continue
            ident_max = max(ident_max, seq_identity(seq, other.get("binder_seq") or ""))
        r["max_binder_seq_identity"] = round(ident_max, 4) if seq else None

        iptm = r.get("binder_target_iptm") or r.get("iptm")
        plddt = r.get("complex_plddt") or r.get("binder_plddt_complex")
        rmsd = r.get("rmsd_binder_on_target")
        if iptm is not None and plddt is not None:
            rmsd_term = 1.0 / (1.0 + (rmsd if rmsd is not None else 0.0))
            r["rank_score"] = float(iptm) * float(plddt) * rmsd_term
        else:
            r["rank_score"] = None

        r["passes_filters"] = bool(
            not r.get("error")
            and iptm is not None
            and iptm >= 0.5
            and plddt is not None
            and plddt >= 0.7
            and (rmsd is None or rmsd <= 4.0)
        )

    ranked = sorted(
        results,
        key=lambda r: (
            1 if r.get("error") else 0,
            -(r.get("rank_score") if r.get("rank_score") is not None else -1.0),
            -(
                r.get("binder_target_iptm")
                if r.get("binder_target_iptm") is not None
                else (r.get("iptm") if r.get("iptm") is not None else -1.0)
            ),
            r.get("rmsd_binder_on_target")
            if r.get("rmsd_binder_on_target") is not None
            else 99.0,
        ),
    )
    for i, r in enumerate(ranked, start=1):
        r["rank"] = i if not r.get("error") else ""

    fieldnames = [
        "rank",
        "design",
        "rank_score",
        "passes_filters",
        "binder_target_iptm",
        "iptm",
        "ptm",
        "ranking_score",
        "complex_plddt",
        "binder_plddt_complex",
        "target_plddt_complex",
        "binder_ptm",
        "interface_pae_mean",
        "interface_pae_min",
        "chain_pair_pae_min",
        "fraction_disordered",
        "has_clash",
        "rmsd_binder_on_target",
        "tm_binder_on_target",
        "rmsd_complex",
        "tm_complex",
        "rmsd_binder_fold",
        "tm_binder_fold",
        "monomer_plddt",
        "monomer_ptm",
        "monomer_ranking_score",
        "rmsd_binder_monomer",
        "tm_binder_monomer",
        "delta_plddt_binder",
        "binder_len",
        "target_len",
        "binder_chain",
        "target_chains",
        "max_binder_seq_identity",
        "binder_seq",
        "target_seq",
        "error",
    ]
    extra = []
    for r in ranked:
        for k in r:
            if k not in fieldnames:
                extra.append(k)
    fieldnames.extend(sorted(set(extra)))

    csv_path = out_dir / "rankings.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(ranked)

    meta = {
        "run_name": run_name,
        "n_designs": len(jobs),
        "n_succeeded": sum(1 for r in ranked if not r.get("error")),
        "n_failed": sum(1 for r in ranked if r.get("error")),
        "unique_targets": len(target_counts),
        "unique_binders": n_unique_binders,
        "near_duplicates": near,
        "params_str": params_str,
        "use_msa": use_msa,
        "run_monomer": run_monomer,
        "seed": seed,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    }
    (out_dir / "summary.json").write_text(
        __import__("json").dumps(meta, indent=2, default=str)
    )
    OUT_VOLUME.commit()

    csv_text = csv_path.read_text()
    print(f"Wrote {csv_path} ({meta['n_succeeded']}/{meta['n_designs']} succeeded)")
    return {
        "volume": OUT_VOLUME_NAME,
        "run_name": run_name,
        "csv": csv_text,
        "summary": meta,
    }


@app.local_entrypoint()
def main(
    input_dir: str | None = None,
    volume_name: str | None = None,
    binder_chain: str = "B",
    target_chains: str = "A",
    target_pdb: str | None = None,
    run_name: str | None = None,
    params_str: str | None = None,
    msa: bool = True,
    monomer: bool = True,
    recursive: bool = False,
    recycling_steps: int = 10,
    diffusion_samples: int = 1,
    seed: int = 1,
    out_dir: str = "./out/alphafast_validate",
    upload_weights: str | None = None,
):
    """Validate binder designs with AlphaFast / AlphaFold 3.

    Args:
        input_dir: Folder of PDB/CIF designs. Path on the volume if
            --volume-name is set, otherwise a local folder.
        volume_name: Modal Volume that holds the designs (bindcraft,
            proteinhunter, or DESIGN_VOLUME).
        binder_chain: Binder chain ID in each design (default B).
        target_chains: Comma-separated target chain IDs (default A).
        target_pdb: Target structure if designs are binder-only.
        run_name: Output subdirectory on the alphafast-validate volume.
        params_str: Extra flags for `run_alphafold.py` (overrides constructed flags).
        msa: Fetch a ColabFold MSA for unique target sequences (default True).
            Pass --no-msa for single-sequence mode on every chain.
        monomer: Also predict the binder alone (default True). Pass --no-monomer to skip.
        recursive: Recurse into subfolders (skips BindCraft Ranked/).
        recycling_steps: AF3 recycles (10 is the official default).
        diffusion_samples: AF3 diffusion samples (1 default; 5 is the paper default).
        seed: Model seed written into the AF3 input JSON.
        out_dir: Local directory for the downloaded rankings.csv.
        upload_weights: Local path to `af3.bin.zst` (or `af3.bin`). Streams the
            file onto the AlphaFast `af3-weights` volume and exits.
    """
    from datetime import datetime

    if upload_weights:
        path = Path(upload_weights).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        print(f"Streaming {path} ({path.stat().st_size / 1e9:.2f} GB) "
              f"to Modal Volume '{MODEL_VOLUME_NAME}' as /{path.name} ...")
        with MODEL_VOLUME.batch_upload() as batch:
            batch.put_file(path, f"/{path.name}")
        print("Upload complete. You can now run validation.")
        print(
            "Same volume AlphaFast uses: "
            "https://github.com/RomeroLab/alphafast#modal-setup"
        )
        return

    if not input_dir:
        raise ValueError("Provide --input-dir (local folder or path on --volume-name)")

    today = datetime.now().strftime("%Y%m%d%H%M")[2:]
    run_name = run_name or today
    do_msa = msa
    do_monomer = monomer

    if params_str is None:
        params_str = (
            f"--num_recycles {recycling_steps} "
            f"--num_diffusion_samples {diffusion_samples}"
        )

    known = set(KNOWN_VOLUME_NAMES)
    if EXTRA_VOLUME_NAME:
        known.add(EXTRA_VOLUME_NAME)
    if volume_name and volume_name not in known:
        raise ValueError(
            f"Volume '{volume_name}' is not mounted. Known: {sorted(known)}. "
            f"Re-run with DESIGN_VOLUME={volume_name} so the volume can be attached."
        )

    target_content = None
    target_filename = None
    target_file_chains = _chain_ids(target_chains)
    if target_pdb:
        tpath = Path(target_pdb)
        if tpath.exists():
            target_content = tpath.read_text()
            target_filename = tpath.name

    designs = None
    if volume_name is None:
        root = Path(input_dir)
        if not root.exists():
            raise FileNotFoundError(f"Local input-dir not found: {root}")
        files = _iter_structure_files(root, recursive)
        if not files:
            raise FileNotFoundError(f"No PDB/CIF files in {root}")
        designs = [
            {"name": f.stem, "filename": f.name, "content": f.read_text()}
            for f in files
        ]
        print(f"Loaded {len(designs)} local structure(s) from {root}")
    else:
        print(f"Reading designs from volume '{volume_name}' at {input_dir}")

    print(f"AF3 weights volume: '{MODEL_VOLUME_NAME}' (mounted at {MODEL_MOUNT})")
    print(
        "  Upload once: uv run --with modal modal run modal_alphafast_validate.py "
        "--upload-weights /path/to/af3.bin.zst"
    )
    print(f"Results → volume '{OUT_VOLUME_NAME}' / {run_name}")
    print("Run with --detach for long batches; the job keeps going after disconnect.")
    print(
        f"Download later: modal volume get {OUT_VOLUME_NAME} {run_name} {out_dir}/"
    )

    result = run_validation.spawn(
        volume_name=volume_name,
        input_dir=input_dir,
        designs=designs,
        binder_chain=binder_chain,
        target_chains=target_chains,
        target_filename=target_filename,
        target_content=target_content,
        target_file_chains=target_file_chains,
        target_pdb_path=target_pdb,
        run_name=run_name,
        params_str=params_str,
        use_msa=do_msa,
        run_monomer=do_monomer,
        recursive=recursive,
        seed=seed,
    ).get()

    local = Path(out_dir) / run_name
    local.mkdir(parents=True, exist_ok=True)
    (local / "rankings.csv").write_text(result["csv"])
    print(f"Copied rankings.csv to {local / 'rankings.csv'}")
    print(f"Summary: {result['summary']}")
    print(
        f"AlphaFast validate finished: volume={result['volume']} "
        f"run={result['run_name']}"
    )
