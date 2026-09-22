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

From BindCraft2 ranked complexes:

    GPU=A100-80GB uv run --with modal modal run --detach modal_alphafast_validate.py \\
      --volume-name bindcraft2 --input-dir <run>/<campaign>/3_Ranked \\
      --binder-chain B --target-chains A

From RFD3 + SolubleMPNN (binder A, target B or B,C). Motif RMSD is taken
from each design's `diffused_index_map`:

    GPU=A100-80GB uv run --with modal modal run --detach modal_alphafast_validate.py \\
      --volume-name rfd3 --input-dir <run>/mpnn --recursive \\
      --binder-chain A --target-chains B,C

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

KNOWN_VOLUME_NAMES = ("bindcraft", "bindcraft2", "proteinhunter", "rfd3")
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


def _design_stem(path: Path) -> str:
    """Stem of a structure file, stripping a trailing .gz if present."""
    name = path.name
    if name.lower().endswith(".gz"):
        name = name[:-3]
    return Path(name).stem


def _is_structure_file(path: Path) -> bool:
    name = path.name.lower()
    if name.endswith(".gz"):
        name = name[:-3]
    return Path(name).suffix.lower() in STRUCTURE_SUFFIXES


def _read_file_text(path: Path) -> tuple[str, str]:
    """Return (filename_for_parser, text). Decompresses .gz CIF/PDB."""
    import gzip

    name = path.name
    data = path.read_bytes()
    if name.lower().endswith(".gz"):
        data = gzip.decompress(data)
        name = name[:-3]
    return name, data.decode("utf-8", errors="replace")


def _parse_res_id_list(text: str | None) -> list[str]:
    if not text:
        return []
    return [p.strip() for p in text.replace(";", ",").split(",") if p.strip()]


def _res_id_chain(token: str) -> str:
    import re

    m = re.match(r"^([A-Za-z]+)", token.strip())
    return m.group(1) if m else ""


def _motif_from_index_map(
    index_map: dict, binder_chain: str, target_chains: list[str]
) -> list[str]:
    """Binder-chain output IDs from RFD3 ``diffused_index_map`` values."""
    seen: set[str] = set()
    motif: list[str] = []
    targets = set(target_chains)
    for out_id in (index_map or {}).values():
        if out_id is None:
            continue
        token = str(out_id).strip()
        if not token or token in seen:
            continue
        chain = _res_id_chain(token)
        if binder_chain and chain != binder_chain:
            continue
        if chain in targets:
            continue
        seen.add(token)
        motif.append(token)
    return motif


def _find_rfd3_json(structure: Path) -> Path | None:
    """Locate the RFD3 metadata JSON for a design or MPNN CIF."""
    import json

    stem = _design_stem(structure)
    stems = [stem]
    s = stem
    while "_" in s:
        s = s.rsplit("_", 1)[0]
        stems.append(s)
    dirs: list[Path] = []
    parent = structure.parent
    for _ in range(4):
        dirs.append(parent)
        if parent.parent == parent:
            break
        parent = parent.parent
    seen: set[Path] = set()
    for d in dirs:
        for st in stems:
            cand = d / f"{st}.json"
            if cand in seen or not cand.is_file():
                continue
            seen.add(cand)
            try:
                meta = json.loads(cand.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(meta, dict) and "diffused_index_map" in meta:
                return cand
    return None


def _structure_for_json(json_path: Path) -> Path | None:
    stem = json_path.with_suffix("")
    for ext in (".cif.gz", ".cif", ".pdb"):
        candidate = Path(str(stem) + ext)
        if candidate.is_file():
            return candidate
    return None


def _resolve_existing_file(path: str | None, volume_name: str | None) -> Path | None:
    if not path:
        return None
    candidates = [Path(path)]
    if volume_name:
        candidates.append(Path(f"/vol/{volume_name}") / path)
    for p in candidates:
        if p.is_file():
            return p
    return None


def _motif_residues_for_design(
    source_path: str | None,
    binder_chain: str,
    target_chains: list[str],
    motif_residues: list[str],
    motif_json: str | None,
    volume_name: str | None = None,
) -> list[str]:
    import json

    if motif_residues:
        return list(motif_residues)
    json_path = _resolve_existing_file(motif_json, volume_name)
    if json_path is not None:
        try:
            meta = json.loads(json_path.read_text())
            imap = meta.get("diffused_index_map") or {}
            if isinstance(imap, dict):
                return _motif_from_index_map(imap, binder_chain, target_chains)
        except (OSError, json.JSONDecodeError):
            pass
    if source_path:
        found = _find_rfd3_json(Path(source_path))
        if found is not None:
            try:
                meta = json.loads(found.read_text())
                imap = meta.get("diffused_index_map") or {}
                if isinstance(imap, dict):
                    ids = _motif_from_index_map(imap, binder_chain, target_chains)
                    print(f"Motif from {found.name}: {len(ids)} residues")
                    return ids
            except (OSError, json.JSONDecodeError):
                pass
    return []


def _attach_rfd3_sidecar(item: dict, source: Path) -> None:
    """Pack the parent RFD3 backbone so motif indices survive a local upload."""
    json_path = _find_rfd3_json(source)
    if json_path is None:
        return
    struct = _structure_for_json(json_path)
    if struct is None:
        return
    try:
        if struct.resolve() == source.resolve():
            return
    except OSError:
        return
    try:
        filename, content = _read_file_text(struct)
    except OSError:
        return
    item["rfd3_filename"] = filename
    item["rfd3_content"] = content


def _indices_in_chains(
    chains: dict[str, dict], binder: str, motif_ids: set[str]
) -> list[int]:
    ids = (chains.get(binder) or {}).get("res_ids") or []
    return [i for i, rid in enumerate(ids) if rid in motif_ids]


def _motif_indices(
    ref_chains: dict[str, dict],
    binder: str,
    motif_ids: list[str],
    source_path: str | None = None,
    rfd3_filename: str | None = None,
    rfd3_content: str | None = None,
) -> list[int]:
    ids = set(motif_ids or [])
    if not ids:
        return []
    idxs = _indices_in_chains(ref_chains, binder, ids)
    if len(idxs) >= 3:
        return idxs
    if rfd3_filename and rfd3_content:
        try:
            orig = extract_chains(rfd3_filename, rfd3_content)
            idxs2 = _indices_in_chains(orig, binder, ids)
            if len(idxs2) >= 3:
                return idxs2
        except Exception:
            pass
    if source_path:
        json_path = _find_rfd3_json(Path(source_path))
        struct = _structure_for_json(json_path) if json_path else None
        if struct is not None:
            try:
                filename, content = _read_file_text(struct)
                orig = extract_chains(filename, content)
                idxs2 = _indices_in_chains(orig, binder, ids)
                if len(idxs2) >= 3:
                    return idxs2
            except Exception:
                pass
    return idxs


def _residue_id(chain_name: str, residue) -> str:
    seqid = residue.seqid
    num = int(seqid.num)
    icode = (seqid.icode or "").strip()
    return f"{chain_name}{num}{icode}"


def _read_structure(name: str, content: str):
    """Parse PDB or mmCIF text into a gemmi Structure."""
    import gemmi

    lower = name.lower()
    if lower.endswith(".gz"):
        lower = lower[:-3]
    if lower.endswith((".cif", ".mmcif")):
        doc = gemmi.cif.read_string(content)
        return gemmi.make_structure_from_block(doc.sole_block())
    return gemmi.read_pdb_string(content)


def extract_chains(name: str, content: str) -> dict[str, dict]:
    """Extract per-chain protein sequence and CA coordinates from a structure.

    Returns:
        Mapping of chain ID to dict with keys sequence (str), ca (N,3 float array),
        plddt (N float array from B-factors), res_ids (list of e.g. A16).
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
        res_ids: list[str] = []
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
            res_ids.append(_residue_id(chain.name, residue))
        if seq:
            chains[chain.name] = {
                "sequence": "".join(seq),
                "ca": np.asarray(ca, dtype=float),
                "plddt": np.asarray(plddt, dtype=float),
                "res_ids": res_ids,
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


def _motif_ca_pair(
    pred_chains: dict[str, dict],
    ref_chains: dict[str, dict],
    motif_ids: set[str],
    binder: str,
    motif_indices: list[int] | None = None,
):
    """Matched motif CA coordinates in reference residue order.

    Prefer residue-ID matches on the design. If the design was renumbered
    (common for some MPNN writers), fall back to sequential binder indices
    taken from the parent RFD3 backbone.
    """
    import numpy as np

    ps, rs = [], []
    if motif_ids:
        for cid, ref in ref_chains.items():
            if cid not in pred_chains:
                continue
            pred = pred_chains[cid]
            n = min(len(pred["ca"]), len(ref["ca"]))
            ids = ref.get("res_ids") or []
            for i, rid in enumerate(ids):
                if i >= n:
                    break
                if rid in motif_ids:
                    ps.append(pred["ca"][i])
                    rs.append(ref["ca"][i])
    if len(ps) < 3 and motif_indices and binder in pred_chains and binder in ref_chains:
        pred = pred_chains[binder]
        ref = ref_chains[binder]
        n = min(len(pred["ca"]), len(ref["ca"]))
        ps, rs = [], []
        for i in motif_indices:
            if i < n:
                ps.append(pred["ca"][i])
                rs.append(ref["ca"][i])
    if len(ps) < 3:
        return None, None
    return np.asarray(ps, dtype=float), np.asarray(rs, dtype=float)


def compute_rmsds(
    pred_chains: dict[str, dict],
    ref_chains: dict[str, dict],
    binder: str,
    targets: list[str],
    motif_res_ids: list[str] | None = None,
    motif_indices: list[int] | None = None,
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
        "rmsd_motif": None,
        "tm_motif": None,
        "rmsd_motif_on_target": None,
        "tm_motif_on_target": None,
        "n_motif_residues": 0,
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
    R_tgt, t_tgt = None, None
    if pred_tgt is not None and pb is not None:
        R_tgt, t_tgt, _, _ = _kabsch(pred_tgt, ref_tgt)
        if R_tgt is not None:
            aligned_b = _apply_rt(pb, R_tgt, t_tgt)
            out["rmsd_binder_on_target"] = float(
                np.sqrt(((aligned_b - rb) ** 2).sum(axis=1).mean())
            )
            out["tm_binder_on_target"] = _tm_score(aligned_b, rb)

    motif_ids = set(motif_res_ids or [])
    if motif_ids or motif_indices:
        pm, rm = _motif_ca_pair(
            pred_chains, ref_chains, motif_ids, binder, motif_indices
        )
        if pm is not None:
            out["n_motif_residues"] = len(pm)
            _, _, rmsd, aligned = _kabsch(pm, rm)
            out["rmsd_motif"] = rmsd
            out["tm_motif"] = _tm_score(aligned, rm) if aligned is not None else None
            if R_tgt is not None:
                aligned_m = _apply_rt(pm, R_tgt, t_tgt)
                out["rmsd_motif_on_target"] = float(
                    np.sqrt(((aligned_m - rm) ** 2).sum(axis=1).mean())
                )
                out["tm_motif_on_target"] = _tm_score(aligned_m, rm)
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


def _empty_protein(
    chain_id: str,
    sequence: str,
    unpaired_msa: str | None,
    templates: list | None = None,
) -> dict:
    """AF3 protein entity. Empty MSA skips the data pipeline."""
    msa = unpaired_msa if unpaired_msa is not None else ""
    return {
        "protein": {
            "id": chain_id,
            "sequence": sequence,
            "unpairedMsa": msa,
            "pairedMsa": "",
            "templates": templates if templates is not None else [],
        }
    }


def _cif_word(value: object) -> str:
    """Quote a value for mmCIF if it is not a simple token."""
    s = str(value).strip()
    if not s:
        return "."
    if s[0] not in "'\"#$_;[" and all(c not in " \t\n" for c in s):
        return s
    if "'" not in s:
        return f"'{s}'"
    if '"' not in s:
        return f'"{s}"'
    return "'" + s.replace("'", '"') + "'"


def _protein_chain_mmcif(name: str, content: str, chain_id: str) -> tuple[str, int] | None:
    """Single-chain protein mmCIF for one AF3 template (exactly one polymer chain).

    Writes a minimal `_atom_site` table only. gemmi's full mmCIF dumps extra
    tables (`_pdbx_poly_seq_scheme`, `_cell`, …) that often contain `.` in
    integer columns, which AF3 then fails to parse (`int('.')`).
    """
    st = _read_structure(name, content)
    if len(st) == 0:
        return None
    try:
        st.merge_chain_parts()
    except Exception:
        pass

    src = None
    for chain in st[0]:
        if chain.name == chain_id:
            src = chain
            break
    if src is None:
        return None

    rows: list[str] = []
    atom_id = 0
    n_res = 0
    for residue in src:
        if AA3TO1.get(residue.name) is None:
            continue
        if residue.find_atom("CA", "*") is None:
            continue
        n_res += 1
        seen: set[str] = set()
        for atom in residue:
            atom_name = (atom.name or "").strip()
            if not atom_name or atom_name in seen:
                continue
            alt = getattr(atom, "altloc", None)
            if isinstance(alt, str) and alt not in ("", " ", "A", "\x00"):
                continue
            elem = ""
            try:
                elem = (atom.element.name or "").strip()
            except Exception:
                elem = ""
            if not elem:
                elem = atom_name[0]
            if elem.upper() in {"H", "D"}:
                continue
            seen.add(atom_name)
            atom_id += 1
            occ = float(getattr(atom, "occ", 1.0) or 1.0)
            bfac = float(getattr(atom, "b_iso", 0.0) or 0.0)
            rows.append(
                " ".join(
                    [
                        "ATOM",
                        str(atom_id),
                        _cif_word(elem),
                        _cif_word(atom_name),
                        ".",
                        _cif_word(residue.name),
                        "A",
                        "1",
                        str(n_res),
                        "?",
                        f"{atom.pos.x:.3f}",
                        f"{atom.pos.y:.3f}",
                        f"{atom.pos.z:.3f}",
                        f"{occ:.2f}",
                        f"{bfac:.2f}",
                        "?",
                        str(n_res),
                        _cif_word(residue.name),
                        "A",
                        _cif_word(atom_name),
                        "1",
                    ]
                )
            )
    if n_res < 3 or not rows:
        return None

    block = f"tmpl{chain_id}"
    cif = "\n".join(
        [
            f"data_{block}",
            "#",
            f"_entry.id {block}",
            "#",
            "# Date before AF3's default max_template_date (2021-09-30).",
            "_pdbx_audit_revision_history.revision_date 2021-01-01",
            "#",
            "loop_",
            "_atom_site.group_PDB",
            "_atom_site.id",
            "_atom_site.type_symbol",
            "_atom_site.label_atom_id",
            "_atom_site.label_alt_id",
            "_atom_site.label_comp_id",
            "_atom_site.label_asym_id",
            "_atom_site.label_entity_id",
            "_atom_site.label_seq_id",
            "_atom_site.pdbx_PDB_ins_code",
            "_atom_site.Cartn_x",
            "_atom_site.Cartn_y",
            "_atom_site.Cartn_z",
            "_atom_site.occupancy",
            "_atom_site.B_iso_or_equiv",
            "_atom_site.pdbx_formal_charge",
            "_atom_site.auth_seq_id",
            "_atom_site.auth_comp_id",
            "_atom_site.auth_asym_id",
            "_atom_site.auth_atom_id",
            "_atom_site.pdbx_PDB_model_num",
            *rows,
            "#",
            "",
        ]
    )
    return cif, n_res


def _target_template(
    name: str, content: str, chain_id: str, query_seq: str
) -> dict | None:
    """AF3 template dict mapping the query 1:1 onto residues in the design PDB."""
    packed = _protein_chain_mmcif(name, content, chain_id)
    if packed is None:
        return None
    mmcif, n_res = packed
    n = min(n_res, len(query_seq))
    if n < 3:
        return None
    return {
        "mmcif": mmcif,
        "queryIndices": list(range(n)),
        "templateIndices": list(range(n)),
    }


def build_af3_json(
    name: str,
    target_seqs: list[tuple[str, str]],
    binder_id: str,
    binder_seq: str,
    seed: int,
    target_msas: dict[str, str] | None,
    target_templates: dict[str, dict] | None = None,
) -> dict:
    """AF3 input JSON: MSA/templates on targets only; empty MSA, no template on binder."""
    sequences = []
    for chain_id, seq in target_seqs:
        msa = (target_msas or {}).get(chain_id) or ""
        tmpl = (target_templates or {}).get(chain_id)
        sequences.append(
            _empty_protein(chain_id, seq, msa, templates=[tmpl] if tmpl else [])
        )
    sequences.append(_empty_protein(binder_id, binder_seq, "", templates=[]))
    return {
        "name": _af3_job_name(name),
        "sequences": sequences,
        "modelSeeds": [int(seed)],
        "dialect": "alphafold3",
        "version": 1,
    }


def _parse_a3m_records(text: str) -> list[tuple[str, str]]:
    """Parse A3M/FASTA into (header, sequence) pairs; skip comment lines."""
    records: list[tuple[str, str]] = []
    header = None
    seq_parts: list[str] = []
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        if line.startswith(">"):
            if header is not None:
                s = "".join(seq_parts).replace("\x00", "")
                if s:
                    records.append((header, s))
            header = line
            seq_parts = []
        elif header is not None:
            seq_parts.append(line.strip())
    if header is not None:
        s = "".join(seq_parts).replace("\x00", "")
        if s:
            records.append((header, s))
    return records


def _ungapped_upper(seq: str) -> str:
    return "".join(c for c in seq if c.isalpha() and c.isupper())


def _a3m_from_records(
    query_seq: str,
    records: list[tuple[str, str]],
    *,
    dedupe_headers: bool = True,
) -> str:
    lines = [">query", query_seq]
    q = query_seq.upper()
    seen: set[str] = set()
    for header, seq in records:
        raw = _ungapped_upper(seq)
        if raw == q:
            continue
        if dedupe_headers:
            key = header.split()[0]
            if key in seen:
                continue
            seen.add(key)
        lines.append(header if header.startswith(">") else f">{header}")
        lines.append(seq)
    return "\n".join(lines) + "\n"


def _a3m_from_tar(tar_bytes: bytes, name_substr: str | None) -> str:
    """Concatenate A3M members from a ColabFold tar.gz. name_substr filters names."""
    import io
    import tarfile

    texts: list[str] = []
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
        for member in tar.getmembers():
            name = member.name.lower()
            if not name.endswith(".a3m"):
                continue
            if name_substr is None:
                if "pair" in name:
                    continue
            elif name_substr not in name:
                continue
            fh = tar.extractfile(member)
            if fh is None:
                continue
            texts.append(fh.read().decode("utf-8", errors="replace"))
    return "\n".join(texts)


def _a3m_from_colabfold_tar(tar_bytes: bytes, query_seq: str) -> str:
    """Merge ColabFold unpaired A3M files into a single AF3 unpairedMsa string."""
    return _a3m_from_records(query_seq, _parse_a3m_records(_a3m_from_tar(tar_bytes, None)))


def _split_concatenated_a3m_seq(seq: str, lengths: list[int]) -> list[str] | None:
    """Split a concatenated pair.a3m row into per-query A3M slices."""
    parts: list[str] = []
    i = 0
    n = len(seq)
    for L in lengths:
        start = i
        counted = 0
        while counted < L:
            if i >= n:
                return None
            c = seq[i]
            i += 1
            if c in "-." or (c.isalpha() and c.isupper()):
                counted += 1
        parts.append(seq[start:i])
    if i < n and parts:
        parts[-1] += seq[i:]
    return parts


def _split_pair_a3m(pair_text: str, unique_seqs: list[str]) -> dict[str, str]:
    """Split ColabFold pair.a3m (concatenated unique queries) into per-sequence A3Ms."""
    lengths = [len(s) for s in unique_seqs]
    per_chain: list[list[tuple[str, str]]] = [[] for _ in unique_seqs]
    for header, seq in _parse_a3m_records(pair_text):
        parts = _split_concatenated_a3m_seq(seq, lengths)
        if parts is None:
            continue
        for i, part in enumerate(parts):
            per_chain[i].append((header, part))
    return {
        seq: _a3m_from_records(seq, recs)
        for seq, recs in zip(unique_seqs, per_chain)
        if recs
    }


def _combine_target_msas(
    chain_seqs: list[tuple[str, str]],
    unpaired_by_seq: dict[str, str],
    paired_by_seq: dict[str, str],
) -> dict[str, str]:
    """Row-align paired hits across target chains, then pad unpaired (AF3 manual pairing).

    Binder is not in chain_seqs, so it stays unpaired. pairedMsa in the JSON remains "".
    """
    if not paired_by_seq:
        return {
            cid: unpaired_by_seq[seq]
            for cid, seq in chain_seqs
            if seq in unpaired_by_seq
        }

    paired_recs = []
    unpaired_recs = []
    for cid, seq in chain_seqs:
        precs = _parse_a3m_records(paired_by_seq.get(seq) or "")
        if precs and _ungapped_upper(precs[0][1]) == seq.upper():
            precs = precs[1:]
        paired_recs.append(precs)
        urecs = _parse_a3m_records(unpaired_by_seq.get(seq) or "")
        if urecs and _ungapped_upper(urecs[0][1]) == seq.upper():
            urecs = urecs[1:]
        unpaired_recs.append(urecs)

    n_pair = min((len(r) for r in paired_recs), default=0)
    paired_recs = [r[:n_pair] for r in paired_recs]

    out: dict[str, str] = {}
    for i, (cid, seq) in enumerate(chain_seqs):
        records = list(paired_recs[i])
        for j, urecs in enumerate(unpaired_recs):
            if j == i:
                records.extend(urecs)
            else:
                gaps = "-" * len(seq)
                records.extend((f">gap_{j}_{k}", gaps) for k in range(len(urecs)))
        out[cid] = _a3m_from_records(seq, records, dedupe_headers=False)
    return out


def _colabfold_download(
    query: str,
    endpoint: str,
    mode: str,
    host: str = "https://api.colabfold.com",
) -> bytes:
    """Submit a ColabFold MMseqs2 ticket and return the result tar.gz bytes."""
    import random
    import time

    import requests

    headers = {"User-Agent": "biomodals-alphafast-validate/1.0"}

    def submit():
        res = requests.post(
            f"{host}/{endpoint}",
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
        raise RuntimeError(f"ColabFold {endpoint} submit failed: {out}")

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
            raise RuntimeError(f"ColabFold ticket {ticket} failed: {payload}")
        time.sleep(5 + random.random() * 5)

    raw = requests.get(
        f"{host}/result/download/{ticket}",
        timeout=120,
        headers=headers,
        verify=False,
    )
    raw.raise_for_status()
    return raw.content


def fetch_colabfold_msa(sequence: str) -> str:
    """A3M from the ColabFold MMseqs2 server (UniRef + environmental)."""
    query = f">101\n{sequence}\n"
    return _a3m_from_colabfold_tar(
        _colabfold_download(query, "ticket/msa", "env"), sequence
    )


def fetch_colabfold_paired_msas(sequences: list[str]) -> dict[str, str]:
    """Paired A3M per unique sequence from ColabFold /ticket/pair (greedy+env)."""
    unique: list[str] = []
    for seq in sequences:
        if seq not in unique:
            unique.append(seq)
    if len(unique) < 2:
        return {}
    query = "".join(f">{101 + i}\n{seq}\n" for i, seq in enumerate(unique))
    raw = _colabfold_download(query, "ticket/pair", "pairgreedy-env")
    pair_text = _a3m_from_tar(raw, "pair")
    if not pair_text.strip():
        raise RuntimeError("ColabFold pair ticket had no pair.a3m")
    split = _split_pair_a3m(pair_text, unique)
    if len(split) < 2:
        raise RuntimeError("Failed to split ColabFold pair.a3m into target chains")
    return split


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
        and _is_structure_file(f)
        and "Ranked" not in f.parts
    )


def _design_item(path: Path) -> dict:
    filename, content = _read_file_text(path)
    return {
        "name": _design_stem(path),
        "filename": filename,
        "content": content,
        "source_path": str(path),
    }


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
        if job.get("msa_paired") and "resolve_msa_overlaps" not in extra:
            extra = f"{extra} --resolve_msa_overlaps=false".strip()
        seed = int(job.get("seed") or 1)

        target_templates: dict[str, dict] = {}
        if job.get("use_templates", False):
            for chain_id, seq in target_seqs:
                src_name, src_content = job["filename"], job["content"]
                if chain_id not in ref_chains:
                    if not job.get("target_content"):
                        continue
                    src_name = job.get("target_filename") or "target.pdb"
                    src_content = job["target_content"]
                tmpl = _target_template(src_name, src_content, chain_id, seq)
                if tmpl:
                    target_templates[chain_id] = tmpl
            if target_templates:
                print(
                    f"Templates on target chains {sorted(target_templates)}; "
                    "binder is template-free"
                )
            else:
                print("WARNING: --templates is on but no target-chain templates were built")

        json_complex = build_af3_json(
            name,
            target_seqs,
            binder_json,
            job["binder_seq"],
            seed,
            target_msas,
            target_templates,
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
            pred_chains = {}
            if struct:
                filename, content = _read_file_text(struct)
                pred_chains = extract_chains(filename, content)
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
                row.update(
                    compute_rmsds(
                        pred_for_rmsd,
                        ref_chains,
                        binder,
                        targets,
                        job.get("motif_residues"),
                        job.get("motif_indices"),
                    )
                )

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
                    m_chains = extract_chains(*_read_file_text(m_struct))
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
    use_pair: bool,
    run_monomer: bool,
    recursive: bool,
    seed: int,
    use_templates: bool,
    motif_residues: str | None = None,
    motif_json: str | None = None,
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
            loaded.append(_design_item(f))

    t_chains = _chain_ids(target_chains)
    explicit_motif = _parse_res_id_list(motif_residues)
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
        if explicit_motif:
            motif_ids = list(explicit_motif)
        elif item.get("motif_residues"):
            motif_ids = list(item["motif_residues"])
        else:
            motif_ids = _motif_residues_for_design(
                item.get("source_path"),
                b_id,
                tgt_ids,
                [],
                motif_json,
                volume_name,
            )
        motif_idxs = _motif_indices(
            chains,
            b_id,
            motif_ids,
            item.get("source_path"),
            item.get("rfd3_filename"),
            item.get("rfd3_content"),
        )
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
                "use_pair": use_pair,
                "use_templates": use_templates,
                "run_monomer": run_monomer,
                "seed": seed,
                "motif_residues": motif_ids,
                "motif_indices": motif_idxs,
                "out_dir": str(out_dir),
            }
        )

    target_counts = Counter(j["target_seq"] for j in jobs)
    n_motif = sum(1 for j in jobs if j.get("motif_residues"))
    print(f"Parsed {len(jobs)} designs; {len(parse_errors)} parse failures")
    print(f"Motif residues resolved for {n_motif}/{len(jobs)} designs")
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
    pair_to_split: dict[tuple[str, ...], dict[str, str]] = {}
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
                print(f"  unpaired MSA ok len={len(seq)} hash={digest[:8]}")
            except Exception as exc:
                print(f"  unpaired MSA failed len={len(seq)} ({exc}); using empty MSA")

        if use_pair:
            pair_keys: list[tuple[str, ...]] = []
            seen_keys: set[tuple[str, ...]] = set()
            for job in jobs:
                key = tuple(sorted(set((job.get("target_chain_seqs") or {}).values())))
                if len(key) >= 2 and key not in seen_keys:
                    seen_keys.add(key)
                    pair_keys.append(key)
            if pair_keys:
                print(
                    f"Fetching ColabFold paired MSAs for {len(pair_keys)} "
                    "unique target-chain set(s) (binder excluded)..."
                )
            for seqs in pair_keys:
                digest = hashlib.md5("\n".join(seqs).encode()).hexdigest()
                try:
                    paired = fetch_colabfold_paired_msas(list(seqs))
                    pair_to_split[seqs] = paired
                    for seq, a3m in paired.items():
                        dest = (
                            msa_dir
                            / f"{digest}_{hashlib.md5(seq.encode()).hexdigest()[:8]}.pair.a3m"
                        )
                        dest.write_text(a3m)
                    print(
                        f"  paired MSA ok n_chains={len(seqs)} "
                        f"hash={digest[:8]} split={len(paired)}"
                    )
                except Exception as exc:
                    print(f"  paired MSA failed ({exc}); unpaired only")
        OUT_VOLUME.commit()

    combined_cache: dict[str, str] = {}
    for job in jobs:
        chain_seqs = list((job.get("target_chain_seqs") or {}).items())
        unpaired_by_seq = {
            seq: Path(seq_to_msa_path[seq]).read_text()
            for _, seq in chain_seqs
            if seq in seq_to_msa_path
        }
        pair_key = tuple(sorted({seq for _, seq in chain_seqs}))
        paired_by_seq = pair_to_split.get(pair_key, {})
        combined = _combine_target_msas(chain_seqs, unpaired_by_seq, paired_by_seq)
        paths = {}
        msa_dir = out_dir / "msas"
        msa_dir.mkdir(parents=True, exist_ok=True)
        for cid, a3m in combined.items():
            digest = hashlib.md5(a3m.encode()).hexdigest()
            cache_key = f"{cid}_{digest}"
            if cache_key not in combined_cache:
                dest = msa_dir / f"{cache_key}.combined.a3m"
                dest.write_text(a3m)
                combined_cache[cache_key] = str(dest)
            paths[cid] = combined_cache[cache_key]
        job["target_msa_paths"] = paths
        job["msa_paired"] = bool(paired_by_seq)
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
        "rmsd_motif",
        "tm_motif",
        "rmsd_motif_on_target",
        "tm_motif_on_target",
        "n_motif_residues",
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
        "use_pair": use_pair,
        "use_templates": use_templates,
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
    pair: bool = True,
    templates: bool = False,
    monomer: bool = True,
    recursive: bool = False,
    recycling_steps: int = 10,
    diffusion_samples: int = 1,
    seed: int = 1,
    out_dir: str = "./out/alphafast_validate",
    upload_weights: str | None = None,
    motif_residues: str | None = None,
    motif_json: str | None = None,
):
    """Validate binder designs with AlphaFast / AlphaFold 3.

    Args:
        input_dir: Folder of PDB/CIF designs. Path on the volume if
            --volume-name is set, otherwise a local folder.
        volume_name: Modal Volume that holds the designs (bindcraft,
            bindcraft2, proteinhunter, rfd3, or DESIGN_VOLUME).
        binder_chain: Binder chain ID in each design (default B).
        target_chains: Comma-separated target chain IDs (default A).
        target_pdb: Target structure if designs are binder-only.
        run_name: Output subdirectory on the alphafast-validate volume.
        params_str: Extra flags for `run_alphafold.py` (overrides constructed flags).
        msa: Fetch a ColabFold MSA for unique target sequences (default True).
            Pass --no-msa for single-sequence mode on every chain.
        pair: Also fetch a ColabFold paired MSA across distinct target chains
            (default True when MSA is on). The binder is never paired.
            Pass --no-pair to skip.
        templates: Use the design PDB (or --target-pdb) as a structural
            template for target chains only. Off by default; pass --templates.
            The binder is never templated.
        monomer: Also predict the binder alone (default True). Pass --no-monomer to skip.
        recursive: Recurse into subfolders (skips BindCraft Ranked/).
        recycling_steps: AF3 recycles (10 is the official default).
        diffusion_samples: AF3 diffusion samples (1 default; 5 is the paper default).
        seed: Model seed written into the AF3 input JSON.
        out_dir: Local directory for the downloaded rankings.csv.
        upload_weights: Local path to `af3.bin.zst` (or `af3.bin`). Streams the
            file onto the AlphaFast `af3-weights` volume and exits.
        motif_residues: Comma-separated motif residue IDs (e.g. A4,A6,A18)
            applied to every design. Omit to read each RFD3 JSON's
            diffused_index_map (recommended for motif scaffolding).
        motif_json: One RFD3 metadata JSON with diffused_index_map. Per-design
            sibling JSON next to the CIF (or parent of mpnn/) is used when omitted.
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
    do_pair = pair and do_msa
    do_templates = templates
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
        t_ids = _chain_ids(target_chains)
        explicit = _parse_res_id_list(motif_residues)
        designs = []
        for f in files:
            item = _design_item(f)
            item["motif_residues"] = _motif_residues_for_design(
                str(f), binder_chain, t_ids, explicit, motif_json
            )
            _attach_rfd3_sidecar(item, f)
            designs.append(item)
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
        use_pair=do_pair,
        run_monomer=do_monomer,
        recursive=recursive,
        seed=seed,
        use_templates=do_templates,
        motif_residues=motif_residues,
        motif_json=motif_json,
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
