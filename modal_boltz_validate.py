# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "modal>=1.0",
# ]
# ///
"""Validate designed protein binders with Boltz-2 (complex + monomer).

For each PDB/CIF in a folder (Modal Volume or local), extract sequences,
predict the binder–target complex and the binder alone, then rank designs
by interface confidence and RMSD to the designed coordinates.

Results go to the Modal Volume `boltz-validate`. Use `--detach` for batches.

From a BindCraft volume folder (chain A = target, B = binder):

    GPU=A100 uv run --with modal modal run --detach modal_boltz_validate.py \\
      --volume-name bindcraft --input-dir <run>/<target>/Accepted \\
      --binder-chain B --target-chains A

From BindCraft2 ranked complexes:

    GPU=A100 uv run --with modal modal run --detach modal_boltz_validate.py \\
      --volume-name bindcraft2 --input-dir <run>/<campaign>/3_Ranked \\
      --binder-chain B --target-chains A

From RFD3 + SolubleMPNN (binder A, target B or B,C). Motif RMSD is taken
from each design's `diffused_index_map`:

    GPU=A100 uv run --with modal modal run --detach modal_boltz_validate.py \\
      --volume-name rfd3 --input-dir <run>/mpnn --recursive \\
      --binder-chain A --target-chains B,C

From a local folder of complex structures:

    GPU=A100 uv run --with modal modal run modal_boltz_validate.py \\
      --input-dir ./designs --binder-chain B --target-chains A

Binder-only PDBs (target supplied separately):

    GPU=A100 uv run --with modal modal run modal_boltz_validate.py \\
      --input-dir ./binders --binder-chain A --target-pdb target.pdb --target-chains A

Download later:

    modal volume get boltz-validate <run_name> ./out/boltz_validate/

A custom input volume that is not bindcraft/bindcraft2/proteinhunter/rfd3:

    DESIGN_VOLUME=my-designs GPU=A100 uv run --with modal modal run \\
      modal_boltz_validate.py --volume-name my-designs --input-dir designs/
"""

from __future__ import annotations

import os
from pathlib import Path

from modal import App, Image, Volume

GPU = os.environ.get("GPU", "L40S")
TIMEOUT = int(os.environ.get("TIMEOUT", 60))
ORCH_TIMEOUT = int(os.environ.get("ORCH_TIMEOUT", 24)) * 60 * 60
MAX_CONTAINERS = int(os.environ.get("MAX_CONTAINERS", 8))
print(f"Using GPU {GPU}; TIMEOUT {TIMEOUT} min/design; MAX_CONTAINERS {MAX_CONTAINERS}")

BOLTZ_VOLUME_NAME = "boltz-models"
BOLTZ_MODEL_VOLUME = Volume.from_name(BOLTZ_VOLUME_NAME, create_if_missing=True)
CACHE_DIR = f"/{BOLTZ_VOLUME_NAME}"

KNOWN_VOLUME_NAMES = ("bindcraft", "bindcraft2", "proteinhunter", "rfd3")
EXTRA_VOLUME_NAME = os.environ.get("DESIGN_VOLUME")

OUT_VOLUME_NAME = "boltz-validate"
OUT_VOLUME = Volume.from_name(OUT_VOLUME_NAME, create_if_missing=True)
OUT_MOUNT = f"/{OUT_VOLUME_NAME}"

_volume_mounts: dict[str, Volume] = {
    CACHE_DIR: BOLTZ_MODEL_VOLUME,
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


def download_model():
    """Download Boltz-2 weights into the shared cache volume."""
    from boltz.main import download_boltz2

    if not Path(f"{CACHE_DIR}/boltz2_conf.ckpt").exists():
        print("downloading boltz 2")
        download_boltz2(Path(CACHE_DIR))


# Official install is `pip install boltz[cuda]`. Torch is pinned from the
# PyTorch CUDA index so pip does not pull a CPU wheel. No ColabFold / JAX:
# `--use_msa_server` talks to api.colabfold.com from inside Boltz.
image = (
    Image.debian_slim(python_version="3.11")
    .uv_pip_install("torch", index_url="https://download.pytorch.org/whl/cu126")
    .uv_pip_install("boltz[cuda]==2.2.1", "ipsae==1.0.1")
    .run_function(
        download_model,
        gpu="a10g",
        volumes={f"/{BOLTZ_VOLUME_NAME}": BOLTZ_MODEL_VOLUME},
    )
)

app = App("boltz-validate", image=image)


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


def _mmcif_block_with_atoms(doc):
    """Prefer the CIF block that actually has coordinate rows."""
    for block in doc:
        col = block.find_loop("_atom_site.id")
        if col is not None and col.get_loop().length() > 0:
            return block
    return doc.sole_block()


def _ensure_mmcif_atom_site_defaults(block) -> None:
    """Fill columns gemmi 0.6.x requires but many design CIFs omit.

    Design tools (and some ModelCIF writers) frequently skip
    ``_atom_site.occupancy`` and sometimes ``B_iso_or_equiv``. Without them
    ``gemmi.make_structure_from_block`` returns a Structure with 0 models.
    """
    col = block.find_loop("_atom_site.id")
    if col is None:
        return
    loop = col.get_loop()
    if loop.length() == 0:
        return
    tags = set(loop.tags)
    if "_atom_site.occupancy" not in tags:
        loop.add_columns(["_atom_site.occupancy"], "1")
    if "_atom_site.B_iso_or_equiv" not in tags:
        loop.add_columns(["_atom_site.B_iso_or_equiv"], "50")


def _read_structure(name: str, content: str):
    """Parse PDB or mmCIF text into a gemmi Structure."""
    import gemmi

    lower = name.lower()
    if lower.endswith(".gz"):
        lower = lower[:-3]
    if lower.endswith((".cif", ".mmcif")):
        doc = gemmi.cif.read_string(content)
        block = _mmcif_block_with_atoms(doc)
        _ensure_mmcif_atom_site_defaults(block)
        return gemmi.make_structure_from_block(block)
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
        raise ValueError(
            f"No models in {name} (mmCIF may be missing _atom_site rows "
            "or required columns like occupancy)"
        )
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
        # shortest chain = binder (common for minibinders)
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
    # Boltz stores pLDDT in [0, 1] or [0, 100] depending on writer
    return float(mean / 100.0) if mean > 1.5 else float(mean)


def _free_chain_id(taken: set[str]) -> str:
    for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        if c not in taken:
            return c
    raise ValueError(f"No free chain ID left; taken={taken}")


def build_yaml(
    target_seqs: list[tuple[str, str]],
    binder_id: str,
    binder_seq: str,
    use_msa: bool,
) -> str:
    """Boltz YAML: MSA server on target chains, single-sequence binder."""
    import yaml

    sequences = []
    for chain_id, seq in target_seqs:
        entity = {"protein": {"id": chain_id, "sequence": seq}}
        if not use_msa:
            entity["protein"]["msa"] = "empty"
        sequences.append(entity)
    sequences.append(
        {"protein": {"id": binder_id, "sequence": binder_seq, "msa": "empty"}}
    )
    return yaml.dump({"version": 1, "sequences": sequences}, sort_keys=False)


def parse_confidence(path: Path, binder_idx: int, target_idxs: list[int]) -> dict:
    """Pull pLDDT / iPTM and binder–target pair iPTM from a Boltz JSON."""
    import json

    data = json.loads(path.read_text())
    out = {
        "confidence_score": data.get("confidence_score"),
        "ptm": data.get("ptm"),
        "iptm": data.get("iptm"),
        "protein_iptm": data.get("protein_iptm"),
        "complex_plddt": data.get("complex_plddt"),
        "complex_iplddt": data.get("complex_iplddt"),
        "complex_pde": data.get("complex_pde"),
        "complex_ipde": data.get("complex_ipde"),
        "binder_target_iptm": None,
    }
    pairs = data.get("pair_chains_iptm") or {}
    vals = []
    for t in target_idxs:
        a = (pairs.get(str(binder_idx)) or {}).get(str(t))
        b = (pairs.get(str(t)) or {}).get(str(binder_idx))
        for v in (a, b):
            if v is not None:
                vals.append(float(v))
    if vals:
        out["binder_target_iptm"] = max(vals)
    chains_ptm = data.get("chains_ptm") or {}
    out["binder_ptm"] = chains_ptm.get(str(binder_idx))
    return out


def parse_interface_pae(
    pae_path: Path, chain_lengths: list[int], binder_idx: int, target_idxs: list[int]
) -> dict[str, float | None]:
    """Mean / min PAE between binder and target tokens (lower is better)."""
    import numpy as np

    pae = np.load(pae_path)
    if isinstance(pae, np.lib.npyio.NpzFile):
        key = "pae" if "pae" in pae.files else pae.files[0]
        pae = pae[key]
    bounds = [0]
    for n in chain_lengths:
        bounds.append(bounds[-1] + n)
    if bounds[-1] > pae.shape[0]:
        return {"interface_pae_mean": None, "interface_pae_min": None}
    b0, b1 = bounds[binder_idx], bounds[binder_idx + 1]
    blocks = []
    for t in target_idxs:
        t0, t1 = bounds[t], bounds[t + 1]
        blocks.append(pae[b0:b1, t0:t1])
        blocks.append(pae[t0:t1, b0:b1])
    if not blocks:
        return {"interface_pae_mean": None, "interface_pae_min": None}
    cat = np.concatenate([b.ravel() for b in blocks])
    return {
        "interface_pae_mean": float(cat.mean()),
        "interface_pae_min": float(cat.min()),
    }


def _pair_metric(nested: dict, chain_a: str, chain_b: str) -> float | None:
    """Read nested[chain_a][chain_b], tolerating missing keys."""
    try:
        val = nested[chain_a][chain_b]
    except (KeyError, TypeError):
        return None
    if val is None:
        return None
    return float(val)


def compute_ipsae(
    pae_path: Path,
    structure_path: Path,
    binder_chain: str,
    target_chains: list[str],
    output_dir: Path | None = None,
    pae_cutoff: float = 10.0,
    dist_cutoff: float = 10.0,
) -> dict[str, float | None]:
    """Run the PyPI ``ipsae`` package on a Boltz complex prediction.

    Uses the Boltz defaults (PAE/dist cutoff 10). Reports the best
    binder–target pair score (``ipSAE`` / d0res max), plus d0chn, d0dom,
    pDockQ2, and LIS.
    """
    from ipsae import calculate_ipsae

    out: dict[str, float | None] = {
        "ipsae": None,
        "ipsae_d0chn": None,
        "ipsae_d0dom": None,
        "ipsae_min": None,
        "pdockq": None,
        "pdockq2": None,
        "lis": None,
    }
    if output_dir is None:
        output_dir = structure_path.parent / "ipsae"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = calculate_ipsae(
        pae_path,
        structure_path,
        pae_cutoff=pae_cutoff,
        dist_cutoff=dist_cutoff,
        output_dir=output_dir,
    )
    ipsae_scores = results.get("ipsae_scores") or {}
    pdockq_scores = results.get("pdockq_scores") or {}
    lis_scores = results.get("lis_scores") or {}

    targets = [c for c in target_chains if c != binder_chain]
    if not targets:
        targets = [
            c
            for c in (results.get("unique_chains") or [])
            if str(c) != binder_chain
        ]
        targets = [str(c) for c in targets]

    ipsae_vals: list[float] = []
    d0chn_vals: list[float] = []
    d0dom_vals: list[float] = []
    pdockq_vals: list[float] = []
    pdockq2_vals: list[float] = []
    lis_vals: list[float] = []

    for t in targets:
        for a, b in ((binder_chain, t), (t, binder_chain)):
            for bucket, vals in (
                (ipsae_scores.get("ipsae_d0res_max"), ipsae_vals),
                (ipsae_scores.get("ipsae_d0chn_max"), d0chn_vals),
                (ipsae_scores.get("ipsae_d0dom_max"), d0dom_vals),
                (pdockq_scores.get("pDockQ"), pdockq_vals),
                (pdockq_scores.get("pDockQ2"), pdockq2_vals),
                (lis_scores, lis_vals),
            ):
                if bucket is None:
                    continue
                v = _pair_metric(bucket, a, b)
                if v is not None:
                    vals.append(v)

    if ipsae_vals:
        out["ipsae"] = max(ipsae_vals)
        out["ipsae_min"] = min(ipsae_vals)
    if d0chn_vals:
        out["ipsae_d0chn"] = max(d0chn_vals)
    if d0dom_vals:
        out["ipsae_d0dom"] = max(d0dom_vals)
    if pdockq_vals:
        out["pdockq"] = max(pdockq_vals)
    if pdockq2_vals:
        out["pdockq2"] = max(pdockq2_vals)
    if lis_vals:
        out["lis"] = max(lis_vals)
    out["ipsae_pae_cutoff"] = float(pae_cutoff)
    out["ipsae_dist_cutoff"] = float(dist_cutoff)
    return out


def _run_boltz(yaml_str: str, out_dir: Path, params: str) -> None:
    from subprocess import run

    in_dir = out_dir / "input"
    in_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = in_dir / "input.yaml"
    yaml_path.write_text(yaml_str)
    run(
        f'boltz predict "{yaml_path}"'
        f' --out_dir "{out_dir}"'
        f' --cache "{CACHE_DIR}"'
        f" {params}",
        shell=True,
        check=True,
    )


def _best_prediction(pred_root: Path) -> tuple[Path | None, Path | None, Path | None]:
    """Return (structure, confidence json, pae npz) for the best-ranked sample."""
    cifs = sorted(pred_root.glob("**/*_model_*.cif")) + sorted(
        pred_root.glob("**/*_model_*.pdb")
    )
    jsons = sorted(pred_root.glob("**/confidence_*.json"))
    paes = sorted(pred_root.glob("**/pae_*.npz"))
    struct = cifs[0] if cifs else None
    conf = jsons[0] if jsons else None
    pae = paes[0] if paes else None
    return struct, conf, pae


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


@app.function(
    timeout=TIMEOUT * 60,
    gpu=GPU,
    volumes=_volume_mounts,
    max_containers=MAX_CONTAINERS,
)
def validate_one(job: dict) -> dict:
    """Boltz complex + optional monomer prediction for one design."""
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
            # Predicted complex uses target-file chain IDs; design has no target coords.
            targets = t_ids

        if not target_seqs:
            raise ValueError(
                f"{name}: no target sequences (missing --target-chains or --target-pdb)"
            )

        taken = {c for c, _ in target_seqs}
        binder_yaml = binder if binder not in taken else _free_chain_id(taken)

        n_target = len(target_seqs)
        use_msa = bool(job.get("use_msa"))
        yaml_complex = build_yaml(
            target_seqs, binder_yaml, job["binder_seq"], use_msa
        )
        yaml_mono = build_yaml([], binder, job["binder_seq"], False)

        params = job["params_str"]
        if use_msa and "--use_msa_server" not in params:
            params = params + " --use_msa_server"

        binder_idx = n_target  # YAML: targets first, binder last
        target_idxs = list(range(n_target))
        chain_lengths = [len(s) for _, s in target_seqs] + [len(job["binder_seq"])]
        target_yaml_ids = [c for c, _ in target_seqs]

        with TemporaryDirectory() as td:
            td_path = Path(td)
            complex_dir = td_path / "complex"
            _run_boltz(yaml_complex, complex_dir, params)
            struct, conf, pae = _best_prediction(complex_dir)
            if conf:
                row.update(parse_confidence(conf, binder_idx, target_idxs))
            if pae:
                row.update(
                    parse_interface_pae(pae, chain_lengths, binder_idx, target_idxs)
                )
            if struct and pae:
                try:
                    ipsae_dir = Path(job["out_dir"]) / "predictions" / name / "ipsae"
                    row.update(
                        compute_ipsae(
                            pae,
                            struct,
                            binder_yaml,
                            target_yaml_ids,
                            output_dir=ipsae_dir,
                            pae_cutoff=float(job.get("ipsae_pae_cutoff", 10.0)),
                            dist_cutoff=float(job.get("ipsae_dist_cutoff", 10.0)),
                        )
                    )
                except Exception as exc:
                    print(f"ipSAE failed for {name}: {exc}")
                    row["ipsae_error"] = str(exc)
            pred_chains = {}
            if struct:
                filename, content = _read_file_text(struct)
                pred_chains = extract_chains(filename, content)
            if pred_chains:
                row["binder_plddt_complex"] = _mean_plddt(
                    pred_chains.get(binder_yaml)
                )
                tgt_plddts = [
                    p
                    for c in target_yaml_ids
                    if (p := _mean_plddt(pred_chains.get(c))) is not None
                ]
                row["target_plddt_complex"] = (
                    sum(tgt_plddts) / len(tgt_plddts) if tgt_plddts else None
                )
                pred_for_rmsd = dict(pred_chains)
                if binder_yaml != binder and binder_yaml in pred_for_rmsd:
                    pred_for_rmsd[binder] = pred_for_rmsd[binder_yaml]
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
                _run_boltz(
                    yaml_mono,
                    mono_dir,
                    params.replace("--write_full_pae", "").replace(
                        "--use_msa_server", ""
                    ),
                )
                m_struct, m_conf, _ = _best_prediction(mono_dir)
                if m_conf:
                    m = parse_confidence(m_conf, 0, [])
                    row["monomer_plddt"] = m.get("complex_plddt")
                    row["monomer_ptm"] = m.get("ptm")
                    row["monomer_confidence"] = m.get("confidence_score")
                if m_struct and binder in ref_chains:
                    m_chains = extract_chains(*_read_file_text(m_struct))
                    m_id = binder if binder in m_chains else next(iter(m_chains))
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
    motif_residues: str | None = None,
    motif_json: str | None = None,
    ipsae_pae_cutoff: float = 10.0,
    ipsae_dist_cutoff: float = 10.0,
) -> dict:
    """Load designs, run Boltz per design, write rankings.csv."""
    import csv
    import hashlib
    from collections import Counter
    from datetime import datetime

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

        # Binder-only: pull target sequences from the supplied target structure
        if (not tgt_ids or b_id in tgt_ids and len(chains) == 1) and target_content:
            tgt_struct = extract_chains(target_filename or "target.pdb", target_content)
            use_ids = [c for c in (target_file_chains or t_chains) if c in tgt_struct]
            if not use_ids:
                use_ids = list(tgt_struct)
            tgt_ids = use_ids
            target_seq = ":".join(tgt_struct[c]["sequence"] for c in tgt_ids)
        else:
            target_seq = ":".join(chains[c]["sequence"] for c in tgt_ids if c in chains)

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
                "target_filename": target_filename,
                "target_content": target_content,
                "target_file_chains": target_file_chains,
                "params_str": params_str,
                "use_msa": use_msa,
                "run_monomer": run_monomer,
                "motif_residues": motif_ids,
                "motif_indices": motif_idxs,
                "ipsae_pae_cutoff": ipsae_pae_cutoff,
                "ipsae_dist_cutoff": ipsae_dist_cutoff,
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
            print(f"  n={n} len={len(seq.split(':')[0])} hash={hashlib.md5(seq.encode()).hexdigest()[:8]}")

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

    seq_csv = out_dir / "sequences.csv"
    if seq_rows:
        with seq_csv.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(seq_rows[0].keys()))
            w.writeheader()
            w.writerows(seq_rows)

    OUT_VOLUME.commit()

    if not jobs:
        raise RuntimeError(f"No valid designs to predict. Parse errors: {parse_errors}")

    print(f"Running Boltz on {len(jobs)} design(s) "
          f"({'complex+monomer' if run_monomer else 'complex only'})...")
    results = list(validate_one.map(jobs, order_outputs=True))
    for err in parse_errors:
        results.append(
            {"design": err["design"], "error": err["error"], "binder_seq": "", "target_seq": ""}
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
        ipsae = r.get("ipsae")
        plddt = r.get("complex_plddt")
        rmsd = r.get("rmsd_binder_on_target")
        interface = ipsae if ipsae is not None else iptm
        if interface is not None and plddt is not None:
            rmsd_term = 1.0 / (1.0 + (rmsd if rmsd is not None else 0.0))
            r["rank_score"] = float(interface) * float(plddt) * rmsd_term
        else:
            r["rank_score"] = None

        r["passes_filters"] = bool(
            not r.get("error")
            and (
                (ipsae is not None and ipsae >= 0.5)
                or (ipsae is None and iptm is not None and iptm >= 0.5)
            )
            and plddt is not None
            and plddt >= 0.7
            and (rmsd is None or rmsd <= 4.0)
        )

    ranked = sorted(
        results,
        key=lambda r: (
            1 if r.get("error") else 0,
            -(r.get("rank_score") if r.get("rank_score") is not None else -1.0),
            -(r.get("ipsae") if r.get("ipsae") is not None else -1.0),
            -(
                r.get("binder_target_iptm")
                if r.get("binder_target_iptm") is not None
                else (
                    r.get("iptm") if r.get("iptm") is not None else -1.0
                )
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
        "protein_iptm",
        "complex_plddt",
        "complex_iplddt",
        "binder_plddt_complex",
        "target_plddt_complex",
        "ptm",
        "confidence_score",
        "ipsae",
        "ipsae_min",
        "ipsae_d0chn",
        "ipsae_d0dom",
        "pdockq",
        "pdockq2",
        "lis",
        "interface_pae_mean",
        "interface_pae_min",
        "complex_ipde",
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
    recycling_steps: int = 3,
    diffusion_samples: int = 1,
    use_potentials: bool = False,
    out_dir: str = "./out/boltz_validate",
    motif_residues: str | None = None,
    motif_json: str | None = None,
    ipsae_pae_cutoff: float = 10.0,
    ipsae_dist_cutoff: float = 10.0,
):
    """Validate binder designs with Boltz-2.

    Args:
        input_dir: Folder of PDB/CIF designs. Path on the volume if
            --volume-name is set, otherwise a local folder.
        volume_name: Modal Volume that holds the designs (bindcraft,
            bindcraft2, proteinhunter, rfd3, or DESIGN_VOLUME).
        binder_chain: Binder chain ID in each design (default B).
        target_chains: Comma-separated target chain IDs (default A).
        target_pdb: Target structure if designs are binder-only.
        run_name: Output subdirectory on the boltz-validate volume.
        params_str: Extra flags for `boltz predict` (overrides constructed flags).
        msa: Fetch a ColabFold MSA for unique target sequences (default True).
            Pass --no-msa for single-sequence mode on every chain.
        monomer: Also predict the binder alone (default True). Pass --no-monomer to skip.
        recursive: Recurse into subfolders (skips BindCraft Ranked/).
        recycling_steps: Boltz recycles (3 default; 10 is more accurate).
        diffusion_samples: Number of Boltz samples (1 default; 5+ is more robust).
        use_potentials: Enable Boltz-2x inference-time potentials.
        out_dir: Local directory for the downloaded rankings.csv.
        motif_residues: Comma-separated motif residue IDs (e.g. A4,A6,A18)
            applied to every design. Omit to read each RFD3 JSON's
            diffused_index_map (recommended for motif scaffolding).
        motif_json: One RFD3 metadata JSON with diffused_index_map. Per-design
            sibling JSON next to the CIF (or parent of mpnn/) is used when omitted.
        ipsae_pae_cutoff: PAE cutoff for ipSAE (Boltz default 10).
        ipsae_dist_cutoff: CA–CA distance cutoff for ipSAE (Boltz default 10).
    """
    from datetime import datetime

    if not input_dir:
        raise ValueError("Provide --input-dir (local folder or path on --volume-name)")

    today = datetime.now().strftime("%Y%m%d%H%M")[2:]
    run_name = run_name or today
    do_msa = msa
    do_monomer = monomer

    if params_str is None:
        flags = [
            "--seed 42",
            f"--recycling_steps {recycling_steps}",
            f"--diffusion_samples {diffusion_samples}",
            "--override",
            "--write_full_pae",
        ]
        if use_potentials:
            flags.append("--use_potentials")
        params_str = " ".join(flags)

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
        elif volume_name:
            # resolved inside run_validation from the volume if local file missing
            pass

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
        motif_residues=motif_residues,
        motif_json=motif_json,
        ipsae_pae_cutoff=ipsae_pae_cutoff,
        ipsae_dist_cutoff=ipsae_dist_cutoff,
    ).get()

    local = Path(out_dir) / run_name
    local.mkdir(parents=True, exist_ok=True)
    (local / "rankings.csv").write_text(result["csv"])
    print(f"Copied rankings.csv to {local / 'rankings.csv'}")
    print(f"Summary: {result['summary']}")
    print(f"Boltz validate finished: volume={result['volume']} run={result['run_name']}")
