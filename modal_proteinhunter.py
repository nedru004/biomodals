# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "modal>=1.0",
# ]
# ///
"""Runs Protein Hunter (Boltz edition) binder design on Modal.

Protein Hunter fork: https://github.com/nedru004/Protein-Hunter
Upstream: https://github.com/yehlincho/Protein-Hunter
Paper: https://www.biorxiv.org/content/10.1101/2025.10.10.681530

Designs de novo protein binders with Boltz-2 hallucination + LigandMPNN
sequence design. AlphaFold3 cross-validation is not included. The fork
locks binder residues during MPNN via --fixed_positions (and optional
--motif amino acids to graft at those sites).

Results are written to the Modal Volume named "proteinhunter". Create it
once, then always use --detach so you can close the terminal:

    modal volume create proteinhunter

    GPU=A100 uv run --with modal modal run --detach modal_proteinhunter.py \\
      --protein-seqs AFTVTVPKDLYVVEYGSNMTIECKFPVEKQLDLAALIVYWEMEDKNIIQFVHGEEDLKVQHSSYRQRARLLKDQLSLGNAALQITDVKLQDAGVYRCMISYGGADYKRITVKVNAPYAAALE \\
      --num-designs 1

    # or from a PDB, like BindCraft:
    GPU=A100 uv run --with modal modal run --detach modal_proteinhunter.py \\
      --input-pdb PDL1.pdb --target-chains A --num-designs 1

    # redesign an existing binder, keeping selected residues (AAs from --seq):
    GPU=A100 uv run --with modal modal run --detach modal_proteinhunter.py \\
      --protein-seqs AFTVTVPKDLYVVEYGSNMTIECKFPVEKQLDLAALIVYWEMEDKNIIQFVHGEEDLKVQHSSYRQRARLLKDQLSLGNAALQITDVKLQDAGVYRCMISYGGADYKRITVKVNAPYAAALE \\
      --seq GPDRERARELARILLKVIKLSDSPEARRQLLRNLEELAEKYKDPEVRRILEEAERYIK \\
      --fixed-positions 12,15,20-24 --num-designs 1

    # graft a motif into a de novo binder (MPNN will not redesign those sites):
    GPU=A100 uv run --with modal modal run --detach modal_proteinhunter.py \\
      --protein-seqs AFTVTVPKDLYVVEYGSNMTIECKFPVEKQLDLAALIVYWEMEDKNIIQFVHGEEDLKVQHSSYRQRARLLKDQLSLGNAALQITDVKLQDAGVYRCMISYGGADYKRITVKVNAPYAAALE \\
      --motif RGD --fixed-positions 45-47 --num-designs 1

    # keep fixed binder residues (or atoms) in contact with the target (Boltz force=true):
    # Protein Hunter chains: A = binder, B/C = target. Shorthand 12:45 means A12-B45.
    # Optional atoms: A12.OG:B45.NE2, A12OG:B45NE2, or ligand A12:C.C20
    GPU=A100 uv run --with modal modal run --detach modal_proteinhunter.py \\
      --protein-seqs AFTVTVPKDLYVVEYGSNMTIECKFPVEKQLDLAALIVYWEMEDKNIIQFVHGEEDLKVQHSSYRQRARLLKDQLSLGNAALQITDVKLQDAGVYRCMISYGGADYKRITVKVNAPYAAALE \\
      --seq GPDRERARELARILLKVIKLSDSPEARRQLLRNLEELAEKYKDPEVRRILEEAERYIK \\
      --fixed-positions 12,15 --force-contacts A12.OG:B45.NE2,A15:B48 --num-designs 1

    # refine a folder of designed complexes (A = target, B = binder):
    GPU=A100 uv run --with modal modal run --detach modal_proteinhunter.py \\
      --input-dir ./designs --binder-chain B --target-chains A --num-designs 1

    # local template CIF is uploaded automatically (repeated for multi-chain targets):
    GPU=A100 uv run --with modal modal run --detach modal_proteinhunter.py \\
      --input-dir ./designs --binder-chain A --target-chains B,C \\
      --template-path ./TPO7_V9-9_small.cif --template-cif-chain-id B,C

Download results later:

    modal volume get proteinhunter <run_name> ./out/proteinhunter/

Folder refine also writes into <input-dir>/proteinhunter/ after the job
finishes (or download there with the command above).

~7-10 min per design on an H100; expect ~20-25 GB VRAM.
"""

import os
import re
from pathlib import Path

from modal import App, Image, Volume

GPU = os.environ.get("GPU", "L40S")
TIMEOUT = int(os.environ.get("TIMEOUT", 12)) * 60 * 60
MAX_CONTAINERS = int(os.environ.get("MAX_CONTAINERS", 4))
print(f"Using GPU {GPU}; TIMEOUT {TIMEOUT}; MAX_CONTAINERS {MAX_CONTAINERS}")

STRUCTURE_SUFFIXES = {".pdb", ".cif", ".mmcif"}
OUT_SUBDIR_DEFAULT = "proteinhunter"

VOLUME_NAME = "proteinhunter"
VOLUME = Volume.from_name(VOLUME_NAME)
VOLUME_MOUNT = f"/{VOLUME_NAME}"

PH_ROOT = "/root/Protein-Hunter"
# Pin so Modal rebuilds the image after the fork's fixed-position changes.
PH_COMMIT = "0ffe83296b3a70bd43ed17bc749b401223a1e679"

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
        f"git clone https://github.com/nedru004/Protein-Hunter.git {PH_ROOT}"
        f" && cd {PH_ROOT} && git checkout {PH_COMMIT}"
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


# Boltz contact pairs: CHAIN?RES(.ATOM)? : CHAIN?RES(.ATOM)?  (A = binder, B = first target).
_FORCE_CONTACT_ATOM = r"[A-Za-z][A-Za-z0-9']*"
_FORCE_CONTACT_SIDE = (
    r"(?:(?P<c{n}>[A-Za-z]+))?"
    r"(?:"
    r"(?P<r{n}>\d+)(?:[./](?P<a{n}>[A-Za-z0-9']+)|(?P<g{n}>" + _FORCE_CONTACT_ATOM + r"))?"
    r"|"
    r"[./](?P<l{n}>[A-Za-z0-9']+)"
    r")"
)
_FORCE_CONTACT_TOKEN = re.compile(
    _FORCE_CONTACT_SIDE.format(n=1) + r"\s*[:=\-]\s*" + _FORCE_CONTACT_SIDE.format(n=2)
)
FORCE_CONTACT_DISTANCE_DEFAULT = 6.0
_FORCE_CONTACT_HELP = (
    "Use binder:target pairs such as A12:B45, A12.OG:B45.NE2, or A12:C.C20 "
    "(A = binder, B = first target; optional atoms after . or /)."
)


def _force_contact_yaml_token(token: list) -> list:
    """Boltz contact YAML is two fields: [chain, res] or [chain, atom]."""
    if len(token) == 3:
        return [token[0], token[1]]
    return list(token)


def _format_force_contact_token(token: list) -> str:
    if len(token) == 3:
        return f"{token[0]}{token[1]}.{token[2]}"
    if isinstance(token[1], str):
        return f"{token[0]}.{token[1]}"
    return f"{token[0]}{token[1]}"


def _force_contact_side_token(match: re.Match, n: int, default_chain: str) -> list:
    """Build one Boltz token: [chain, res], [chain, res, atom], or [chain, atom]."""
    chain = (match.group(f"c{n}") or default_chain).upper()
    res_raw = match.group(f"r{n}")
    atom = match.group(f"a{n}") or match.group(f"g{n}") or match.group(f"l{n}")
    atom = atom.upper() if atom else None
    if res_raw is None:
        if not atom:
            raise ValueError(f"Invalid --force-contacts token {match.group(0)!r}. {_FORCE_CONTACT_HELP}")
        return [chain, atom]
    res = int(res_raw)
    if res < 1:
        raise ValueError(f"Contact residues are 1-indexed: {match.group(0)!r}")
    if atom:
        return [chain, res, atom]
    return [chain, res]


def parse_force_contacts(spec: str) -> list[tuple[list, list]]:
    """Parse residue/atom pairs into Boltz ``contact`` tokens.

    Protein Hunter chain IDs: A is the designed binder, B/C/... are targets.
    Accepts ``A12:B45``, ``A12.OG:B45.NE2``, ``A12OG:B45NE2``, ligand
    ``A12:C.C20``, or shorthand ``12:45`` (A12 to B45). Residue numbers
    are 1-indexed.
    """
    if not spec or not spec.strip():
        return []
    text = spec.strip()
    pairs: list[tuple[list, list]] = []
    pos = 0
    for match in _FORCE_CONTACT_TOKEN.finditer(text):
        gap = text[pos : match.start()].strip(" ,;")
        if gap:
            raise ValueError(
                f"Invalid --force-contacts token {gap!r}. {_FORCE_CONTACT_HELP}"
            )
        token1 = _force_contact_side_token(match, 1, "A")
        token2 = _force_contact_side_token(match, 2, "B")
        pairs.append((token1, token2))
        pos = match.end()
    gap = text[pos:].strip(" ,;")
    if gap:
        raise ValueError(
            f"Invalid --force-contacts token {gap!r}. {_FORCE_CONTACT_HELP}"
        )
    if not pairs:
        raise ValueError(f"Invalid --force-contacts token {text!r}. {_FORCE_CONTACT_HELP}")
    return pairs


def pdb_chains_to_seqs(pdb_str: str, chains: str, filename: str = "input.pdb") -> str:
    """Extract one-letter sequences for the given chains, colon-separated.

    Uses CA atoms so each residue is counted once. MSE is mapped to M.
    PDB and mmCIF text are both accepted.
    """
    all_seqs = extract_chain_sequences(filename, pdb_str)
    wanted = _chain_ids(chains)
    missing = [c for c in wanted if not all_seqs.get(c)]
    if missing:
        have = ", ".join(all_seqs) or "(none)"
        raise ValueError(
            f"No protein sequence found for chain(s): {', '.join(missing)} "
            f"(chains present: {have})"
        )
    return ":".join(all_seqs[c] for c in wanted)


def _looks_like_cif(filename: str, content: str) -> bool:
    if filename.lower().endswith((".cif", ".mmcif")):
        return True
    for line in content.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        return s.lower().startswith("data_")
    return False


def extract_chain_sequences(filename: str, content: str) -> dict[str, str]:
    """Return {chain_id: one-letter sequence} from PDB or mmCIF text."""
    if _looks_like_cif(filename, content):
        seqs = _cif_chain_sequences(content)
    else:
        seqs = _pdb_chain_sequences(content)
    if not seqs:
        raise ValueError(f"No protein CA atoms in {filename}")
    return seqs


def _pdb_chain_sequences(pdb_str: str) -> dict[str, str]:
    seqs: dict[str, list[str]] = {}
    seen: dict[str, set[int]] = {}
    for line in pdb_str.splitlines():
        if not line.startswith("ATOM") and not (
            line.startswith("HETATM") and len(line) >= 26 and line[17:20] == "MSE"
        ):
            continue
        if len(line) < 26:
            continue
        chain = line[21].strip() or "A"
        atom = line[12:16].strip()
        if atom != "CA":
            continue
        try:
            resnum = int(line[22:26])
        except ValueError:
            continue
        aa = AA3TO1.get(line[17:20].strip())
        if aa is None:
            continue
        if chain not in seqs:
            seqs[chain] = []
            seen[chain] = set()
        if resnum in seen[chain]:
            continue
        seen[chain].add(resnum)
        seqs[chain].append(aa)
    return {c: "".join(s) for c, s in seqs.items() if s}


def _tokenize_cif(s: str) -> list[str]:
    tokens: list[str] = []
    i = 0
    n = len(s)
    while i < n:
        c = s[i]
        if c.isspace():
            i += 1
            continue
        if c in "'\"":
            j = i + 1
            while j < n and s[j] != c:
                j += 1
            tokens.append(s[i + 1 : j])
            i = j + 1
            continue
        j = i
        while j < n and not s[j].isspace():
            j += 1
        tokens.append(s[i:j])
        i = j
    return tokens


def _cif_chain_sequences(content: str) -> dict[str, str]:
    """Parse CA atoms from an mmCIF atom_site loop."""
    lines = content.splitlines()
    i = 0
    n = len(lines)
    seqs: dict[str, list[str]] = {}
    seen: dict[str, set[tuple[int, str]]] = {}

    while i < n:
        if lines[i].strip().lower() != "loop_":
            i += 1
            continue
        i += 1
        cols: list[str] = []
        while i < n:
            s = lines[i].strip()
            if s.startswith("#"):
                i += 1
                continue
            if s.startswith("_"):
                cols.append(s.split()[0])
                i += 1
                continue
            break
        tags = [c.split(".", 1)[-1] for c in cols]
        is_atom_site = any(c.startswith("_atom_site.") for c in cols)
        if not is_atom_site or not tags:
            while i < n:
                s = lines[i].strip()
                if s.lower() == "loop_" or s.lower().startswith("data_") or (
                    s.startswith("_") and not s.startswith("_atom_site")
                ):
                    break
                i += 1
            continue

        idx = {name: k for k, name in enumerate(tags)}
        buf: list[str] = []
        ncols = len(tags)
        while i < n:
            s = lines[i].strip()
            if not s or s.startswith("#"):
                i += 1
                continue
            low = s.lower()
            if low == "loop_" or low.startswith("data_") or s.startswith("_"):
                break
            buf.extend(_tokenize_cif(s))
            while len(buf) >= ncols:
                row = buf[:ncols]
                buf = buf[ncols:]
                _add_cif_ca(row, idx, seqs, seen)
            i += 1

    return {c: "".join(s) for c, s in seqs.items() if s}


def _add_cif_ca(
    row: list[str],
    idx: dict[str, int],
    seqs: dict[str, list[str]],
    seen: dict[str, set[tuple[int, str]]],
) -> None:
    def col(*names: str) -> str:
        for name in names:
            if name in idx:
                val = row[idx[name]]
                if val not in (".", "?"):
                    return val
        return ""

    group = col("group_PDB")
    if group and group not in ("ATOM", "HETATM"):
        return
    model = col("pdbx_PDB_model_num")
    if model and model not in ("1",):
        return
    alt = col("label_alt_id")
    if alt and alt not in ("A",):
        return
    atom = col("auth_atom_id", "label_atom_id").strip()
    if atom != "CA":
        return
    resname = col("auth_comp_id", "label_comp_id").strip()
    aa = AA3TO1.get(resname)
    if aa is None:
        return
    chain = col("auth_asym_id", "label_asym_id").strip() or "A"
    res_s = col("auth_seq_id", "label_seq_id")
    try:
        resnum = int(res_s)
    except (TypeError, ValueError):
        resnum = len(seqs.get(chain, [])) + 1
    ins = col("pdbx_PDB_ins_code", "label_ins_code") or ""
    key = (resnum, ins)
    if chain not in seqs:
        seqs[chain] = []
        seen[chain] = set()
    if key in seen[chain]:
        return
    seen[chain].add(key)
    seqs[chain].append(aa)


def resolve_roles(
    chain_seqs: dict[str, str],
    binder_chain: str,
    target_chains: list[str],
) -> tuple[str, list[str]]:
    """Pick binder vs target chain IDs.

    A single-chain file is always the binder (use --target-pdb for the target).
    """
    chain_ids = list(chain_seqs)
    if binder_chain and binder_chain in chain_seqs:
        binder = binder_chain
    elif len(chain_ids) == 1:
        binder = chain_ids[0]
    else:
        raise ValueError(
            f"Binder chain {binder_chain!r} not in {chain_ids}; "
            "pass --binder-chain with one of these IDs"
        )

    targets = [c for c in target_chains if c in chain_seqs and c != binder]
    if not targets:
        targets = [c for c in chain_ids if c != binder]
    return binder, targets


def _iter_structure_files(
    root: Path, recursive: bool, skip_names: set[str]
) -> list[Path]:
    files = root.rglob("*") if recursive else root.glob("*")
    out: list[Path] = []
    for f in files:
        if not f.is_file() or f.suffix.lower() not in STRUCTURE_SUFFIXES:
            continue
        rel_parts = f.relative_to(root).parts
        if any(p in skip_names for p in rel_parts[:-1]):
            continue
        out.append(f)
    return sorted(out)


def _safe_stem(name: str) -> str:
    cleaned = "".join(c if c.isalnum() or c in "-_." else "_" for c in name)
    return cleaned.strip("._") or "design"


def _collect_refine_jobs(
    files: list[Path],
    binder_chain: str,
    target_chains: str,
    target_pdb: str | None,
    protein_seqs_override: str | None,
) -> list[dict]:
    """Parse each structure into a Protein Hunter refine job."""
    t_ids = _chain_ids(target_chains)
    target_override = protein_seqs_override or ""
    target_from_file = ""
    if target_pdb:
        tpath = Path(target_pdb)
        if not tpath.exists():
            raise FileNotFoundError(f"target-pdb not found: {tpath}")
        t_seqs = extract_chain_sequences(tpath.name, tpath.read_text())
        use = [c for c in t_ids if c in t_seqs] or list(t_seqs)
        target_from_file = ":".join(t_seqs[c] for c in use)
        print(f"Target sequence(s) from {tpath}: {target_from_file}")

    used_names: set[str] = set()
    jobs: list[dict] = []
    for path in files:
        content = path.read_text()
        chains = extract_chain_sequences(path.name, content)
        b_id, tgt_ids = resolve_roles(chains, binder_chain, t_ids)
        binder_seq = chains[b_id]
        if target_override:
            target_seq = target_override
            tgt_label = "override"
        elif tgt_ids and any(c in chains and c != b_id for c in tgt_ids):
            target_seq = ":".join(chains[c] for c in tgt_ids if c in chains)
            tgt_label = ",".join(tgt_ids)
        elif target_from_file:
            target_seq = target_from_file
            tgt_label = f"{Path(target_pdb).name}"
        else:
            raise ValueError(
                f"{path.name}: no target chain(s) {target_chains!r} "
                f"(chains: {', '.join(chains)}; binder={b_id}). "
                "Pass --target-pdb or --protein-seqs, or fix --target-chains."
            )
        stem = _safe_stem(path.stem)
        name = stem
        n = 2
        while name in used_names:
            name = f"{stem}_{n}"
            n += 1
        used_names.add(name)
        jobs.append(
            {
                "name": name,
                "filename": path.name,
                "binder_chain": b_id,
                "target_chains": tgt_label,
                "binder_seq": binder_seq,
                "target_seq": target_seq,
            }
        )
        tgt_len = sum(len(s) for s in target_seq.split(":"))
        print(
            f"  {path.name}: binder {b_id} ({len(binder_seq)} aa) "
            f"target {tgt_label} ({tgt_len} aa)"
        )
        if tgt_len and len(binder_seq) > tgt_len:
            print(
                "    WARNING: binder is longer than target; "
                "check --binder-chain (BoltzGen often uses A)"
            )
        if (
            tgt_label
            and tgt_label != "override"
            and not tgt_label.endswith((".pdb", ".cif", ".mmcif"))
        ):
            matching = [
                label
                for label, s in zip(tgt_label.split(","), target_seq.split(":"))
                if s == binder_seq
            ]
            if matching:
                print(
                    f"    WARNING: binder sequence equals target chain(s) "
                    f"{', '.join(matching)}. Boltz requires a shared MSA; "
                    f"consider --target-chains without those IDs if they "
                    f"are not true targets (e.g. binder homodimer copies)."
                )
    return jobs


def _sync_duplicate_seq_msas(data: dict) -> None:
    """Boltz requires identical protein sequences to share the same MSA path.

    Protein Hunter starts the binder as sequence 'X' with msa='empty', then
    swaps in --seq. If that sequence already exists on a target chain with a
    real MSA, leave the binder on 'empty' and Boltz raises:
    'All proteins with the same sequence must share the same MSA!'
    """
    proteins = [
        entry["protein"]
        for entry in data.get("sequences", [])
        if isinstance(entry, dict) and "protein" in entry
    ]
    seq_to_msa: dict[str, str] = {}

    def _msa_rank(m: str) -> int:
        if not m or m in ("empty", "mmseqs"):
            return 0
        return 1  # concrete path / file

    for prot in proteins:
        seq = prot.get("sequence") or ""
        if not seq or seq == "X":
            continue
        msa = prot.get("msa", "empty")
        prev = seq_to_msa.get(seq)
        if prev is None or _msa_rank(msa) > _msa_rank(prev):
            seq_to_msa[seq] = msa

    for prot in proteins:
        seq = prot.get("sequence") or ""
        if seq not in seq_to_msa:
            continue
        want = seq_to_msa[seq]
        if prot.get("msa") != want:
            print(
                f"Syncing MSA for chain {prot.get('id')} "
                f"(duplicate sequence) → {want!r}"
            )
            prot["msa"] = want


def _patch_binder_msa_sync() -> None:
    """Ensure binder/target duplicate sequences share an MSA before predict."""
    import model_utils

    orig = model_utils.run_prediction

    def run_prediction_synced(data, *args, **kwargs):
        if isinstance(data, dict):
            _sync_duplicate_seq_msas(data)
        return orig(data, *args, **kwargs)

    model_utils.run_prediction = run_prediction_synced
    try:
        import pipeline as pipeline_mod

        pipeline_mod.run_prediction = run_prediction_synced
    except Exception:
        pass
    print("Patched run_prediction to sync MSAs for duplicate sequences")


def _download_volume_dir(run_subdir: str, dest: Path) -> bool:
    """Copy a volume subdirectory into dest. Returns True on success."""
    import shutil
    import subprocess

    dest.mkdir(parents=True, exist_ok=True)
    modal_bin = shutil.which("modal") or "modal"
    cmd = [
        modal_bin,
        "volume",
        "get",
        "--force",
        VOLUME_NAME,
        f"{run_subdir}/",
        str(dest) + "/",
    ]
    print("Downloading:", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    return result.returncode == 0


def _smart_split(s: str) -> list[str]:
    """Split comma- or colon-separated Protein Hunter path lists."""
    if not s or not s.strip():
        return []
    if "," in s:
        return [x.strip() for x in s.split(",") if x.strip()]
    if ":" in s:
        return [x.strip() for x in s.split(":") if x.strip()]
    return [s.strip()]


def _n_target_slots(
    protein_seqs: str | None,
    target_chains: str,
    template_cif_chain_id: str,
) -> int:
    """How many template slots Protein Hunter expects (one per target chain)."""
    n = 1
    if protein_seqs:
        n = max(n, len(_smart_split(protein_seqs)))
    t_chains = _chain_ids(target_chains)
    if t_chains:
        n = max(n, len(t_chains))
    cif_ids = _smart_split(template_cif_chain_id)
    if cif_ids:
        n = max(n, len(cif_ids))
    return n


def prepare_template_uploads(
    template_path: str,
    *,
    n_targets: int = 1,
) -> tuple[str, list[dict]]:
    """Load local template files for Modal upload.

    Local paths become upload entries; 4-letter PDB codes and existing absolute
    remote/volume paths pass through unchanged. A single template is repeated
    when n_targets > 1 (Protein Hunter maps one template per target chain).

    Returns:
        (remote_template_path, uploads) where remote_template_path is
        comma-separated and uploads is a list of
        {"key": str, "name": str, "content": bytes}.
    """
    paths = _smart_split(template_path)
    if not paths:
        return "", []
    if len(paths) == 1 and n_targets > 1:
        paths = paths * n_targets

    uploads: list[dict] = []
    remote_parts: list[str] = []
    seen_keys: dict[str, str] = {}  # local abspath -> upload key

    for i, part in enumerate(paths):
        local = Path(part).expanduser()
        if local.is_file():
            abspath = str(local.resolve())
            if abspath in seen_keys:
                remote_parts.append(seen_keys[abspath])
                continue
            key = f"tmpl_{i}_{_safe_stem(local.stem)}{local.suffix.lower() or '.cif'}"
            uploads.append(
                {"key": key, "name": local.name, "content": local.read_bytes()}
            )
            seen_keys[abspath] = key
            remote_parts.append(key)
            print(f"Will upload template: {local} → {key}")
        elif part.startswith(VOLUME_MOUNT) or part.startswith("/"):
            # Already a volume / absolute path on the remote filesystem
            remote_parts.append(part)
        elif len(part) == 4 and part.isalnum():
            remote_parts.append(part)  # PDB ID → Protein Hunter wget
        else:
            # Relative path that is not a local file — fail early rather than
            # letting Protein Hunter invent AF-{name}-F1-model_v3.cif
            raise FileNotFoundError(
                f"Template not found locally: {part!r}. "
                "Pass a path to an existing .pdb/.cif, a 4-letter PDB ID, "
                f"or an absolute path already on the volume ({VOLUME_MOUNT}/...)."
            )

    return ",".join(remote_parts), uploads


def materialize_templates(
    template_path: str,
    uploads: list[dict] | None,
    dest_dir: Path,
) -> str:
    """Write uploaded templates under dest_dir and rewrite path keys.

    upload keys in template_path are replaced with absolute paths on disk so
    Protein Hunter's get_cif() sees a real file.
    """
    if not template_path:
        return ""
    if not uploads:
        return template_path

    dest_dir.mkdir(parents=True, exist_ok=True)
    key_to_path: dict[str, str] = {}
    for item in uploads:
        key = item["key"]
        out = dest_dir / key
        out.write_bytes(item["content"])
        key_to_path[key] = str(out.resolve())
        print(f"Wrote template {item.get('name', key)} → {out}")

    parts = _smart_split(template_path)
    resolved = [key_to_path.get(p, p) for p in parts]
    return ",".join(resolved)


def _patch_boltz_parse_pdb() -> None:
    """Protein Hunter schema.py passes ignore_ligands= to parse_pdb; pdb.py omitted it."""
    from tempfile import NamedTemporaryFile
    from typing import Optional

    import gemmi
    from rdkit.Chem.rdchem import Mol

    from boltz.data.parse import pdb as pdb_mod
    from boltz.data.parse.mmcif import ParsedStructure, parse_mmcif

    def parse_pdb(
        path: str,
        mols: Optional[dict[str, Mol]] = None,
        moldir: Optional[str] = None,
        use_assembly: bool = True,
        compute_interfaces: bool = True,
        ignore_ligands: bool = False,
    ) -> ParsedStructure:
        with NamedTemporaryFile(suffix=".cif") as tmp_cif_file:
            tmp_cif_path = tmp_cif_file.name
            structure = gemmi.read_structure(str(path))
            structure.setup_entities()

            subchain_counts: dict[str, int] = {}
            subchain_renaming: dict[str, str] = {}
            for chain in structure[0]:
                subchain_counts[chain.name] = 0
                for res in chain:
                    if res.subchain not in subchain_renaming:
                        subchain_renaming[res.subchain] = (
                            chain.name + str(subchain_counts[chain.name] + 1)
                        )
                        subchain_counts[chain.name] += 1
                    res.subchain = subchain_renaming[res.subchain]
            for entity in structure.entities:
                entity.subchains = [
                    subchain_renaming[subchain] for subchain in entity.subchains
                ]

            structure.make_mmcif_document().write_file(tmp_cif_path)
            return parse_mmcif(
                path=tmp_cif_path,
                mols=mols,
                moldir=moldir,
                use_assembly=use_assembly,
                compute_interfaces=compute_interfaces,
                ignore_ligands=ignore_ligands,
            )

    pdb_mod.parse_pdb = parse_pdb
    try:
        from boltz.data.parse import schema as schema_mod

        schema_mod.parse_pdb = parse_pdb
    except Exception:
        pass
    print("Patched boltz parse_pdb to accept ignore_ligands")


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


def _decode_atom_name(raw) -> str:
    """Normalize Boltz atom-name storage (str, bytes, or 4-int code)."""
    if isinstance(raw, bytes):
        return raw.decode("ascii", "ignore").strip()
    if isinstance(raw, str):
        return raw.strip()
    try:
        if hasattr(raw, "tolist"):
            raw = raw.tolist()
        if isinstance(raw, (list, tuple)):
            return "".join(chr(int(c) + 32) for c in raw if int(c) > 0).strip()
    except (TypeError, ValueError):
        pass
    return str(raw).strip()


def _residue_atom_span(token, atom_name, data) -> tuple[int, int]:
    """Atom index range for a contact token, optionally restricted to one name."""
    start = int(token["atom_idx"])
    end = start + int(token["atom_num"])
    if not atom_name:
        return start, end
    want = atom_name.strip().upper()
    atoms = data.structure.atoms
    found: list[str] = []
    for i in range(start, end):
        name = _decode_atom_name(atoms[i]["name"])
        found.append(name)
        if name.upper() == want:
            return i, i + 1
    names = ", ".join(found) or "none"
    raise ValueError(
        f"Atom {atom_name!r} not in residue (found: {names}). "
        "Use CCD/PDB names such as CA, OG, NE2; the amino acid must have that atom."
    )


def _patch_force_contacts(
    pairs: list[tuple[list, list]],
    max_distance: float,
) -> None:
    """Inject Boltz forced residue/atom-pair contacts into Protein Hunter.

    Protein Hunter only builds pocket constraints from --contact_residues.
    This adds ``contact`` constraints with force=true, turns on steering
    potentials, and applies them on every design cycle. Named polymer atoms
    are enforced by restricting the force potential to those atoms.
    """
    if not pairs:
        return

    import model_utils
    import pipeline as pipeline_mod
    import torch
    from boltz.data import const as boltz_const
    from boltz.data.feature import featurizerv2

    orig_build = pipeline_mod.InputDataBuilder._build_conditional_data
    yaml_pairs = [
        (_force_contact_yaml_token(t1), _force_contact_yaml_token(t2))
        for t1, t2 in pairs
    ]
    atom_pairs = [
        (t1[2] if len(t1) == 3 else None, t2[2] if len(t2) == 3 else None)
        for t1, t2 in pairs
    ]
    nonpolymer = boltz_const.chain_type_ids["NONPOLYMER"]

    def _token_matches(token, spec) -> bool:
        if token["mol_type"] == nonpolymer:
            return (token["asym_id"], token["atom_idx"]) == spec
        return (token["asym_id"], token["res_idx"]) == spec

    def _build_conditional_data(self):
        data, _pocket = orig_build(self)
        constraints = list(data.get("constraints") or [])
        for token1, token2 in yaml_pairs:
            constraints.append(
                {
                    "contact": {
                        "token1": token1,
                        "token2": token2,
                        "max_distance": max_distance,
                        "force": True,
                    }
                }
            )
        data["constraints"] = constraints
        pretty = ", ".join(
            f"{_format_force_contact_token(a)}:{_format_force_contact_token(b)}"
            for a, b in pairs
        )
        print(
            f"Boltz forced contacts (max_distance={max_distance} Å, force=true): {pretty}"
        )
        return data, True

    pipeline_mod.InputDataBuilder._build_conditional_data = _build_conditional_data

    if any(a or b for a, b in atom_pairs):
        orig_pcc = featurizerv2.process_contact_feature_constraints

        def process_contact_feature_constraints(
            data,
            inference_pocket_constraints,
            inference_contact_constraints,
        ):
            inference_pocket_constraints = inference_pocket_constraints or []
            inference_contact_constraints = list(
                inference_contact_constraints or []
            )
            n = len(atom_pairs)
            atom_for = [(None, None)] * len(inference_contact_constraints)
            if n and len(inference_contact_constraints) >= n:
                atom_for[-n:] = atom_pairs

            token_data = data.tokens
            pair_index, union_index, negation_mask, thresholds = [], [], [], []
            union_idx = 0

            def _add(atom_idx_pairs, dist):
                nonlocal union_idx
                pair_index.append(atom_idx_pairs)
                count = atom_idx_pairs.shape[1]
                union_index.append(torch.full((count,), union_idx))
                negation_mask.append(torch.ones((count,), dtype=torch.bool))
                thresholds.append(torch.full((count,), dist))
                union_idx += 1

            for binder, contacts, dist, force in inference_pocket_constraints:
                if not force:
                    continue
                binder_chain = data.structure.chains[binder]
                for token in token_data:
                    if (
                        token["mol_type"] != nonpolymer
                        and (token["asym_id"], token["res_idx"]) in contacts
                    ) or (
                        token["mol_type"] == nonpolymer
                        and (token["asym_id"], token["atom_idx"]) in contacts
                    ):
                        _add(
                            torch.cartesian_prod(
                                torch.arange(
                                    binder_chain["atom_idx"],
                                    binder_chain["atom_idx"] + binder_chain["atom_num"],
                                ),
                                torch.arange(
                                    token["atom_idx"],
                                    token["atom_idx"] + token["atom_num"],
                                ),
                            ).T,
                            dist,
                        )

            for i, (token1, token2, dist, force) in enumerate(
                inference_contact_constraints
            ):
                if not force:
                    continue
                atom1, atom2 = atom_for[i]
                for _token1 in token_data:
                    if not _token_matches(_token1, token1):
                        continue
                    for _token2 in token_data:
                        if not _token_matches(_token2, token2):
                            continue
                        s1, e1 = _residue_atom_span(_token1, atom1, data)
                        s2, e2 = _residue_atom_span(_token2, atom2, data)
                        _add(
                            torch.cartesian_prod(
                                torch.arange(s1, e1),
                                torch.arange(s2, e2),
                            ).T,
                            dist,
                        )
                        break
                    break

            if pair_index:
                return {
                    "contact_pair_index": torch.cat(pair_index, dim=1),
                    "contact_union_index": torch.cat(union_index),
                    "contact_negation_mask": torch.cat(negation_mask),
                    "contact_thresholds": torch.cat(thresholds),
                }
            return orig_pcc(
                data, inference_pocket_constraints, inference_contact_constraints
            )

        featurizerv2.process_contact_feature_constraints = (
            process_contact_feature_constraints
        )

    orig_gbm = model_utils.get_boltz_model

    def get_boltz_model(*args, **kwargs):
        kwargs["no_potentials"] = False
        return orig_gbm(*args, **kwargs)

    model_utils.get_boltz_model = get_boltz_model
    pipeline_mod.get_boltz_model = get_boltz_model

    orig_rp = model_utils.run_prediction

    def run_prediction(*args, **kwargs):
        kwargs["pocket_conditioning"] = True
        return orig_rp(*args, **kwargs)

    model_utils.run_prediction = run_prediction
    pipeline_mod.run_prediction = run_prediction

    orig_cycle = pipeline_mod.ProteinHunter_Boltz._run_design_cycle

    class _SkipPocketLookup(dict):
        """Skip Protein Hunter's pocket-only chain lookup when only contacts exist."""

        def __contains__(self, key):
            if key == "constraints":
                cons = dict.get(self, "constraints")
                if cons and "pocket" not in cons[0]:
                    return False
            return dict.__contains__(self, key)

    def _run_design_cycle(self, data_cp, run_id, pocket_conditioning):
        if isinstance(data_cp, dict) and not isinstance(data_cp, _SkipPocketLookup):
            data_cp = _SkipPocketLookup(data_cp)
        return orig_cycle(self, data_cp, run_id, True)

    pipeline_mod.ProteinHunter_Boltz._run_design_cycle = _run_design_cycle


@app.function(
    image=image,
    gpu=GPU,
    timeout=TIMEOUT,
    volumes={VOLUME_MOUNT: VOLUME},
    max_containers=MAX_CONTAINERS,
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
    refiner_mode: bool = False,
    template_uploads: list[dict] | None = None,
    fixed_positions: str = "",
    motif: str = "",
    force_contacts: str = "",
    force_contact_distance: float = FORCE_CONTACT_DISTANCE_DEFAULT,
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
        template_path: PDB code(s) or remote path(s); upload keys rewritten below.
        template_cif_chain_id: Chain ID in a template mmCIF.
        refiner_mode: Pass --refiner_mode to Protein Hunter (existing-seq refine).
        template_uploads: Local template file contents from prepare_template_uploads.
        fixed_positions: 1-indexed binder residues to keep during MPNN
            (e.g. "12,15,20-24" or "A12 A15 A20"). Requires --seq or --motif.
        motif: Amino acids to graft at fixed_positions. If omitted, residues
            are taken from seq at those positions.
        force_contacts: Binder:target residue or atom pairs kept in contact
            with a Boltz force potential, e.g. "A12:B45", "A12.OG:B45.NE2",
            or ligand "A12:C.C20".
        force_contact_distance: Max distance in Å for every forced pair
            (Boltz default 6; supported 4-20).

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

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    template_path = materialize_templates(
        template_path,
        template_uploads,
        Path(save_dir) / "templates",
    )
    _patch_boltz_parse_pdb()

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
    if fixed_positions:
        argv += ["--fixed_positions", fixed_positions]
    if motif:
        argv += ["--motif", motif]
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
    if refiner_mode:
        print("Refiner mode: starting from the existing binder sequence")

    sys.argv = argv
    from design import parse_args, print_args
    from pipeline import ProteinHunter_Boltz

    _patch_colabfold_msa()
    _patch_binder_msa_sync()
    pairs = parse_force_contacts(force_contacts)
    if pairs:
        if not 4 <= force_contact_distance <= 20:
            raise ValueError(
                "--force-contact-distance must be between 4 and 20 Å "
                f"(got {force_contact_distance})"
            )
        _patch_force_contacts(pairs, force_contact_distance)
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


def _run_kwargs(
    *,
    save_dir: str,
    name: str,
    protein_seqs: str,
    seq: str,
    min_len: int,
    max_len: int,
    num_designs: int,
    num_cycles: int,
    percent_x: int,
    msa_mode: str,
    high_iptm_threshold: float,
    contact_residues: str,
    cyclic: bool,
    alanine_bias: bool,
    ligand_ccd: str,
    ligand_smiles: str,
    nucleic_seq: str,
    nucleic_type: str,
    omit_aa: str,
    plot: bool,
    template_path: str,
    template_cif_chain_id: str,
    refiner_mode: bool,
    template_uploads: list[dict] | None = None,
    fixed_positions: str = "",
    motif: str = "",
    force_contacts: str = "",
    force_contact_distance: float = FORCE_CONTACT_DISTANCE_DEFAULT,
) -> dict:
    return dict(
        save_dir=save_dir,
        name=name,
        protein_seqs=protein_seqs,
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
        refiner_mode=refiner_mode,
        template_uploads=template_uploads,
        fixed_positions=fixed_positions,
        motif=motif,
        force_contacts=force_contacts,
        force_contact_distance=force_contact_distance,
    )


@app.local_entrypoint()
def main(
    protein_seqs: str | None = None,
    input_pdb: str | None = None,
    input_dir: str | None = None,
    binder_chain: str = "B",
    target_chains: str = "A",
    target_pdb: str | None = None,
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
    fixed_positions: str = "",
    motif: str = "",
    force_contacts: str = "",
    force_contact_distance: float = FORCE_CONTACT_DISTANCE_DEFAULT,
    binder_name: str | None = None,
    run_name: str | None = None,
    out_subdir: str = OUT_SUBDIR_DEFAULT,
    recursive: bool = False,
):
    """Local entrypoint to run Protein Hunter binder design.

    Uses spawn().get() (required for long detached jobs). Run with
    `modal run --detach` so closing the terminal does not kill the job.
    Results are stored on the Modal Volume named "proteinhunter".

    Provide `--protein-seqs` and/or `--input-pdb` for de novo design, or
    `--input-dir` to refine every PDB/CIF in a folder (binder sequence is
    taken from `--binder-chain`). Ligand/nucleic-only design needs
    `--ligand-ccd`, `--ligand-smiles`, or `--nucleic-seq`.

    Args:
        protein_seqs: Target amino-acid sequence(s). Colon-separated for multimers.
        input_pdb: Path to a target PDB/CIF; sequences from target_chains.
        input_dir: Folder of designed complexes to refine (PDB/CIF).
        binder_chain: Binder chain in each input-dir structure. Defaults to "B".
        target_chains: Target chain(s). Defaults to "A".
        target_pdb: Target structure when input-dir files are binder-only.
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
        template_path: Optional local .pdb/.cif path(s), 4-letter PDB ID, or
            absolute volume path. Local files are uploaded automatically. A
            single file is repeated once per target chain when needed.
        template_cif_chain_id: Chain ID(s) in template mmCIF(s), comma-separated
            in the same order as target chains.
        fixed_positions: 1-indexed binder residues to keep during MPNN
            (e.g. "12,15,20-24"). Binder is Protein Hunter chain A.
            Requires --seq, --motif, or --input-dir.
        motif: Amino acids to graft at --fixed-positions. If omitted,
            residues are taken from --seq at those positions.
        force_contacts: Binder:target residue or atom pairs to keep in
            contact with a Boltz force potential. Protein Hunter chains:
            A = binder, B/C = target. Example: "A12:B45", "A12.OG:B45.NE2",
            or ligand "A12:C.C20". Shorthand "12:45" is A12-B45.
        force_contact_distance: Max distance in Å applied to every pair
            (default 6; Boltz supports 4-20).
        binder_name: Job name. Defaults to the PDB stem or "binder".
        run_name: Volume subdirectory. Defaults to a timestamp.
        out_subdir: Local results folder created inside input-dir. Defaults to proteinhunter.
        recursive: Recurse into input-dir subfolders (skips out-subdir).
    """
    import csv
    from datetime import datetime

    parts = [int(x) for x in lengths.split(",")]
    if len(parts) == 1:
        min_len = max_len = parts[0]
    else:
        min_len, max_len = parts[0], parts[1]

    if motif and not fixed_positions:
        raise SystemExit("--motif requires --fixed-positions")
    if fixed_positions and not seq and not motif and not input_dir:
        raise SystemExit(
            "--fixed-positions requires --seq, --motif, or --input-dir "
            "so the kept amino acids are known"
        )
    if force_contacts:
        parse_force_contacts(force_contacts)
        if not 4 <= force_contact_distance <= 20:
            raise SystemExit(
                "--force-contact-distance must be between 4 and 20 Å"
            )

    today = datetime.now().strftime("%Y%m%d%H%M")[2:]
    run_subdir = run_name or today

    n_tmpl = _n_target_slots(protein_seqs, target_chains, template_cif_chain_id)
    remote_template_path, template_uploads = prepare_template_uploads(
        template_path or "",
        n_targets=n_tmpl,
    )
    if remote_template_path and len(_smart_split(remote_template_path)) > 1:
        print(f"Templates for {n_tmpl} target chain(s): {remote_template_path}")

    common = dict(
        num_designs=num_designs,
        num_cycles=num_cycles,
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
        omit_aa=omit_aa,
        plot=plot,
        template_path=remote_template_path,
        template_cif_chain_id=template_cif_chain_id,
        template_uploads=template_uploads or None,
        fixed_positions=fixed_positions,
        motif=motif,
        force_contacts=force_contacts,
        force_contact_distance=force_contact_distance,
    )

    if input_dir:
        root = Path(input_dir)
        if not root.exists():
            raise FileNotFoundError(f"input-dir not found: {root}")
        skip = {out_subdir, "high_iptm_pdb", "0_protein_hunter_design", "Ranked"}
        files = _iter_structure_files(root, recursive, skip)
        if not files:
            raise FileNotFoundError(f"No PDB/CIF files in {root}")
        print(f"Refining {len(files)} structure(s) from {root}")
        jobs = _collect_refine_jobs(
            files, binder_chain, target_chains, target_pdb, protein_seqs
        )
        local_out = (root / out_subdir).resolve()
        local_out.mkdir(parents=True, exist_ok=True)
        csv_path = local_out / "jobs.csv"
        with csv_path.open("w", newline="") as fh:
            w = csv.DictWriter(
                fh,
                fieldnames=[
                    "name",
                    "filename",
                    "binder_chain",
                    "target_chains",
                    "binder_len",
                    "binder_seq",
                    "target_seq",
                ],
            )
            w.writeheader()
            for job in jobs:
                w.writerow({**job, "binder_len": len(job["binder_seq"])})
        print(f"Wrote {csv_path}")
        print(f"Results → volume '{VOLUME_NAME}' / {run_subdir}/<design>")
        print(f"Local folder: {local_out}")
        print("Run with --detach; you can close the terminal and the job will keep going.")
        print(
            f"Download later: modal volume get {VOLUME_NAME} {run_subdir} {local_out}/"
        )

        handles = []
        for job in jobs:
            blen = len(job["binder_seq"])
            kw = _run_kwargs(
                save_dir=f"{VOLUME_MOUNT}/{run_subdir}/{job['name']}",
                name=job["name"],
                protein_seqs=job["target_seq"],
                seq=job["binder_seq"],
                min_len=blen,
                max_len=blen,
                refiner_mode=True,
                **common,
            )
            handles.append((job["name"], proteinhunter.spawn(**kw)))

        results = []
        n_fail = 0
        for name, handle in handles:
            try:
                results.append(handle.get())
                print(f"Finished {name}")
            except Exception as exc:
                n_fail += 1
                print(f"FAILED {name}: {exc}")
                results.append({"name": name, "error": str(exc)})

        if _download_volume_dir(run_subdir, local_out):
            print(f"Copied volume results into {local_out}")
        else:
            print(
                f"Could not download automatically. "
                f"modal volume get {VOLUME_NAME} {run_subdir} {local_out}/"
            )
        with csv_path.open("w", newline="") as fh:
            w = csv.DictWriter(
                fh,
                fieldnames=[
                    "name",
                    "filename",
                    "binder_chain",
                    "target_chains",
                    "binder_len",
                    "binder_seq",
                    "target_seq",
                ],
            )
            w.writeheader()
            for job in jobs:
                w.writerow({**job, "binder_len": len(job["binder_seq"])})
        print(
            f"Protein Hunter refine finished: {len(results) - n_fail}/{len(results)} "
            f"succeeded → {local_out}"
        )
        return

    if not protein_seqs and input_pdb:
        protein_seqs = pdb_chains_to_seqs(
            Path(input_pdb).read_text(), target_chains, Path(input_pdb).name
        )
        print(f"Extracted target sequence(s) from {input_pdb} chains {target_chains}:")
        print(protein_seqs)

    if not protein_seqs and not ligand_ccd and not ligand_smiles and not nucleic_seq:
        raise SystemExit(
            "Provide --input-dir, --protein-seqs, --input-pdb, "
            "--ligand-ccd, --ligand-smiles, or --nucleic-seq"
        )

    binder_name = binder_name or (Path(input_pdb).stem if input_pdb else "binder")
    save_dir = f"{VOLUME_MOUNT}/{run_subdir}/{binder_name}"

    print(f"Results will be written to volume '{VOLUME_NAME}' at {save_dir}")
    print("Run with --detach; you can close the terminal and the job will keep going.")
    print(
        f"Download later: modal volume get {VOLUME_NAME} {run_subdir} ./out/proteinhunter/"
    )
    print(f"List volume:     modal volume ls {VOLUME_NAME} {run_subdir}")

    result = proteinhunter.spawn(
        **_run_kwargs(
            save_dir=save_dir,
            name=binder_name,
            protein_seqs=protein_seqs or "",
            seq=seq,
            min_len=min_len,
            max_len=max_len,
            refiner_mode=bool(seq),
            **common,
        )
    ).get()
    print(f"Protein Hunter finished: {result}")
