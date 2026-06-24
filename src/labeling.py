"""Labeling kontrak terbaik menggunakan Double Dummy Solver via endplay library.

Menggunakan package `endplay` (pip install endplay>=0.4.7) yang menyediakan
wrapper Python yang bersih untuk Bo Haglund DDS — tanpa perlu DLL manual
atau path hardcode.

Fungsi utama yang dipakai: endplay.dds.calc_dd_table — menghitung trik
maksimum untuk semua 20 kontrak (5 strain × 4 declarer) sekaligus.

Label yang dihasilkan:
  best_contract_strain    — C/D/H/S/N/P  (untuk Stage 1)
  best_contract_category  — partscore/game/small_slam/grand_slam/pass  (untuk Stage 2)
  best_contract_level     — level kontrak (1-7, 0=PASS)
  best_contract_token     — teks kontrak, e.g. '4HN='
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# endplay import (graceful fallback jika tidak terinstall)
# ---------------------------------------------------------------------------

try:
    from endplay.types import Deal, Denom, Player, Vul
    from endplay.dds import calc_dd_table
    _ENDPLAY_AVAILABLE = True
except ImportError:
    _ENDPLAY_AVAILABLE = False


def dds_available() -> bool:
    """True jika endplay terinstall dan DDS bisa digunakan."""
    if not _ENDPLAY_AVAILABLE:
        return False
    # Quick sanity check: coba buat deal sederhana
    try:
        _test_deal = Deal("N:AKQJT98765432... ...")
        return True
    except Exception:
        return True  # Import berhasil, anggap tersedia


# ---------------------------------------------------------------------------
# Mapping konstan
# ---------------------------------------------------------------------------

# Mapping dari kode suit proyek → endplay Denom
_STRAIN_TO_DENOM: Dict[str, object] = {}  # diisi setelah import

# Mapping dari endplay Denom → kode suit proyek
_DENOM_TO_STRAIN: Dict[object, str] = {}

# Mapping dari kode seat proyek → endplay Player
_SEAT_TO_PLAYER: Dict[str, object] = {}

if _ENDPLAY_AVAILABLE:
    _STRAIN_TO_DENOM = {
        "S": Denom.spades,
        "H": Denom.hearts,
        "D": Denom.diamonds,
        "C": Denom.clubs,
        "N": Denom.nt,
    }
    _DENOM_TO_STRAIN = {v: k for k, v in _STRAIN_TO_DENOM.items()}
    _SEAT_TO_PLAYER = {
        "N": Player.north,
        "E": Player.east,
        "S": Player.south,
        "W": Player.west,
    }

# Vulnerability mapping: kode BBO → endplay Vul
_VULN_MAP: Dict[str, object] = {}
if _ENDPLAY_AVAILABLE:
    _VULN_MAP = {
        "o":    Vul.none,  "none": Vul.none,
        "n":    Vul.ns,    "ns":   Vul.ns,
        "e":    Vul.ew,    "ew":   Vul.ew,
        "b":    Vul.both,  "both": Vul.both,
    }

# NS vulnerability dari endplay Vul object
_NS_VUL_FROM_VUL: Dict[object, bool] = {}
if _ENDPLAY_AVAILABLE:
    _NS_VUL_FROM_VUL = {
        Vul.none: False,
        Vul.ns:   True,
        Vul.ew:   False,
        Vul.both: True,
    }


# ---------------------------------------------------------------------------
# Konstanta scoring bridge
# ---------------------------------------------------------------------------

_TRICK_VALUE: Dict[str, int] = {"C": 20, "D": 20, "H": 30, "S": 30, "N": 30}


def bridge_score(level: int, strain: str, tricks: int, vul: bool) -> int:
    """Hitung skor bridge undoubled untuk NS sebagai declarer."""
    needed = level + 6
    if tricks < needed:
        down = needed - tricks
        return -down * (100 if vul else 50)

    if strain == "N":
        trick_pts = 40 + 30 * (level - 1)
    else:
        trick_pts = _TRICK_VALUE[strain] * level

    if level == 7:
        bonus = 1500 if vul else 1000
    elif level == 6:
        bonus = 750 if vul else 500
    elif trick_pts >= 100:
        bonus = 500 if vul else 300
    else:
        bonus = 50

    ot_val = 30 if strain == "N" else _TRICK_VALUE[strain]
    return trick_pts + bonus + ot_val * (tricks - needed)


def contract_category(level: int, strain: str) -> str:
    """Kategori kontrak berdasarkan level + suit."""
    if level == 7:
        return "grand_slam"
    if level == 6:
        return "small_slam"
    if strain == "N" and level >= 3:
        return "game"
    if strain in {"H", "S"} and level >= 4:
        return "game"
    if strain in {"C", "D"} and level >= 5:
        return "game"
    return "partscore"


# ---------------------------------------------------------------------------
# Parsing hand dari format parser.py
# ---------------------------------------------------------------------------

def _parse_hand(hand_str: str) -> Dict[str, List[str]]:
    """Parse 'SKT87HT76D873C743' ke {'S':['K','T','8','7'], ...}."""
    suits: Dict[str, List[str]] = {"S": [], "H": [], "D": [], "C": []}
    if not hand_str or not isinstance(hand_str, str):
        return suits
    parts = re.split(r"([SHDC])", hand_str.upper())
    current = None
    for part in parts:
        if part in suits:
            current = part
        elif current and part:
            suits[current] = list(part)
    return suits


def _hand_to_pbn(hand_dict: Dict[str, List[str]]) -> str:
    """Konversi dict {S:[...], H:[...], D:[...], C:[...]} ke PBN hand string.
    
    Format PBN: 'AKQT.JT98.765.432' (spades.hearts.diamonds.clubs)
    Kartu kosong direpresentasikan sebagai '-'
    """
    parts = []
    for suit in ["S", "H", "D", "C"]:
        cards = hand_dict.get(suit, [])
        parts.append("".join(cards) if cards else "-")
    return ".".join(parts)


def _board_hand_counts(row: pd.Series) -> Dict[str, int]:
    """Hitung jumlah kartu per hand dari baris parsed_boards.csv."""
    return {
        "N": sum(len(v) for v in _parse_hand(str(row.get("north_hand_norm", ""))).values()),
        "S": sum(len(v) for v in _parse_hand(str(row.get("south_hand_norm", ""))).values()),
        "E": sum(len(v) for v in _parse_hand(str(row.get("east_hand_norm", ""))).values()),
        "W": sum(len(v) for v in _parse_hand(str(row.get("west_hand_norm", ""))).values()),
    }


def is_valid_bridge_board(row: pd.Series) -> bool:
    """True jika setiap hand punya 13 kartu dan total kartu 52."""
    counts = _board_hand_counts(row)
    total = sum(counts.values())
    return total == 52 and all(count == 13 for count in counts.values())


# ---------------------------------------------------------------------------
# Konversi ke endplay Deal
# ---------------------------------------------------------------------------

def _row_to_endplay_deal(row: pd.Series) -> Optional[object]:
    """Konversi baris parsed_boards.csv ke endplay Deal object.
    
    Format PBN yang digunakan endplay:
    'N:north_hand east_hand south_hand west_hand'
    dimana setiap hand berformat 'SPADES.HEARTS.DIAMONDS.CLUBS'
    """
    if not _ENDPLAY_AVAILABLE:
        return None

    north = _parse_hand(str(row.get("north_hand_norm", "")))
    east  = _parse_hand(str(row.get("east_hand_norm", "")))
    south = _parse_hand(str(row.get("south_hand_norm", "")))
    west  = _parse_hand(str(row.get("west_hand_norm", "")))

    north_pbn = _hand_to_pbn(north)
    east_pbn  = _hand_to_pbn(east)
    south_pbn = _hand_to_pbn(south)
    west_pbn  = _hand_to_pbn(west)

    pbn_str = f"N:{north_pbn} {east_pbn} {south_pbn} {west_pbn}"
    try:
        return Deal(pbn_str)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# DDS computation via endplay
# ---------------------------------------------------------------------------

def compute_best_contract(
    row: pd.Series,
    vulnerability: Optional[str] = None,
) -> Optional[Dict]:
    """Hitung kontrak NS terbaik menggunakan endplay DDS.

    Args:
        row: Satu baris dari parsed_boards.csv (sudah tervalidasi 52 kartu)
        vulnerability: kode BBO ('o','n','e','b') atau None

    Returns:
        Dict: level, strain, tricks, token, category, is_pass
        None: jika endplay tidak tersedia atau deal tidak valid
    """
    if not _ENDPLAY_AVAILABLE:
        return None

    deal = _row_to_endplay_deal(row)
    if deal is None:
        return None

    try:
        dd_table = calc_dd_table(deal)
    except Exception:
        return None

    vuln_key = str(vulnerability or "o").lower()
    vul_obj  = _VULN_MAP.get(vuln_key, Vul.none)
    ns_vul   = _NS_VUL_FROM_VUL.get(vul_obj, False)

    best_score = -1
    best = None

    # Iterasi semua kombinasi strain (SHDCN) × declarer NS
    for strain_str, denom in _STRAIN_TO_DENOM.items():
        for seat_str, player in [("N", Player.north), ("S", Player.south)]:
            try:
                max_tricks = dd_table[denom, player]
            except (KeyError, TypeError, Exception):
                continue

            # Cari level tertinggi yang bisa making
            for level in range(7, 0, -1):
                if max_tricks >= level + 6:
                    score = bridge_score(level, strain_str, max_tricks, ns_vul)
                    if score > best_score:
                        best_score = score
                        best = {
                            "level":    level,
                            "strain":   strain_str,
                            "tricks":   max_tricks,
                            "declarer": seat_str,
                            "score":    score,
                            "token":    f"{level}{strain_str}{seat_str}=",
                        }
                    break

    if best is None:
        return {
            "level": 0, "strain": "P", "tricks": 0,
            "token": "PASS", "category": "pass", "is_pass": True,
        }

    return {
        "level":    best["level"],
        "strain":   best["strain"],
        "tricks":   best["tricks"],
        "token":    best["token"],
        "category": contract_category(best["level"], best["strain"]),
        "is_pass":  False,
    }


# ---------------------------------------------------------------------------
# Label satu baris dari parsed_boards.csv
# ---------------------------------------------------------------------------

def label_row(row: pd.Series) -> Dict:
    """Hitung label kontrak terbaik untuk satu baris parsed_boards.csv."""
    _empty = {
        "best_contract_strain":   None,
        "best_contract_category": None,
        "best_contract_level":    None,
        "best_contract_token":    None,
        "dds_available":          False,
    }

    if not _ENDPLAY_AVAILABLE:
        return _empty

    if not is_valid_bridge_board(row):
        return _empty

    vuln = str(row.get("vulnerability_code", row.get("vulnerability", "o"))).lower()
    result = compute_best_contract(row, vuln)

    if result is None:
        return _empty

    return {
        "best_contract_strain":   result["strain"],
        "best_contract_category": result["category"],
        "best_contract_level":    result["level"],
        "best_contract_token":    result["token"],
        "dds_available":          True,
    }


# ---------------------------------------------------------------------------
# Label seluruh dataset
# ---------------------------------------------------------------------------

def label_dataset(df: pd.DataFrame, verbose: bool = True, progress_every: int = 1000) -> pd.DataFrame:
    """Tambahkan kolom label DDS ke parsed_boards DataFrame.

    Args:
        df: DataFrame dari parsed_boards.csv
        verbose: Tampilkan progress
        progress_every: Cetak progress setiap N baris

    Returns:
        DataFrame dengan kolom best_contract_* tambahan
    """
    if not _ENDPLAY_AVAILABLE:
        if verbose:
            print("PERINGATAN: endplay tidak terinstall.")
            print("Install dengan: pip install endplay>=0.4.7")
        return df

    if verbose:
        print(f"endplay DDS siap. Melabeli {len(df)} board...")

    label_rows = []
    for i, (_, row) in enumerate(df.iterrows()):
        label_rows.append(label_row(row))
        if verbose and progress_every > 0 and (i + 1) % progress_every == 0:
            pct = (i + 1) / len(df) * 100
            print(f"  {i + 1}/{len(df)} ({pct:.0f}%) selesai")

    df_labels = pd.DataFrame(label_rows, index=df.index)
    result = pd.concat([df, df_labels], axis=1)

    if verbose:
        n_pass  = (result["best_contract_strain"] == "P").sum()
        n_valid = result["best_contract_strain"].notna().sum()
        n_fail  = len(df) - n_valid
        print(f"\nSelesai. Valid: {n_valid} | PASS: {n_pass} | Gagal: {n_fail}")

    return result


def _label_row_from_dict(row_dict: dict) -> Dict:
    """Helper for multiprocessing: accept a plain dict and return label dict."""
    return label_row(pd.Series(row_dict))


def label_dataset_parallel(df: pd.DataFrame, processes: Optional[int] = None,
                           chunksize: int = 100, verbose: bool = True,
                           progress_every: int = 1000) -> pd.DataFrame:
    """Parallelized version of label_dataset using multiprocessing.Pool.

    Notes:
    - Pada Windows, setiap worker akan melakukan import endplay saat pertama digunakan.
    - Bisa signifikan mengurangi waktu pada mesin multi-core.
    """
    if not _ENDPLAY_AVAILABLE:
        if verbose:
            print("PERINGATAN: endplay tidak terinstall. Install: pip install endplay>=0.4.7")
        return df

    import multiprocessing as mp

    records = df.to_dict(orient="records")
    n = len(records)

    if processes is None:
        processes = max(1, mp.cpu_count() - 1)

    if verbose:
        print(f"endplay DDS siap. Melabeli {n} board secara paralel ({processes} worker)...")

    results = []
    with mp.Pool(processes=processes) as pool:
        for i, res in enumerate(pool.imap(_label_row_from_dict, records, chunksize)):
            results.append(res)
            if verbose and progress_every > 0 and (i + 1) % progress_every == 0:
                pct = (i + 1) / n * 100
                print(f"  {i + 1}/{n} ({pct:.0f}%) selesai")

    df_labels = pd.DataFrame(results, index=df.index)
    result = pd.concat([df, df_labels], axis=1)

    if verbose:
        n_pass  = (result["best_contract_strain"] == "P").sum()
        n_valid = result["best_contract_strain"].notna().sum()
        n_fail  = n - n_valid
        print(f"\nSelesai. Valid: {n_valid} | PASS: {n_pass} | Gagal: {n_fail}")

    return result


def label_dataset_threaded(df: pd.DataFrame, max_workers: Optional[int] = None,
                           chunksize: int = 100, verbose: bool = True,
                           progress_every: int = 1000) -> pd.DataFrame:
    """Threaded version menggunakan concurrent.futures.ThreadPoolExecutor.

    Lebih aman dijalankan dari Jupyter di Windows dibanding multiprocessing.Pool
    karena tidak spawn proses Python baru.
    """
    if not _ENDPLAY_AVAILABLE:
        if verbose:
            print("PERINGATAN: endplay tidak terinstall. Install: pip install endplay>=0.4.7")
        return df

    from concurrent.futures import ThreadPoolExecutor, as_completed
    import os

    records = df.to_dict(orient="records")
    n = len(records)
    if max_workers is None:
        try:
            max_workers = min(32, (os.cpu_count() or 1) + 4)
        except Exception:
            max_workers = 4

    if verbose:
        print(f"endplay DDS siap. Melabeli {n} board dengan threads ({max_workers})...")

    results: List[Optional[Dict]] = [None] * n
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_label_row_from_dict, rec): idx for idx, rec in enumerate(records)}
        for i, fut in enumerate(as_completed(futures)):
            idx = futures[fut]
            try:
                results[idx] = fut.result()
            except Exception:
                results[idx] = {
                    "best_contract_strain":   None,
                    "best_contract_category": None,
                    "best_contract_level":    None,
                    "best_contract_token":    None,
                    "dds_available":          False,
                }
            if verbose and progress_every > 0 and (i + 1) % progress_every == 0:
                pct = (i + 1) / n * 100
                print(f"  {i + 1}/{n} ({pct:.0f}%) selesai")

    _empty_label = {
        "best_contract_strain":   None,
        "best_contract_category": None,
        "best_contract_level":    None,
        "best_contract_token":    None,
        "dds_available":          False,
    }
    typed_results = [r if r is not None else _empty_label for r in results]

    df_labels = pd.DataFrame(typed_results, index=df.index)
    result = pd.concat([df, df_labels], axis=1)

    if verbose:
        n_pass  = (result["best_contract_strain"] == "P").sum()
        n_valid = result["best_contract_strain"].notna().sum()
        n_fail  = n - n_valid
        print(f"\nSelesai. Valid: {n_valid} | PASS: {n_pass} | Gagal: {n_fail}")

    return result


# ---------------------------------------------------------------------------
# Deduplikasi board (satu board = satu record, bukan open+closed)
# ---------------------------------------------------------------------------

def deduplicate_boards(df: pd.DataFrame) -> pd.DataFrame:
    """Pertahankan hanya satu record per board (prioritaskan open room).

    Parser menghasilkan 2 record per board (open + closed room).
    Untuk labeling, kita hanya butuh satu karena hand-nya sama.
    """
    if "room" not in df.columns:
        return df
    open_rooms   = df[df["room"] == "open"]
    closed_rooms = df[df["room"] == "closed"]
    key = ["source_file", "board_number"]
    open_keys = set(zip(open_rooms["source_file"], open_rooms["board_number"]))
    closed_only = closed_rooms[
        ~closed_rooms.apply(lambda r: (r["source_file"], r["board_number"]) in open_keys, axis=1)
    ]
    result = pd.concat([open_rooms, closed_only], ignore_index=True)
    return result


def repair_missing_boards(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pisahkan board yang valid dari board yang masih kehilangan data penting.

    Fungsi ini tetap memilih satu record per board, lalu menyaring board yang
    tidak punya 13 kartu lengkap di setiap tangan. Board yang rusak dikembalikan
    terpisah agar bisa dianalisis tanpa menjatuhkan DDS.
    """
    df_unique = deduplicate_boards(df).copy()
    valid_mask = df_unique.apply(is_valid_bridge_board, axis=1)

    df_valid   = df_unique[valid_mask].copy().reset_index(drop=True)
    df_invalid = df_unique[~valid_mask].copy().reset_index(drop=True)

    if not df_valid.empty:
        df_valid["board_data_status"] = "valid"
    if not df_invalid.empty:
        df_invalid["board_data_status"] = "missing_or_invalid"

    return df_valid, df_invalid


if __name__ == "__main__":
    print(f"endplay tersedia: {_ENDPLAY_AVAILABLE}")
    print(f"DDS siap: {dds_available()}")

    parsed_csv = Path("data/parsed/parsed_boards.csv")
    if not parsed_csv.exists():
        parsed_csv = Path("data/processed/parsed_boards.csv")

    df = pd.read_csv(parsed_csv)
    print(f"Loaded {len(df)} baris")

    # Deduplikasi sebelum labeling untuk efisiensi
    df_unique = deduplicate_boards(df)
    print(f"Setelah dedup: {len(df_unique)} unique boards")

    df_labeled = label_dataset(df_unique, verbose=True)
    out = Path("data/processed/labeled_boards.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df_labeled.to_csv(out, index=False)
    print(f"Disimpan ke {out}")
