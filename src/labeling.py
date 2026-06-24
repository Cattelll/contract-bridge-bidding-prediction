"""Labeling kontrak terbaik menggunakan Double Dummy Solver (DDS) via ctypes.

Menggunakan dds.dll dari C:\\dds (Bo Haglund DDS v2.9.x).
DLL path: C:\\dds\\src\\dds.dll
Deps    : C:\\Program Files\\Git\\mingw64\\bin  (libstdc++, libgcc, libgomp, dll)

Fungsi utama yang dipakai: CalcDDtable — menghitung trik maksimum untuk
semua 20 kontrak (5 strain × 4 declarer) sekaligus.

Label yang dihasilkan:
  best_contract_strain    — C/D/H/S/N/P  (untuk Stage 1)
  best_contract_category  — partscore/game/small_slam/grand_slam/pass  (untuk Stage 2)
  best_contract_level     — level kontrak (1-7, 0=PASS)
  best_contract_token     — teks kontrak, e.g. '4HN='
"""

from __future__ import annotations

import ctypes
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# DDS DLL setup
# ---------------------------------------------------------------------------

DDS_DLL_PATH = r"C:\dds\src\dds.dll"

# Mingw64 dirs yang dibutuhkan sebagai dependency DLL
_MINGW_DIRS = [
    r"C:\Program Files\Git\mingw64\bin",
    r"C:\Users\acerp\AppData\Local\Microsoft\WinGet\Packages\BrechtSanders.WinLibs.POSIX.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe\mingw64\bin",
]

DDS_HANDS = 4
DDS_SUITS = 4
DDS_STRAINS = 5


class _ddTableDeal(ctypes.Structure):
    """DDS struct: cards[hand][suit] bitmask (bit N = rank N, 2-14)."""
    _fields_ = [("cards", (ctypes.c_uint * DDS_SUITS) * DDS_HANDS)]


class _ddTableResults(ctypes.Structure):
    """DDS struct: resTable[strain][hand] = max tricks for that declarer."""
    _fields_ = [("resTable", (ctypes.c_int * DDS_HANDS) * DDS_STRAINS)]


# DDS encoding constants
# Hand order: 0=N, 1=E, 2=S, 3=W
# Suit order: 0=S(pades), 1=H(earts), 2=D(iamonds), 3=C(lubs)
# Strain order: 0=S, 1=H, 2=D, 3=C, 4=NT
_HAND_IDX = {"N": 0, "E": 1, "S": 2, "W": 3}
_SUIT_IDX = {"S": 0, "H": 1, "D": 2, "C": 3}
_STRAIN_IDX = {"S": 0, "H": 1, "D": 2, "C": 3, "N": 4}
_STRAIN_STR = {0: "S", 1: "H", 2: "D", 3: "C", 4: "N"}

_RANK_BIT = {
    "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7,
    "8": 8, "9": 9, "T": 10, "J": 11, "Q": 12, "K": 13, "A": 14,
}

_dds_lib: Optional[ctypes.WinDLL] = None


def _load_dds() -> Optional[ctypes.WinDLL]:
    """Load dds.dll, kembalikan None jika gagal."""
    global _dds_lib
    if _dds_lib is not None:
        return _dds_lib

    for d in _MINGW_DIRS:
        if os.path.isdir(d):
            try:
                os.add_dll_directory(d)
            except Exception:
                pass

    try:
        lib = ctypes.WinDLL(DDS_DLL_PATH)
        lib.CalcDDtable.argtypes = [_ddTableDeal, ctypes.POINTER(_ddTableResults)]
        lib.CalcDDtable.restype = ctypes.c_int
        _dds_lib = lib
        return lib
    except Exception:
        return None


def dds_available() -> bool:
    """True jika dds.dll bisa di-load."""
    return _load_dds() is not None


# ---------------------------------------------------------------------------
# Konstanta scoring bridge
# ---------------------------------------------------------------------------

_TRICK_VALUE: Dict[str, int] = {"C": 20, "D": 20, "H": 30, "S": 30, "N": 30}

_VULN_MAP: Dict[str, Tuple[bool, bool]] = {
    "o": (False, False), "none": (False, False),
    "n": (True, False),  "ns": (True, False),
    "e": (False, True),  "ew": (False, True),
    "b": (True, True),   "both": (True, True),
}


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


def _hand_to_bitmask(cards: List[str]) -> int:
    """Konversi list kartu ke bitmask DDS (bit N = rank N)."""
    return sum(1 << _RANK_BIT[c] for c in cards if c in _RANK_BIT)


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
# DDS computation
# ---------------------------------------------------------------------------

def compute_best_contract(
    hands: Dict[str, Dict[str, List[str]]],
    vulnerability: Optional[str],
) -> Optional[Dict]:
    """Hitung kontrak NS terbaik menggunakan DDS.

    Args:
        hands: {seat: {suit: [kartu]}} untuk seat N,E,S,W dan suit S,H,D,C
        vulnerability: kode BBO ('o','n','e','b') atau None

    Returns:
        Dict: level, strain, tricks, token, category, is_pass
        None: jika DDS tidak tersedia
    """
    lib = _load_dds()
    if lib is None:
        return None

    deal = _ddTableDeal()
    for seat, suit_cards in hands.items():
        h = _HAND_IDX.get(seat)
        if h is None:
            continue
        for suit, cards in suit_cards.items():
            s = _SUIT_IDX.get(suit)
            if s is not None:
                deal.cards[h][s] = _hand_to_bitmask(cards)

    res = _ddTableResults()
    ret = lib.CalcDDtable(deal, ctypes.byref(res))
    if ret != 1:
        return None

    vuln_key = str(vulnerability or "o").lower()
    ns_vul, _ = _VULN_MAP.get(vuln_key, (False, False))

    best_score = -1
    best = None

    # Iterasi semua kontrak NS (North=idx 0, South=idx 2)
    for strain_idx, strain in _STRAIN_STR.items():
        for seat, seat_idx in [("N", 0), ("S", 2)]:
            max_tricks = res.resTable[strain_idx][seat_idx]
            # Cari level tertinggi yang bisa making
            for level in range(7, 0, -1):
                if max_tricks >= level + 6:
                    score = bridge_score(level, strain, max_tricks, ns_vul)
                    if score > best_score:
                        best_score = score
                        best = {
                            "level":    level,
                            "strain":   strain,
                            "tricks":   max_tricks,
                            "declarer": seat,
                            "score":    score,
                            "token":    f"{level}{strain}{seat}=",
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
    if not is_valid_bridge_board(row):
        return {
            "best_contract_strain":   None,
            "best_contract_category": None,
            "best_contract_level":    None,
            "best_contract_token":    None,
            "dds_available":          False,
        }

    hands = {
        "N": _parse_hand(str(row.get("north_hand_norm", ""))),
        "S": _parse_hand(str(row.get("south_hand_norm", ""))),
        "E": _parse_hand(str(row.get("east_hand_norm", ""))),
        "W": _parse_hand(str(row.get("west_hand_norm", ""))),
    }
    vuln = str(row.get("vulnerability_code", row.get("vulnerability", "o"))).lower()
    result = compute_best_contract(hands, vuln)

    if result is None:
        return {
            "best_contract_strain":   None,
            "best_contract_category": None,
            "best_contract_level":    None,
            "best_contract_token":    None,
            "dds_available":          False,
        }

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

    Returns:
        DataFrame dengan kolom best_contract_* tambahan
    """
    if not dds_available():
        if verbose:
            print(f"PERINGATAN: DDS tidak tersedia. DLL path: {DDS_DLL_PATH}")
            print("Pastikan C:\\dds\\src\\dds.dll ada dan mingw64 bin di path.")
        return df

    if verbose:
        print(f"DDS siap. Melabeli {len(df)} board...")

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
    # Convert to Series locally so existing label_row can be reused
    return label_row(pd.Series(row_dict))


def label_dataset_parallel(df: pd.DataFrame, processes: Optional[int] = None,
                           chunksize: int = 100, verbose: bool = True,
                           progress_every: int = 1000) -> pd.DataFrame:
    """Parallelized version of label_dataset using multiprocessing.Pool.

    Notes:
    - On Windows each worker will load the DDS DLL when first needed.
    - This can significantly reduce wall-clock time on multi-core machines.
    """
    if not dds_available():
        if verbose:
            print(f"PERINGATAN: DDS tidak tersedia. DLL path: {DDS_DLL_PATH}")
        return df

    import multiprocessing as mp

    records = df.to_dict(orient="records")
    n = len(records)

    if processes is None:
        processes = max(1, mp.cpu_count() - 1)

    if verbose:
        print(f"DDS siap. Melabeli {n} board secara paralel ({processes} worker)...")

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
    """Threaded version using concurrent.futures.ThreadPoolExecutor.

    This is safer to run from Jupyter on Windows than multiprocessing.Pool
    because it doesn't spawn new Python interpreter processes.
    """
    if not dds_available():
        if verbose:
            print(f"PERINGATAN: DDS tidak tersedia. DLL path: {DDS_DLL_PATH}")
        return df

    from concurrent.futures import ThreadPoolExecutor, as_completed

    records = df.to_dict(orient="records")
    n = len(records)
    if max_workers is None:
        try:
            max_workers = min(32, (os.cpu_count() or 1) + 4)
        except Exception:
            max_workers = 4

    if verbose:
        print(f"DDS siap. Melabeli {n} board dengan threads ({max_workers})...")

    results: List[Optional[Dict]] = [None] * n
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_label_row_from_dict, rec): idx for idx, rec in enumerate(records)}
        for i, fut in enumerate(as_completed(futures)):
            idx = futures[fut]
            try:
                results[idx] = fut.result()
            except Exception as e:
                results[idx] = {
                    "best_contract_strain": None,
                    "best_contract_category": None,
                    "best_contract_level": None,
                    "best_contract_token": None,
                    "dds_available": False,
                }
            if verbose and progress_every > 0 and (i + 1) % progress_every == 0:
                pct = (i + 1) / n * 100
                print(f"  {i + 1}/{n} ({pct:.0f}%) selesai")

    typed_results = [
        res if res is not None else {
            "best_contract_strain": None,
            "best_contract_category": None,
            "best_contract_level": None,
            "best_contract_token": None,
            "dds_available": False,
        }
        for res in results
    ]

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
    # Gunakan open room; tambahkan closed room jika board belum ada
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

    df_valid = df_unique[valid_mask].copy().reset_index(drop=True)
    df_invalid = df_unique[~valid_mask].copy().reset_index(drop=True)

    if not df_valid.empty:
        df_valid["board_data_status"] = "valid"
    if not df_invalid.empty:
        df_invalid["board_data_status"] = "missing_or_invalid"

    return df_valid, df_invalid


if __name__ == "__main__":
    print(f"DDS tersedia: {dds_available()}")

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
