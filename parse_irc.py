"""
Downloads and parses the IRC Poker Dataset into a flat CSV for training.

Usage:
    python parse_irc.py                  # download + parse, save to data/hands.csv
    python parse_irc.py --skip-download  # skip download if already extracted
    python parse_irc.py --sample 100000  # parse only N hands
"""

import argparse
import os
import re
import ssl
import tarfile
import urllib.request
from pathlib import Path

import pandas as pd
from tqdm import tqdm

IRC_URL = "https://poker.cs.ualberta.ca/IRC/IRCdata.tgz"
DATA_DIR = Path("data")
ARCHIVE_PATH = DATA_DIR / "IRCdata.tgz"
EXTRACT_DIR = DATA_DIR / "IRC"
OUTPUT_CSV = DATA_DIR / "hands.csv"

CARD_RE = re.compile(r"[2-9TJQKA][shdcr]", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def download(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {dest}  (this may take a few minutes)...")
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    with urllib.request.urlopen(url, context=ctx) as response:
        total = int(response.headers.get("Content-Length", 0))
        chunk = 1 << 16  # 64 KB
        downloaded = 0
        with open(dest, "wb") as f:
            with tqdm(total=total, unit="B", unit_scale=True, desc="IRC dataset") as bar:
                while True:
                    data = response.read(chunk)
                    if not data:
                        break
                    f.write(data)
                    downloaded += len(data)
                    bar.update(len(data))
    print("Download complete.")


def extract(archive: Path, dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {archive} -> {dest} ...")
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(dest)
    print("Extraction complete.")


# ---------------------------------------------------------------------------
# HDB parsing
# ---------------------------------------------------------------------------

def parse_hdb_line(line: str) -> dict | None:
    """
    HDB line format (space-delimited):
        timestamp table_id hand_id num_players preflop/pot flop/pot turn/pot river/pot [board_cards...]

    The per-street fields are "num_raises/total_pot". Board cards follow after index 7.
    Returns None if the line can't be parsed.
    """
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    try:
        hand_id = parts[2]
        num_players = int(parts[3])
        # Parse pot size from the last played street (format: "num_raises/pot")
        pot_size = 0.0
        for field in parts[4:8]:
            if "/" in field:
                try:
                    pot_size = float(field.split("/")[1])
                except ValueError:
                    pass
        board_cards = [p for p in parts[8:] if CARD_RE.match(p)]
        return {
            "hand_id": hand_id,
            "num_players": num_players,
            "pot_size": pot_size,
            "board_cards": board_cards,
        }
    except (ValueError, IndexError):
        return None


# ---------------------------------------------------------------------------
# PDB parsing
# ---------------------------------------------------------------------------

def parse_action_string(s: str) -> list[str]:
    """
    Converts an action string like 'Brc200f' into tokens ['B','r','c','r200','f'].
    Numeric amounts are attached to the preceding raise action.
    """
    if not s or s.strip() in ("-", ""):
        return []
    tokens = []
    i = 0
    while i < len(s):
        ch = s[i]
        if ch in "BfckrAqQ":
            # collect following digits as amount
            j = i + 1
            while j < len(s) and s[j].isdigit():
                j += 1
            tokens.append(s[i:j])
            i = j
        else:
            i += 1
    return tokens


def parse_pdb_line(line: str) -> dict | None:
    """
    PDB line format (space-delimited):
        player_name timestamp hand_id seat preflop_actions flop_actions turn_actions river_actions bankroll winnings unknown [hole_card1 hole_card2]

    hand_id (parts[2]) matches hdb hand_id (parts[2]).
    Hole cards appear at parts[11:13] only when shown at showdown.
    """
    parts = line.strip().split()
    if len(parts) < 8:
        return None
    try:
        player = parts[0]
        hand_id = parts[2]
        seat = int(parts[3])
        actions = {
            "preflop": parse_action_string(parts[4]),
            "flop": parse_action_string(parts[5]) if len(parts) > 5 else [],
            "turn": parse_action_string(parts[6]) if len(parts) > 6 else [],
            "river": parse_action_string(parts[7]) if len(parts) > 7 else [],
        }
        winnings = float(parts[9]) if len(parts) > 9 else 0.0
        hole_cards = []
        for card_field in (parts[11:13] if len(parts) >= 13 else []):
            if CARD_RE.match(card_field):
                hole_cards.append(card_field[0].upper() + card_field[1].lower())
        return {
            "player": player,
            "hand_id": hand_id,
            "seat": seat,
            "actions": actions,
            "winnings": winnings,
            "hole_cards": hole_cards,
        }
    except (ValueError, IndexError):
        return None


def last_voluntary_action(actions_by_street: dict) -> tuple[str, str]:
    """
    Returns (street, action_char) for the last meaningful action the player
    took (ignoring blinds B and folds that end the hand early).
    """
    order = ["river", "turn", "flop", "preflop"]
    for street in order:
        tokens = actions_by_street.get(street, [])
        # Walk backwards to find a voluntary action
        for tok in reversed(tokens):
            ch = tok[0]
            if ch in "fckrA":
                return street, ch
    return "preflop", "f"


def action_to_label(ch: str) -> int:
    """f -> 0, c/k -> 1, r/A -> 2"""
    if ch == "f":
        return 0
    if ch in "ck":
        return 1
    return 2  # r, A


STREET_INT = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def compute_hand_strength_preflop(hole_cards: list[str]) -> float:
    """Chen formula approximation, normalised to [0, 1]."""
    if len(hole_cards) != 2:
        return 0.5
    try:
        r1 = "23456789TJQKA".index(hole_cards[0][0].upper())
        r2 = "23456789TJQKA".index(hole_cards[1][0].upper())
        suited = hole_cards[0][1].lower() == hole_cards[1][1].lower()
        hi, lo = max(r1, r2), min(r1, r2)
        # Chen base score from high card rank
        chen_ranks = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7, 8, 10]
        score = chen_ranks[hi]
        # Pair bonus
        if hi == lo:
            score = max(score * 2, 5)
        else:
            if suited:
                score += 2
            gap = hi - lo - 1
            if gap == 0:
                pass
            elif gap == 1:
                score -= 1
            elif gap == 2:
                score -= 2
            elif gap == 3:
                score -= 4
            else:
                score -= 5
            if hi < 12 and gap <= 2:  # straight potential bonus
                score += 1
        # Normalise: chen ranges roughly -1 to 20
        return max(0.0, min(1.0, (score + 1) / 21.0))
    except (ValueError, IndexError):
        return 0.5


def compute_hand_strength_postflop(hole_cards: list[str], board_cards: list[str]) -> float:
    """Use treys evaluator; returns normalised 0-1 (higher = stronger)."""
    try:
        from treys import Card, Evaluator
        evaluator = Evaluator()
        hand = [Card.new(c) for c in hole_cards]
        board = [Card.new(c) for c in board_cards]
        score = evaluator.evaluate(board, hand)
        return 1.0 - (score / 7462.0)
    except Exception:
        return 0.5


def build_row(pdb_rec: dict, hdb_rec: dict) -> dict | None:
    """
    Combine one player-hand record with the table-level hand record into
    a feature row ready for training.
    """
    hole = pdb_rec["hole_cards"]
    board = [
        c.upper()[:1] + c[1:].lower()
        for c in hdb_rec["board_cards"]
        if CARD_RE.match(c)
    ]
    num_players = hdb_rec["num_players"]
    pot_size = hdb_rec["pot_size"]

    street, action_ch = last_voluntary_action(pdb_rec["actions"])
    action_label = action_to_label(action_ch)

    # Hand strength
    if len(hole) == 2:
        postflop_board = board[:{"preflop": 0, "flop": 3, "turn": 4, "river": 5}[street]]
        if len(postflop_board) >= 3:
            hand_strength = compute_hand_strength_postflop(hole, postflop_board)
        else:
            hand_strength = compute_hand_strength_preflop(hole)
    else:
        hand_strength = None  # unknown hole cards, skip

    if hand_strength is None:
        return None

    # Position: use seat mod num_players as a proxy (0=early, higher=later)
    position = pdb_rec["seat"] % num_players

    # Simplified pot odds: rough estimate since IRC doesn't store bet sizes
    # Use 1/num_players as a proxy for the pot-odds fraction needed to call
    pot_odds = 1.0 / max(num_players, 2)

    return {
        "hand_strength": round(hand_strength, 4),
        "pot_odds": round(pot_odds, 4),
        "pot_size": pot_size,
        "position": position,
        "num_players": num_players,
        "street": STREET_INT[street],
        "is_last_to_act": int(position == num_players - 1),
        "winnings": pdb_rec["winnings"],
        "action_label": action_label,
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def read_tar_text(tf: tarfile.TarFile, member: tarfile.TarInfo) -> list[str]:
    """Read lines from a tarfile member as text."""
    f = tf.extractfile(member)
    if f is None:
        return []
    return f.read().decode(errors="replace").splitlines()


def parse_month_tarball(tgz_path: Path, sample_limit: int | None, rows: list):
    """Open a holdem.YYYYMM.tgz and parse all hdb + pdb entries inside it."""
    try:
        with tarfile.open(tgz_path, "r:gz", errorlevel=0) as tf:
            try:
                members = tf.getmembers()
            except EOFError:
                members = tf.members  # use whatever was read before EOF

            # Find hdb file
            hdb_members = [m for m in members if m.name.endswith("/hdb")]
            if not hdb_members:
                return
            hdb_lines = read_tar_text(tf, hdb_members[0])
            hdb_lookup = {}
            for line in hdb_lines:
                r = parse_hdb_line(line)
                if r:
                    hdb_lookup[r["hand_id"]] = r

            # Find all pdb files
            pdb_members = [m for m in members if "/pdb/" in m.name and m.isfile()]
            for pdb_member in pdb_members:
                if sample_limit and len(rows) >= sample_limit:
                    return
                for line in read_tar_text(tf, pdb_member):
                    if sample_limit and len(rows) >= sample_limit:
                        return
                    pdb_rec = parse_pdb_line(line)
                    if not pdb_rec:
                        continue
                    hdb_rec = hdb_lookup.get(pdb_rec["hand_id"])
                    if not hdb_rec:
                        continue
                    row = build_row(pdb_rec, hdb_rec)
                    if row:
                        rows.append(row)
    except (tarfile.TarError, OSError, EOFError):
        pass


def build_dataset(extract_dir: Path, output_csv: Path, sample_limit: int | None = None):
    # The main archive extracts to IRCdata/ which contains holdem.YYYYMM.tgz files
    irc_data_dir = extract_dir / "IRCdata"
    if not irc_data_dir.exists():
        irc_data_dir = extract_dir  # fallback

    month_archives = sorted(irc_data_dir.glob("holdem.*.tgz"))
    if not month_archives:
        print(f"ERROR: Could not find holdem.*.tgz files inside {irc_data_dir}")
        return

    print(f"Found {len(month_archives)} month archives.")
    rows = []
    for tgz_path in tqdm(month_archives, desc="Parsing months"):
        if sample_limit and len(rows) >= sample_limit:
            break
        parse_month_tarball(tgz_path, sample_limit, rows)

    df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\nSaved {len(df):,} rows to {output_csv}")
    print(df["action_label"].value_counts().rename({0: "fold", 1: "call", 2: "raise"}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip download/extraction if already done")
    parser.add_argument("--sample", type=int, default=None,
                        help="Stop after this many parsed rows (faster for testing)")
    args = parser.parse_args()

    if not args.skip_download:
        if not ARCHIVE_PATH.exists():
            download(IRC_URL, ARCHIVE_PATH)
        if not EXTRACT_DIR.exists() or not any(EXTRACT_DIR.iterdir()):
            extract(ARCHIVE_PATH, EXTRACT_DIR)
    else:
        print("Skipping download/extraction.")

    build_dataset(EXTRACT_DIR, OUTPUT_CSV, sample_limit=args.sample)


if __name__ == "__main__":
    main()
