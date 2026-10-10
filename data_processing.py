"""
data_processing.py
Data preprocessing, spatial grid binning, and exact playing time calculation.
"""

import numpy as np
import pandas as pd

# Standard StatsBomb pitch dimensions
PITCH_LENGTH = 120.0
PITCH_WIDTH = 80.0

# Standardized player names for clean visualizations
NAME_MAPPING = {
    "Rafael Alexandre Conceição Leão": "Rafael Leão",
    "Jeremy Doku": "Jérémy Doku",
    "Vladimír Coufal": "Vladimir Coufal",
    "Francisco Fernandes Conceição": "Francisco Conceição",
    "Dušan Tadić": "Dušan Tadić",
    "Ivan Perišić": "Ivan Perisic",
    "Bruno Miguel Borges Fernandes": "Bruno Fernandes",
    "Nicholas Williams Arthuer": "Nico Williams",
    "Lamine Yamal Nasraoui Ebana": "Lamine Yamal",
    "Kylian Mbappé Lottin": "Kylian Mbappé",
    "Cody Mathès Gakpo": "Cody Gakpo",
    "Dominik Szoboszlai": "Dominik Szoboszlai",
    "Christian Dannemann Eriksen": "Christian Eriksen",
    "Lukáš Haraslín": "Lukas Haraslin",
    "Andrija Živković": "Andrija Zivkovic",
}


def get_bin(x: float, y: float, bins_x: int, bins_y: int):
    """Converts (x, y) pitch coordinates into grid cell indices (bx, by)."""
    bx = int(np.clip(x / (PITCH_LENGTH / bins_x), 0, bins_x - 1))
    by = int(np.clip(y / (PITCH_WIDTH / bins_y), 0, bins_y - 1))
    return bx, by


def extract_coords(col_value):
    """Extracts (x, y) float coordinates from location lists/arrays."""
    if isinstance(col_value, (list, np.ndarray)) and len(col_value) >= 2:
        return float(col_value[0]), float(col_value[1])
    return np.nan, np.nan


def load_and_preprocess_data(filepath: str = "euro_2024_events.parquet"):
    """Loads parquet dataset, applies name mapping and extracts coordinates."""
    print(f"-> Loading dataset from: {filepath}")
    df = pd.read_parquet(filepath)

    df["player"] = df["player"].replace(NAME_MAPPING)

    coords = df["location"].apply(extract_coords)
    df["x"] = [c[0] for c in coords]
    df["y"] = [c[1] for c in coords]
    return df


def prepare_events(df: pd.DataFrame, min_carry_distance: float = 3.0):
    """Filters shots, completed passes, and progressive carries."""
    print(f"-> Filtering actions (Carry threshold: >= {min_carry_distance}m)...")

    # 1. Shots for Markov chain transition fitting
    shots = df[df["type"] == "Shot"].dropna(subset=["x", "y"]).copy()
    shots["is_goal"] = (shots["shot_outcome"] == "Goal").astype(int)

    # 2. Completed passes
    passes = (
        df[(df["type"] == "Pass") & (df["pass_outcome"].isna())]
        .dropna(subset=["x", "y"])
        .copy()
    )
    p_ends = passes["pass_end_location"].apply(extract_coords)
    passes["end_x"] = [c[0] for c in p_ends]
    passes["end_y"] = [c[1] for c in p_ends]
    passes = passes.dropna(subset=["end_x", "end_y"])

    # 3. Progressive carries
    carries = df[df["type"] == "Carry"].dropna(subset=["x", "y"]).copy()
    c_ends = carries["carry_end_location"].apply(extract_coords)
    carries["end_x"] = [c[0] for c in c_ends]
    carries["end_y"] = [c[1] for c in c_ends]
    carries = carries.dropna(subset=["end_x", "end_y"])

    carries["dist"] = np.sqrt(
        (carries["end_x"] - carries["x"]) ** 2
        + (carries["end_y"] - carries["y"]) ** 2
    )
    carries = carries[carries["dist"] >= min_carry_distance].copy()

    return shots, passes, carries


def calculate_player_minutes(df: pd.DataFrame):
    """Calculates minutes played per player accounting for extra time and subs."""
    print("-> Calculating exact minutes played per player...")
    match_durations = df.groupby("match_id")["minute"].max().to_dict()
    player_minutes_dict = {}

    for match_id, match_events in df.groupby("match_id"):
        has_extra_time = match_events["period"].max() > 2
        match_end_minute = 120 if has_extra_time else 90

        # Starting XI detection
        starters = set()
        starting_xi_events = match_events[match_events["type"] == "Starting XI"]
        for _, row in starting_xi_events.iterrows():
            if isinstance(row.get("tactics"), dict) and "lineup" in row["tactics"]:
                for p in row["tactics"]["lineup"]:
                    p_name = p.get("player", {}).get("name")
                    if p_name:
                        starters.add(NAME_MAPPING.get(p_name, p_name))

        if not starters:
            starters = set(match_events[match_events["minute"] <= 1]["player"].dropna().unique())

        sub_outs = {}
        sub_ins = {}

        # Substitution detection
        for _, s_row in match_events[match_events["type"] == "Substitution"].iterrows():
            out_p = NAME_MAPPING.get(s_row["player"], s_row["player"])
            sub_minute = s_row["minute"]
            sub_outs[out_p] = sub_minute

            in_p_obj = s_row.get("substitution_replacement")
            in_p_name = in_p_obj.get("name") if isinstance(in_p_obj, dict) else str(in_p_obj) if pd.notna(in_p_obj) else None
            if in_p_name:
                sub_ins[NAME_MAPPING.get(in_p_name, in_p_name)] = sub_minute

        all_players = set(match_events["player"].dropna().unique()).union(starters)

        for p in all_players:
            if p in starters:
                start_m = 0
                end_m = sub_outs.get(p, match_end_minute)
            elif p in sub_ins:
                start_m = sub_ins[p]
                end_m = sub_outs.get(p, match_end_minute)
            else:
                p_events = match_events[match_events["player"] == p]
                start_m = p_events["minute"].min()
                end_m = p_events["minute"].max()

            played = max(end_m - start_m, 1)
            player_minutes_dict[p] = player_minutes_dict.get(p, 0) + played

    return pd.DataFrame(list(player_minutes_dict.items()), columns=["player", "minutes_played"])