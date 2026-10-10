import matplotlib.pyplot as plt
from mplsoccer import Pitch
import numpy as np
import pandas as pd
from statsbombpy import sb

# ==========================================
# 1. NAČÍTANIE VOĽNÝCH DÁT ZO STATSBOMB
# ==========================================
print("Sťahujem dáta zo StatsBomb...")
# Vyberieme napríklad UEFA Euro 2024 (alebo starší ročník dostupný v Open Data, napr. competition_id=55)
# Pre ukážku načítame zápasy špecifického turnaja
competitions = sb.competitions()

# Načítame eventy pre konkrétny turnaj (napr. FIFA World Cup 2022: competition_id=43, season_id=106)
events = sb.competition_events(
    country="International",
    division="FIFA World Cup",
    season="2022",
    gender="male",
)

print(f"Načítaných {len(events)} udalostí.")

# ==========================================
# 2. FILTRÁCIA A PRÍPRAVA DÁT
# ==========================================
# StatsBomb ihrisko má rozmery: x patrí do [0, 120], y patrí do [0, 80]
PITCH_LENGTH = 120.0
PITCH_WIDTH = 80.0
BINS_X = 16
BINS_Y = 12

# Extrahujeme súradnice pre začiatok a koniec akcie
# StatsBomb ukladá súradnice ako zoznam [x, y] v stĺpci 'location' a 'pass_end_location'
events_clean = events[events["location"].notna()].copy()
events_clean["x"] = events_clean["location"].apply(lambda loc: loc[0])
events_clean["y"] = events_clean["location"].apply(lambda loc: loc[1])

# Filtrujeme úspešné prihrávky
passes = events_clean[
    (events_clean["type"] == "Pass") & (events_clean["pass_outcome"].isna())
].copy()
passes["end_x"] = passes["pass_end_location"].apply(lambda loc: loc[0])
passes["end_y"] = passes["pass_end_location"].apply(lambda loc: loc[1])

# Filtrujeme strely
shots = events_clean[events_clean["type"] == "Shot"].copy()
shots["is_goal"] = (shots["shot_outcome"] == "Goal").astype(int)

# ==========================================
# 3. ZOSTAVENIE PRIESTOROVEJ MRIEŽKY
# ==========================================
# Výpočet priameho ohrozenia zo strely (Pravdepodobnosť gólu z každej zóny)
shot_counts, x_edges, y_edges = np.histogram2d(
    shots["x"],
    shots["y"],
    bins=[BINS_X, BINS_Y],
    range=[[0, PITCH_LENGTH], [0, PITCH_WIDTH]],
)
goal_counts, _, _ = np.histogram2d(
    shots[shots["is_goal"] == 1]["x"],
    shots[shots["is_goal"] == 1]["y"],
    bins=[BINS_X, BINS_Y],
    range=[[0, PITCH_LENGTH], [0, PITCH_WIDTH]],
)

# Pravdepodobnosť skórovania pri strele zo zóny (ošetrenie delenia nulou)
shot_success_prob = np.divide(
    goal_counts,
    shot_counts,
    out=np.zeros_like(goal_counts),
    where=shot_counts != 0,
)

# Celkový počet akcií (prihrávky + strely) na výpočet pravdepodobnosti strely
pass_counts, _, _ = np.histogram2d(
    passes["x"],
    passes["y"],
    bins=[BINS_X, BINS_Y],
    range=[[0, PITCH_LENGTH], [0, PITCH_WIDTH]],
)
total_actions = pass_counts + shot_counts

shot_prob = np.divide(
    shot_counts,
    total_actions,
    out=np.zeros_like(shot_counts),
    where=total_actions != 0,
)

# Základná matica okamžitého ohrozenia: pravdepodobnosť strely * pravdepodobnosť gólu
direct_threat_matrix = shot_prob * shot_success_prob

# ==========================================
# 4. VIZUALIZÁCIA MRIEŽKY OHROZENIA (HEATMAP)
# ==========================================
pitch = Pitch(
    pitch_type="statsbomb",
    pitch_color="#181717",
    line_color="#c7d5cc",
    axis=True,
    label=True,
)
fig, ax = pitch.draw(figsize=(10, 7))

# Vykreslenie hodnôt z mriežky
heatmap = ax.imshow(
    direct_threat_matrix.T,
    extent=[0, PITCH_LENGTH, 0, PITCH_WIDTH],
    origin="lower",
    cmap="hot",
    alpha=0.7,
    interpolation="nearest",
)

cbar = fig.colorbar(heatmap, ax=ax, fraction=0.03, pad=0.04)
cbar.set_label("Okamžité ohrozenie (Direct Threat)", color="white")
cbar.ax.yaxis.set_tick_params(color="white")
plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")

ax.set_title(
    "Základná mriežka ohrozenia ihriska (StatsBomb)",
    fontsize=14,
    color="white",
    pad=15,
)
plt.show()