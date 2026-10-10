"""
main.py
Primary analysis pipeline featuring your custom Tournament Model (16x12)
with side-by-side comparison to the Industry Benchmark (12x8).
"""

from data_processing import (
    calculate_player_minutes,
    load_and_preprocess_data,
    prepare_events,
)
from model import ExpectedThreatModel
from visualization import plot_model_comparison, plot_stacked_xt_p90


def run_pipeline(model_mode: str, df, passes, carries, player_minutes, min_minutes: int = 150):
    """Executes full evaluation pipeline for a specified model mode."""
    model = ExpectedThreatModel(mode=model_mode)

    if model_mode == "tournament":
        shots = df[df["type"] == "Shot"].dropna(subset=["x", "y"]).copy()
        shots["is_goal"] = (shots["shot_outcome"] == "Goal").astype(int)
        model.fit(shots, passes)
    else:
        model.fit(None, None)

    # Rate actions (Passes = Net xT, Carries = Progressive xT)
    passes_rated = model.rate_dataframe(passes, is_carry=False)
    carries_rated = model.rate_dataframe(carries, is_carry=True)

    # Aggregate xT by player
    pass_xt = (
        passes_rated.groupby("player")["xt_added"]
        .sum()
        .reset_index()
        .rename(columns={"xt_added": "pass_xt"})
    )
    carry_xt = (
        carries_rated.groupby("player")["xt_added"]
        .sum()
        .reset_index()
        .rename(columns={"xt_added": "carry_xt"})
    )

    merged = (
        player_minutes.merge(pass_xt, on="player", how="left")
        .merge(carry_xt, on="player", how="left")
        .fillna(0)
    )

    # Filter players with minimum 150 minutes played
    filtered = merged[merged["minutes_played"] >= min_minutes].copy()
    filtered["pass_xt_p90"] = (filtered["pass_xt"] / filtered["minutes_played"]) * 90
    filtered["carry_xt_p90"] = (filtered["carry_xt"] / filtered["minutes_played"]) * 90
    filtered["total_xt_p90"] = filtered["pass_xt_p90"] + filtered["carry_xt_p90"]

    return filtered


def main():
    MIN_MINUTES = 150

    # 1. Load dataset and compute accurate playing time
    df = load_and_preprocess_data("euro_2024_events.parquet")
    shots, passes, carries = prepare_events(df, min_carry_distance=3.0)
    player_minutes = calculate_player_minutes(df)

    # 2. Primary Execution: Custom Tournament Model (16x12)
    print("\n=======================================================")
    print("PRIMARY MODEL: CUSTOM TOURNAMENT MODEL (EURO 2024 16x12)")
    print("=======================================================")
    df_tournament = run_pipeline("tournament", df, passes, carries, player_minutes, min_minutes=MIN_MINUTES)
    print(
        df_tournament.sort_values(by="total_xt_p90", ascending=False)[
            ["player", "minutes_played", "pass_xt_p90", "carry_xt_p90", "total_xt_p90"]
        ]
        .head(10)
        .to_string(index=False)
    )

    # 3. Secondary Execution: Industry Benchmark Model (12x8)
    print("\n=======================================================")
    print("COMPARISON MODEL: INDUSTRY BENCHMARK (MCKAY JOHNS 12x8)")
    print("=======================================================")
    df_benchmark = run_pipeline("benchmark", df, passes, carries, player_minutes, min_minutes=MIN_MINUTES)
    print(
        df_benchmark.sort_values(by="total_xt_p90", ascending=False)[
            ["player", "minutes_played", "pass_xt_p90", "carry_xt_p90", "total_xt_p90"]
        ]
        .head(10)
        .to_string(index=False)
    )

    # 4. Visualizations:
    # A) Standalone plot for your custom model
    plot_stacked_xt_p90(
        df_tournament,
        title="Custom Tournament xT per 90 (EURO 2024 - 16x12 Grid)",
        top_n=20
    )

    # B) Side-by-side comparison of both methodologies
    plot_model_comparison(
        df_tournament,
        df_benchmark,
        primary_name="A: Custom Tournament Model (EURO 2024 16x12)",
        secondary_name="B: Industry Benchmark Model (McKay Johns 12x8)",
        top_n=15
    )


if __name__ == "__main__":
    main()