"""
model.py
Expected Threat (xT) model module.
Supports both custom Tournament fitting (16x12) and League Benchmark (12x8).
"""

import numpy as np
import pandas as pd
from data_processing import get_bin

# Industry benchmark grid (McKay Johns / Karun Singh 12x8)
MCKAY_XT_GRID = np.array([
    [0.006, 0.008, 0.008, 0.010, 0.011, 0.012, 0.015, 0.017, 0.021, 0.028, 0.035, 0.038],
    [0.008, 0.009, 0.009, 0.011, 0.012, 0.014, 0.016, 0.019, 0.024, 0.030, 0.041, 0.046],
    [0.009, 0.010, 0.010, 0.011, 0.013, 0.014, 0.017, 0.019, 0.024, 0.029, 0.055, 0.064],
    [0.009, 0.011, 0.010, 0.011, 0.013, 0.015, 0.017, 0.020, 0.024, 0.035, 0.108, 0.257],
    [0.009, 0.011, 0.010, 0.011, 0.013, 0.015, 0.017, 0.020, 0.024, 0.035, 0.108, 0.257],
    [0.009, 0.010, 0.010, 0.011, 0.013, 0.014, 0.017, 0.019, 0.024, 0.029, 0.055, 0.064],
    [0.008, 0.009, 0.009, 0.011, 0.012, 0.014, 0.016, 0.019, 0.024, 0.030, 0.041, 0.046],
    [0.006, 0.008, 0.008, 0.010, 0.011, 0.012, 0.015, 0.017, 0.021, 0.028, 0.035, 0.038]
])


class ExpectedThreatModel:
    def __init__(self, mode: str = "tournament"):
        """
        mode:
          - 'tournament': Custom fitted model on competition data (16x12).
          - 'benchmark': Pre-calibrated industry standard grid (12x8).
        """
        self.mode = mode
        if mode == "tournament":
            self.bins_x = 16
            self.bins_y = 12
            self.xt_matrix = None
        elif mode == "benchmark":
            self.bins_x = 12
            self.bins_y = 8
            self.xt_matrix = MCKAY_XT_GRID
        else:
            raise ValueError("mode must be 'tournament' or 'benchmark'")

    def fit(self, shots: pd.DataFrame, passes: pd.DataFrame, iterations: int = 5):
        """Fits Markov chain transition probabilities for the tournament model."""
        if self.mode == "benchmark":
            print("-> Using pre-calibrated Benchmark grid (12x8).")
            return self.xt_matrix

        print(f"-> Fitting Tournament Markov Chain model ({self.bins_x}x{self.bins_y})...")
        shot_mat = np.zeros((self.bins_x, self.bins_y))
        goal_mat = np.zeros((self.bins_x, self.bins_y))
        move_mat = np.zeros((self.bins_x, self.bins_y))
        transition_mat = np.zeros((self.bins_x, self.bins_y, self.bins_x, self.bins_y))

        # Calculate shot and goal distributions
        for _, row in shots.iterrows():
            bx, by = get_bin(row["x"], row["y"], self.bins_x, self.bins_y)
            shot_mat[bx, by] += 1
            if row["is_goal"] == 1:
                goal_mat[bx, by] += 1

        # Calculate pass transition matrix
        for _, row in passes.iterrows():
            sbx, sby = get_bin(row["x"], row["y"], self.bins_x, self.bins_y)
            ebx, eby = get_bin(row["end_x"], row["end_y"], self.bins_x, self.bins_y)
            move_mat[sbx, sby] += 1
            transition_mat[sbx, sby, ebx, eby] += 1

        total_actions = shot_mat + move_mat
        p_shot = np.divide(shot_mat, total_actions, out=np.zeros_like(shot_mat), where=total_actions != 0)
        p_goal = np.divide(goal_mat, shot_mat, out=np.zeros_like(goal_mat), where=shot_mat != 0)
        p_move = np.divide(move_mat, total_actions, out=np.zeros_like(move_mat), where=total_actions != 0)

        direct_threat = p_shot * p_goal

        # Normalize transitions
        for x in range(self.bins_x):
            for y in range(self.bins_y):
                tot = np.sum(transition_mat[x, y, :, :])
                if tot > 0:
                    transition_mat[x, y, :, :] /= tot

        # Markov Chain iteration
        xt_mat = np.copy(direct_threat)
        for _ in range(iterations):
            new_xt = np.copy(direct_threat)
            for x in range(self.bins_x):
                for y in range(self.bins_y):
                    new_xt[x, y] += p_move[x, y] * np.sum(transition_mat[x, y, :, :] * xt_mat)
            xt_mat = new_xt

        self.xt_matrix = xt_mat
        print("-> Tournament model fitting complete.")
        return self.xt_matrix

    def rate_action(self, start_x: float, start_y: float, end_x: float, end_y: float, is_carry: bool = False):
        """Calculates Delta xT = xT(end) - xT(start)."""
        if np.isnan(start_x) or np.isnan(end_x) or self.xt_matrix is None:
            return 0.0

        bx1, by1 = get_bin(start_x, start_y, self.bins_x, self.bins_y)
        bx2, by2 = get_bin(end_x, end_y, self.bins_x, self.bins_y)

        if self.mode == "benchmark":
            val_start = self.xt_matrix[by1, bx1]
            val_end = self.xt_matrix[by2, bx2]
        else:
            val_start = self.xt_matrix[bx1, by1]
            val_end = self.xt_matrix[bx2, by2]

        delta = float(val_end - val_start)
        # Carries only credit forward progression
        if is_carry:
            return max(0.0, delta)
        return delta

    def rate_dataframe(self, df_actions: pd.DataFrame, is_carry: bool = False):
        """Rates an entire DataFrame of passes or carries."""
        df_actions = df_actions.copy()
        df_actions["xt_added"] = [
            self.rate_action(sx, sy, ex, ey, is_carry=is_carry)
            for sx, sy, ex, ey in zip(
                df_actions["x"], df_actions["y"], df_actions["end_x"], df_actions["end_y"]
            )
        ]
        return df_actions