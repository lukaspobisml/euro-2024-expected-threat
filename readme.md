# Expected Threat (xT) Spatial Modelling & Pass Valuation — EURO 2024

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)
![StatsBomb](https://img.shields.io/badge/Data-StatsBomb_Open_Data-red?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

An end-to-end spatial data science project evaluating player progression and threat creation during **UEFA EURO 2024** using raw StatsBomb event data. 

The project features:
1. **A Custom Tournament-Specific Model ($16 \times 12$ Grid):** Trained directly on EURO 2024 micro-events using Markov chains to capture short-tournament transition dynamics.
2. **An Industry Benchmark Model ($12 \times 8$ Grid):** Implemented using the calibrated Karun Singh / McKay Johns framework for comparative baseline analysis.
3. **Action Decomposition (Passes vs. Carries):** Decoupling spatial threat generated via distribution versus progressive ball carries per 90 minutes.

---

## 📊 Visualizations

### 1. Custom Tournament Model (EURO 2024 — $16 \times 12$ Grid)
Top creators per 90 minutes (minimum 150 minutes played), split into passing threat and progressive carries:

![Custom Tournament Model](visuals/tournament_xt_top20.png)

### 2. Methodological Comparison: Custom ($16 \times 12$) vs. Benchmark ($12 \times 8$)
Side-by-side comparison illustrating how tournament-fitted transitions prioritize dynamic wide carries compared to league-calibrated distribution baselines:

![Model Comparison](visuals/model_comparison.png)

---

## 🧠 What is Expected Threat (xT)?

Traditional metrics (goals, assists, key passes) heavily bias credit toward the final touch. **Expected Threat (xT)** is a possession-value model that measures how much a player increases their team's probability of scoring by moving the ball from location $A$ to location $B$.

### Mathematical Formulation
The pitch is discretized into a grid of zones. In any given zone $(x, y)$, a player can either **shoot** or **move the ball** (via pass or carry):

$$xT(x, y) = \underbrace{P(\text{Shot}) \times P(\text{Goal} \mid \text{Shot})}_{\text{Direct Threat}} + \underbrace{P(\text{Move}) \times \sum_{(x', y')} \Big( T_{(x, y) \to (x', y')} \times xT(x', y') \Big)}_{\text{Future Threat}}$$

Where:
* $P(\text{Shot})$ is the probability of taking a shot from zone $(x, y)$.
* $P(\text{Goal} \mid \text{Shot})$ is the probability of a shot resulting in a goal from zone $(x, y)$.
* $P(\text{Move})$ is the probability of choosing to move the ball rather than shoot.
* $T_{(x, y) \to (x', y')}$ is the transition probability of moving the ball from $(x, y)$ to $(x', y')$.

Action valuation is computed as the net threat added:
$$\Delta xT = xT_{\text{end}} - xT_{\text{start}}$$

---

## 🔬 Methodology & Model Comparison

| Feature | Model A: Custom Tournament Model | Model B: Industry Benchmark |
| :--- | :--- | :--- |
| **Grid Resolution** | $16 \times 12$ (7.5m $\times$ 6.6m cells) | $12 \times 8$ (10m $\times$ 10m cells) |
| **Training Sample** | 51 matches (EURO 2024 raw events) | Multi-season league sample (Karun Singh baseline) |
| **Pass Valuation** | Net $\Delta xT$ (penalizes backward retention) | Net $\Delta xT$ |
| **Carry Valuation** | Progressive forward carries ($\ge 3.0$ m, $\Delta xT \ge 0$) | Progressive forward carries ($\ge 3.0$ m, $\Delta xT \ge 0$) |
| **Playing Time** | Precise minutes calculated via lineups & substitutions | Standardized match minutes |

### Why a 16x12 Grid? (Spatial Resolution vs. Data Sparsity)

Choosing the optimal spatial grid requires balancing tactile granularity with sample size constraints:

1. **Doubled Spatial Granularity ($50\text{ m}^2$ vs. $100\text{ m}^2$):**
   * The benchmark $12 \times 8$ grid divides the pitch into $10\text{m} \times 10\text{m}$ cells ($100\text{ m}^2$). Because $\Delta xT$ is awarded only when an action crosses a cell boundary, short combination passes and 5-to-7-meter progressive carries inside a single 10-meter square receive zero value.
   * Our $16 \times 12$ grid narrows cells to $7.5\text{m} \times 6.67\text{m}$ ($\approx 50\text{ m}^2$), doubling spatial sensitivity and crediting tight dribbling actions in the final third.

2. **Penalty Box & Half-Space Alignment:**
   * In StatsBomb coordinates ($120 \times 80$), the penalty box extends to $X = 102$ and spans $Y = 18$ to $Y = 62$. 
   * A 16-column layout places zone transitions at $X = 97.5$ and $X = 105$, capturing the decisive "edge of the box" zone for cutbacks and second-phase shooting lanes far more organically than coarse 10-meter increments.

3. **Why Not an Even Finer Grid (e.g., $24 \times 16$)?**
   * While a $24 \times 16$ grid ($5\text{m} \times 5\text{m}$ cells) offers high theoretical resolution, it suffers heavily from **data sparsity** on a 51-match tournament sample:
     * A $16 \times 12$ grid contains **192 cells**, creating $192^2 = \mathbf{36,864}$ possible transition states.
     * A $24 \times 16$ grid contains **384 cells**, exploding to $384^2 = \mathbf{147,456}$ possible transitions.
   * With $\approx 42,000$ successful passes and only **117 goals** scored across EURO 2024, a $24 \times 16$ transition matrix would be predominantly empty ($>75\%$ zero transitions). Moreover, individual goals scored from distance would introduce severe statistical noise and spatial overfitting. The $16 \times 12$ configuration represents the optimal data-to-parameter ratio.

### Analytical Rationale
* **The Benchmark Model ($12 \times 8$)** favors deep-lying orchestrators (e.g., Dušan Tadić, Christian Eriksen, Toni Kroos) whose value stems from sustained diagonal switches and final-third entries in structured possession.
* **The Custom Model ($16 \times 12$)** captures the tournament reality of EURO 2024, where compact defensive low-blocks necessitated 1v1 wing isolation and dynamic carries into the box (e.g., Jérémy Doku, Rafael Leão, Ousmane Dembélé).

---

## 📁 Repository Structure

```text
├── euro_2024_events.parquet    # Raw StatsBomb event dataset
├── data_processing.py          # Data ingestion, coordinate transforms & minute calculation
├── model.py                    # ExpectedThreatModel implementation (Markov chain & benchmark)
├── visualization.py            # Matplotlib styling, stacked charts & comparison plots
├── main.py                     # Execution pipeline running both methodologies
├── visuals/                    # Generated charts and exported graphics
│   ├── tournament_xt_top20.png
│   └── model_comparison.png
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation