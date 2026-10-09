# Learn2Slither Walkthrough & Technical Guide

<p align="right">
  <a href="#learn2slither-專案導讀與技術答辯指南">
    <img src="https://img.shields.io/badge/中文-forestgreen?style=for-the-badge" />
  </a>
</p>

This document provides a comprehensive walkthrough of the **Learn2Slither** Reinforcement Learning project: 42 Subject specifications, algorithmic foundations (Q-learning & Ray-Cast Vision), system architecture, code reading order, and peer evaluation defense preparation.

---

## 0. Language Choice & Project Architecture

This project is implemented in **Python 3 + Pygame**.

| Criteria | Python 3 + Pygame | C++ / Raylib | Verdict |
|---|---|---|---|
| **RL Prototyping** | ⭐⭐⭐ Highly readable, clean matrix/tuple data structures | ⭐⭐ Boilerplate-heavy | Python accelerates algorithm tuning |
| **Training Speed** | ⭐⭐⭐ Tens of thousands of steps/sec (in headless mode) | ⭐⭐⭐ Native C-speed | Completely decoupled UI ensures instant convergence |
| **UI Integration** | ⭐⭐⭐ Native Pygame event & grid rendering | ⭐⭐ Low-level windowing | Pygame delivers responsive step-by-step controls |
| **Norm Compliance** | ⭐⭐⭐ Verified with `flake8` (`make norm`) | ⭐⭐⭐ 42 Norminette | Clean modular Python code conforming to PEP 8 |

> **💡 Defense Answer**: "The 42 Subject allows any language and library. Because the training loop (`-visual off`) is fully decoupled from the visual interface, our tabular Q-learning engine processes tens of thousands of frames per second in headless mode, completing 100 sessions in under 2 seconds. In visual mode (`-visual on`), Pygame provides smooth rendering and step-by-step keyboard control."

### Project Structure

```
Learn2Slither/
├── Makefile                      # Targets: all, clean, fclean, re, norm, test, help
├── requirements.txt              # Dependencies: pygame>=2.5.0, flake8>=6.0.0
├── README.md                     # Project overview, installation, CLI manual
├── models/                       # Evaluation models: 1sess.txt, 10sess.txt, 100sess.txt, 1000sess.txt
├── docs/
│   ├── en.subject.pdf            # 42 School official subject specification
│   ├── TODO.md                   # Implementation roadmap and compliance checklist
│   └── walkthrough.md            # Technical walkthrough & defense guide
└── src/
    ├── main.py                   # CLI parser, headless training loop, and visual UI loop
    ├── config/
    │   ├── game.py               # Board size (10x10), tokens ('W','H','S','G','R','0'), DIRECTIONS
    │   ├── ai.py                 # Hyperparameters (α, γ, ε, decay) and rewards
    │   └── ui.py                 # Window dimensions, colors, KEY_TO_DIR keycodes
    ├── core/
    │   ├── snake.py              # Snake body deque, contiguous spawn, move, collisions
    │   └── environment.py        # 10x10 grid, apple spawns, 4-direction ray casting, cross vision
    ├── ai/
    │   └── q_table.py            # QTableAgent: ε-greedy, Bellman updates, save/load models
    └── ui/
        └── gui.py                # Pygame UI: grid rendering, HUD, step-by-step loop
```

---

## 1. 42 Subject Core Specifications & The 3 Zero-Tolerance Rules

### Core Rules
1. **Grid**: Fixed $10 \times 10$ board (dimension customization supported via `-board-size`).
2. **Apples**: Exactly **2 Green Apples** (+1 length) and **1 Red Apple** (-1 length) randomly placed on empty spaces.
3. **Snake**: Starts with **3 contiguous segments** placed randomly on empty cells.
4. **Game Over Conditions**:
   - Snake head hits a wall (`W`).
   - Snake head collides with its own body (`S`).
   - Snake length drops to 0 (after consuming red apples).
5. **Goal**: Reach length $\ge 10$ cells and maximize survival duration.

### 🚨 The 3 Zero-Tolerance Subject Penalties

1. **Algorithm Restriction (Score = 0 if violated)**:
   > *"No other model is allowed; otherwise, a score of 0 will be given."*
   - Only **Q-learning** (Q-table or Neural Network) is permitted. No Minimax, Alpha-Beta, or A* search algorithms are used for decision making.
2. **Vision Restriction (Penalty = -42 if violated)**:
   > *"You can only provide to the agent the information visible to your snake. Providing more information will result in a penalty of -42."*
   - The agent only receives observation data along 4 orthogonal ray casts (`UP`, `LEFT`, `DOWN`, `RIGHT`) originating from the snake head.
   - **Zero global coordinates $(x, y)$ of the snake head or apples are ever fed to the agent.**
   - In terminal mode, the cross-shaped line-of-sight vision must be printed before each action.
3. **Mandatory Models & `-dontlearn` Verification**:
   - The repository must include pre-trained models: `models/1sess.txt`, `models/10sess.txt`, `models/100sess.txt`, and `models/1000sess.txt`.
   - The `-dontlearn` flag freezes the Q-table ($\epsilon = 0$, updates disabled) so evaluators can inspect learned policy performance.

---

## 2. Mathematical & Algorithmic Foundations: Q-Learning

Tabular Q-learning solves the Markov Decision Process (MDP) by learning an action-value function $Q(s, a)$, representing the expected cumulative discounted reward of taking action $a$ in state $s$:

### The Bellman Optimality Equation
After taking action $a$ in state $s$, observing reward $r$, and transitioning to state $s'$:

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$

- **$\alpha$ (Learning Rate)**: Configured to `0.1`. Balances incorporation of new experiences with stability.
- **$\gamma$ (Discount Factor)**: Configured to `0.95`. Values long-term survival and future apple consumption.
- **Terminal State**: If the episode ends (`done == True`), future value $\max_{a'} Q(s', a') = 0$, updating:
  $$Q(s, a) \leftarrow Q(s, a) + \alpha [r - Q(s, a)]$$

### Exploration vs. Exploitation ($\epsilon$-greedy Policy)
At each step:
$$\text{action} = \begin{cases} \text{random action from } \{\text{UP}, \text{LEFT}, \text{DOWN}, \text{RIGHT}\} & \text{with probability } \epsilon \\ \arg\max_{a} Q(s, a) & \text{with probability } 1 - \epsilon \end{cases}$$

- **Decay Schedule**: Starts at $\epsilon = 1.0$, decayed per step by $\epsilon \leftarrow \max(\epsilon_{\text{min}}, \epsilon \times 0.9999)$ down to $\epsilon_{\text{min}} = 0.05$.
- **Evaluation Mode (`-dontlearn`)**: $\epsilon$ is forced to `0.0`, ensuring pure exploitation of learned values with zero updates.

---

## 3. State Space Representation & 4-Direction Ray Vision

### Why Raw Coordinates are Banned
- Feeding absolute coordinates $(x_{\text{head}}, y_{\text{head}}, x_{\text{apple}}, \dots)$ directly violates the Subject's strict line-of-sight mandate (-42 penalty).
- Furthermore, a $10 \times 10$ board with arbitrary body positions yields $> 10^{100}$ states, making tabular Q-learning completely intractable.

### Ray-Casting Extraction (`Environment._cast_ray`)
From the head position $(x, y)$, a ray marches in each direction $d \in \{\text{UP}, \text{LEFT}, \text{DOWN}, \text{RIGHT}\}$ until reaching the boundary:
- Ray cells are converted to tokens: `W` (Wall), `H` (Head), `S` (Body), `G` (Green apple), `R` (Red apple), `0` (Empty).
- In visual/step-by-step mode, this generates the required terminal cross pattern:
  ```
          [W]
          [0]
          [G]
  [W][0]  [H]  [0][R][W]
          [S]
          [W]
  ```

### The 20-Element Compact State Tuple (`Environment.get_state`)
To ensure rapid tabular convergence while preserving all critical tactical information, the raw ray cast is encoded into a 20-element discrete tuple (5 features per direction):

For each direction $d \in \{\text{UP}, \text{LEFT}, \text{DOWN}, \text{RIGHT}\}$:
1. `immediate_danger` (int `0` or `1`): Is the adjacent cell 1 step away a wall (`W`) or body segment (`S`)?
2. `first_obstacle` (str): First non-empty token seen along the ray (`'0'`, `'G'`, `'R'`, `'S'`, `'W'`).
3. `distance_bin` (int `0`, `1`, or `2`): Distance to that first obstacle:
   - `0`: Far ($> 4$ cells)
   - `1`: Medium ($2-4$ cells)
   - `2`: Close ($1$ cell)
4. `has_green` (int `0` or `1`): Does any cell along this ray contain a Green Apple?
5. `has_red` (int `0` or `1`): Does any cell along this ray contain a Red Apple?

$$\text{State Space Size} \approx (2 \times 5 \times 3 \times 2 \times 2)^4 \text{ theoretical maximum}$$
In practice, reachable states number under 10,000, allowing tabular Q-learning to fully converge within seconds.

---

## 4. Reward Function Design

| Event | Reward Value | Purpose |
|---|---|---|
| **Eat Green Apple** | `+10.0` | Primary incentive: Grow length and progress toward goal $\ge 10$. |
| **Eat Red Apple** | `-10.0` | Strong deterrent: Avoid losing length and risk of dying at length 0. |
| **Step Survival** | `-0.1` | Small step penalty: Encourages efficient navigation, prevents infinite harmless loops. |
| **Death (Wall / Self / 0 Length)** | `-50.0` | Catastrophic penalty: Teaches snake to prioritize obstacle avoidance above all else. |

---

## 5. Architectural Breakdown: The 3 Subject Blocks

```
       ┌───────────────────────────┐
       │        ENVIRONMENT        │
       │    (Grid, Snake, Step)    │
       └─────┬───────────────▲─────┘
             │               │
       (Observation)     (Action)
             │               │
             ▼               │
       ┌───────────┐         │
       │INTERPRETER│         │
       │(State 20D)│         │
       └─────┬─────┘         │
             │               │
      (State, Reward)        │
             │               │
             ▼               │
       ┌───────────┐         │
       │   AGENT   ├─────────┘
       │ (QTable)  │
       └───────────┘
```

### 1. `core/snake.py` (Snake Entity)
- Maintains `body` using `collections.deque` of `(x, y)` tuples.
- `_init_contiguous_body()`: Randomly spawns 3 strictly contiguous segments on unoccupied cells.
- `move(action)`: Computes new head position, appends to left of deque, pops tail (unless growing).
- `check_collision()`: Detects boundary collisions and self-intersections.

### 2. `core/environment.py` (Environment & Interpreter)
- Manages grid dimensions, green apple set (2 items), and red apple set (1 item).
- `step(action)`: Executes snake movement, checks collisions, updates apples upon consumption, computes reward, and returns `(next_state, reward, done, info)`.
- `render_terminal_cross()`: Formats the 4-direction line-of-sight into the required ASCII cross display.

### 3. `ai/q_table.py` (Q-Learning Agent)
- `q_table`: `defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])` mapping state tuple to action values for `["UP", "LEFT", "DOWN", "RIGHT"]`.
- `choose_action(state)`: Selects action via $\epsilon$-greedy policy.
- `update(state, action, reward, next_state, done)`: Executes the Bellman equation update.
- `save(path)` / `load(path)`: Serializes Q-table to plain text, printing subject-compliant confirmation messages.

### 4. `ui/gui.py` (Pygame UI)
- `UI` class: Initializes Pygame window, renders grid background, apples, and snake segments with contrasting head color.
- HUD overlay: Displays current Length, Max Length, Survival Duration, Session counter, and FPS.
- Step-by-step loop: Listens for `KEY_SPACE` to advance AI by one step, and allows manual player steering.

---

## 6. Peer Evaluation Defense Preparation (Q&A)

### Q1: "How do you prove the agent doesn't receive full board coordinates?"
**Answer**:
"You can inspect `Environment.get_state()` in `src/core/environment.py`. The agent receives only a 20-element tuple composed of ray-cast observations (`immediate_danger`, `first_obstacle`, `distance_bin`, `has_green`, `has_red`) in the 4 orthogonal directions. No coordinates $(x, y)$ of either the snake or the apples are ever passed to the agent. This guarantees zero coordinate leakage and 100% compliance with Subject requirements."

### Q2: "Why use tabular Q-learning instead of a Deep Q-Network (DQN)?"
**Answer**:
"The Subject specifically requires Q-learning and tests with models trained over 1, 10, 100, and 1000 sessions. Tabular Q-learning converges instantly without GPU dependencies, avoids overfitting on small replay buffers, and can be serialized to human-readable plain text (`models/*.txt`). Our 20-element discrete ray state space is small enough to train 100 sessions in under 2 seconds while maintaining strong decision capability."

### Q3: "What differences should we observe between the 4 provided models?"
**Answer**:
1. `models/1sess.txt`: Almost untrained. Snake mostly explores randomly ($\epsilon \approx 1.0$) and frequently dies within a few steps.
2. `models/10sess.txt`: Beginning of learning. Has encountered walls and learned to avoid immediate head-on collisions, surviving slightly longer.
3. `models/100sess.txt`: Competent policy. Consistently avoids walls and self-collision, seeks green apples along its line of sight, and actively avoids red apples.
4. `models/1000sess.txt`: Highly optimized policy. Reaches length $\ge 10$ consistently and survives for hundreds of steps.

### Q4: "How do you test the step-by-step mode with terminal output?"
**Answer**:
```bash
./snake -visual on -load models/100sess.txt -sessions 10 -dontlearn -step-by-step
```
"In this mode, the UI window will pause before each step. Pressing Space advances the snake by one frame. The terminal concurrently outputs the ASCII cross vision and the selected action, demonstrating pure exploitation (`-dontlearn`) of pre-trained weights."

---

<br/>

---

<br/>

<p align="right">
  <a href="#learn2slither-walkthrough--technical-guide">
    <img src="https://img.shields.io/badge/-TOP-forestgreen?style=for-the-badge" />
  </a>
</p>

# Learn2Slither 專案導讀與技術答辯指南

本文件為 **Learn2Slither** 強化學習貪食蛇專案之完整架構導讀與口試答辯手冊：包含 42 官方規範手冊要求、Q-learning 演算法數學原理、四向射線特徵工程、系統模組架構以及同儕評分（Peer Evaluation）之應答範例。

---

## 0. 語言選擇與架構評估

本專案採用 **Python 3 + Pygame** 實現。

| 評估維度 | Python 3 + Pygame | C++ / SFML | 綜合評估 |
|---|---|---|---|
| **RL 原型開發** | ⭐⭐⭐ 高可讀性、資料結構（Tuple/Dict）極度契合 | ⭐⭐ 語法冗長、記憶體管理繁重 | Python 在強化學習調參具壓倒性優勢 |
| **訓練效能** | ⭐⭐⭐ 無介面模式下每秒數萬步 | ⭐⭐⭐ 原生 C++ 速度 | 透過環境與 UI 解耦，Python 數秒即可收斂 |
| **圖形與互動** | ⭐⭐⭐ 輕量且功能完善的事件循環 | ⭐⭐ 跨平台視窗設定較複雜 | Pygame 原生支援單步步進與即時 HUD |
| **代碼規範** | ⭐⭐⭐ 透過 `make norm` 嚴格遵循 flake8 (PEP 8) | ⭐⭐⭐ 42 Norminette | 高度整潔、模組化且易於閱讀 |

> **💡 Defense 對答**：「Subject 允許自由選擇語言與函式庫。由於我們將無介面極速訓練迴圈（`-visual off`）與 Pygame 圖形介面（`-visual on`）徹底解耦，在訓練模式下關閉所有圖形與終端輸出，純 Python 每秒可模擬數萬步，100 回合訓練僅需不到 2 秒即可完成。在展示模式下，Pygame 提供流暢的 10x10 網格渲染與空白鍵單步步進控制。」

### 專案模組目錄

```
Learn2Slither/
├── Makefile                      # 核心規則：all, clean, fclean, re, norm, test, help
├── requirements.txt              # 外部依賴：pygame>=2.5.0, flake8>=6.0.0
├── README.md                     # 專案首頁、安裝說明與 CLI 參數手冊
├── models/                       # 評分指定模型：1sess.txt, 10sess.txt, 100sess.txt, 1000sess.txt
├── docs/
│   ├── en.subject.pdf            # 42 官方規範手冊原件
│   ├── TODO.md                   # 實作任務檢查清單與規範對照
│   └── walkthrough.md            # 深度技術導讀與答辯手冊
└── src/
    ├── main.py                   # CLI 參數解析、無介面訓練迴圈與視覺化 UI 迴圈
    ├── config/
    │   ├── game.py               # 棋盤常數 (10x10)、符號標記 ('W','H','S','G','R','0')、DIRECTIONS
    │   ├── ai.py                 # Q-learning 超參數 (α, γ, ε, decay) 與獎勵權重
    │   └── ui.py                 # 視窗解析度、配色方案與按鍵映射 (KEY_TO_DIR)
    ├── core/
    │   ├── snake.py              # 蛇本體 deque、隨機連續 3 格生成、移動、碰撞檢測
    │   └── environment.py        # 10x10 棋盤、蘋果生成、四向射線特徵提取、十字視野終端輸出
    ├── ai/
    │   └── q_table.py            # QTableAgent: ε-greedy, Bellman 方程更新, 模型存讀檔
    └── ui/
        └── gui.py                # Pygame UI: 網格渲染、HUD 即時資訊、單步步進事件循環
```

---

## 1. 42 Subject 核心規範與三大零容忍禁區

### 核心規則
1. **棋盤維度**：固定 $10 \times 10$ 格（支援 `-board-size` 自訂尺寸加分項）。
2. **蘋果數量**：盤面常態維持 **2 顆綠蘋果**（長度 $+1$）與 **1 顆紅蘋果**（長度 $-1$），隨機出現於空格。
3. **蛇初始狀態**：長度為 **3 格**，隨機且**連續**（contiguously）放置於空格。
4. **遊戲結束條件 (Game Over)**：
   - 蛇頭撞牆 (`W`)。
   - 蛇頭撞自身身軀 (`S`)。
   - 蛇長度因吃紅蘋果萎縮至 0。
5. **最終目標**：回合結束時長度達 **10 格以上**（$\ge 10$），且存活時間越長越好。

### 🚨 三大零容忍扣分禁區

1. **AI 演算法限制（違者 0 分）**：
   > *"No other model is allowed; otherwise, a score of 0 will be given."*
   - 強制只能使用 **Q-learning**（Q-table 或 Neural Network）。嚴禁使用 Minimax、Alpha-Beta、A* 尋路或單純貪婪啟發式規則作為決策主體。
2. **視野限制（違者 -42 分）**：
   > *"You can only provide to the agent the information visible to your snake. Providing more information will result in a penalty of -42."*
   - 智能體僅能獲取蛇頭朝向四個方向（`UP`, `LEFT`, `DOWN`, `RIGHT`）直線射線上的視野資訊。
   - **嚴禁洩漏棋盤全域座標 $(x, y)$ 給 Agent**（杜絕作弊違規）。
   - 終端機在每次請求 Agent 決策前，必須印出十字型視野與採取之 Action。
3. **模型持久化與驗證規範**：
   - 繳交專案必須附帶已訓練的模型檔：`models/1sess.txt`、`models/10sess.txt`、`models/100sess.txt` 與 `models/1000sess.txt`。
   - 必須支援 `-dontlearn` 旗標（凍結 Q-table，$\epsilon = 0$ 且不更新權重），供評分者檢驗學習成果。

---

## 2. 強化學習核心原理：Q-Learning

表格型 Q-learning 旨在求解馬可夫決策過程（MDP）中的最佳動作價值函數 $Q(s, a)$，表示在狀態 $s$ 下執行動作 $a$ 後所能獲得的期望累積折扣獎勵：

### Bellman 最優方程式
智能體在狀態 $s$ 下採取動作 $a$，獲得即時獎勵 $r$ 並轉移至新狀態 $s'$ 後，依據 Bellman 方程式更新 Q 值：

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$

- **$\alpha$（學習率，Learning Rate）**：設定為 `0.1`，平衡新舊經驗的權重。
- **$\gamma$（折扣因子，Discount Factor）**：設定為 `0.95`，重視未來的存活與尋獲蘋果的長期回報。
- **終止狀態（Terminal State）**：當蛇死亡時（`done == True`），未來預期價值為 0，更新公式簡化為：
  $$Q(s, a) \leftarrow Q(s, a) + \alpha [r - Q(s, a)]$$

### 探索與利用平衡（$\epsilon$-greedy Policy）
在每個時間步：
$$\text{Action} = \begin{cases} \text{隨機動作} \in \{\text{UP}, \text{LEFT}, \text{DOWN}, \text{RIGHT}\} & \text{機率為 } \epsilon \\ \arg\max_{a} Q(s, a) & \text{機率為 } 1 - \epsilon \end{cases}$$

- **衰減排程**：初始 $\epsilon = 1.0$，每步依 $\epsilon \leftarrow \max(\epsilon_{\text{min}}, \epsilon \times 0.9999)$ 衰減至 $\epsilon_{\text{min}} = 0.05$。
- **純利用驗證（`-dontlearn`）**：強制設定 $\epsilon = 0.0$ 並關閉更新，完全依據 Q-table 最大值做決定。

---

## 3. 狀態空間設計與四向射線特徵提取（杜絕 -42 分）

### 為什麼不能傳入絕對座標？
- 傳入 $(x_{\text{head}}, y_{\text{head}}, x_{\text{apple}}, \dots)$ 嚴重違反題目「只能提供蛇可見視野」之硬性限制（會直接獲得 -42 分）。
- 此外，$10 \times 10$ 棋盤若包含全盤狀態，狀態數量超過 $10^{100}$，表格型 Q-table 將永遠無法收斂。

### 四向射線掃描 (`Environment._cast_ray`)
從蛇頭座標出發，沿著四個正交方向（上、左、下、右）直線前進直到邊界：
- 沿線格子轉為標準字元：`W` (牆)、`H` (頭)、`S` (身)、`G` (綠蘋果)、`R` (紅蘋果)、`0` (空)。
- 在展示/單步模式下，終端機會渲染出標準十字視野：
  ```
          [W]
          [0]
          [G]
  [W][0]  [H]  [0][R][W]
          [S]
          [W]
  ```

### 20 元素緊湊狀態元組 (`Environment.get_state`)
為了讓 Q-table 在幾秒內快速收斂，我們將原始射線視野抽象提煉為 20 元素的離散特徵元組（每個方向 5 個特徵）：

針對四個方向 $d \in \{\text{UP}, \text{LEFT}, \text{DOWN}, \text{RIGHT}\}$：
1. `immediate_danger`（布林值 `0` 或 `1`）：鄰近 1 格是否為牆壁 (`W`) 或身體 (`S`)。
2. `first_obstacle`（字串）：沿射線看到的第一個非空格物件（`'0'`, `'G'`, `'R'`, `'S'`, `'W'`）。
3. `distance_bin`（分桶整數 `0`, `1`, `2`）：與該首個物件的距離分桶（`0`: 遠 $>4$、`1`: 中 $2-4$、`2`: 近 $1$）。
4. `has_green`（布林值 `0` 或 `1`）：該方向射線上是否存在綠蘋果。
5. `has_red`（布林值 `0` 或 `1`）：該方向射線上是否存在紅蘋果。

這個設計確保了：
- **零座標洩漏**：100% 遵守 42 Subject 視野規範。
- **緊湊離散化**：實際訪問的狀態空間小於 10,000，Q-table 可以在 100~1000 回合內迅速學會避障與尋找蘋果。

---

## 4. 獎勵函數設計 (Reward Shaping)

| 觸發事件 | 獎勵數值 | 設計目的 |
|---|---|---|
| **吃綠蘋果** | `+10.0` | 核心正向目標：引導蛇增長長度並朝 $\ge 10$ 邁進。 |
| **吃紅蘋果** | `-10.0` | 強烈負向懲罰：學會繞過紅蘋果，避免縮短或歸零死亡。 |
| **存活每步** | `-0.1` | 微小時間懲罰：促使蛇以最短路徑覓食，杜絕原地打轉繞圈。 |
| **死亡 (撞牆 / 撞身 / 長度0)** | `-50.0` | 極大懲罰：促使 Agent 將「安全生存」視為第一優先法則。 |

---

## 5. 模組架構職責分工 (Subject Part 6)

```
       ┌───────────────────────────┐
       │        ENVIRONMENT        │
       │    (棋盤、蛇本體、Step)    │
       └─────┬───────────────▲─────┘
             │               │
       (環境觀察事件)       (動作 Action)
             │               │
             ▼               │
       ┌───────────┐         │
       │INTERPRETER│         │
       │(20維特徵) │         │
       └─────┬─────┘         │
             │               │
      (狀態 State, 獎勵)      │
             │               │
             ▼               │
       ┌───────────┐         │
       │   AGENT   ├─────────┘
       │ (Q-table) │
       └───────────┘
```

1. **`core/snake.py`**：以 `collections.deque` 維護身體座標隊列，處理隨機連續 3 格生成、長度增減與自身碰撞檢查。
2. **`core/environment.py`**：管理 10x10 棋盤與蘋果刷新、實現四向射線特徵提取 `get_state()`、終端十字視野輸出 `render_terminal_cross()`、以及 `step(action)` 執行引擎。
3. **`ai/q_table.py`**：以 `defaultdict` 儲存 Q-table，實現 $\epsilon$-greedy 決策、Bellman 方程式更新、以及純文字模型存取。
4. **`ui/gui.py`**：Pygame 視覺化類別 `UI`，負責棋盤繪圖、HUD 資訊儀表板、速度調整與單步步進事件監聽。
5. **`main.py`**：協調極速無介面訓練迴圈 `run_training_loop()` 與圖形互動展示迴圈 `run_visual_loop()`。

---

## 6. 答辯防守與評分應對 (Peer Evaluation Q&A)

### Q1：「如何證明你的蛇沒有作弊讀取全域座標？」
**回答**：
「請直接檢視 `src/core/environment.py` 中的 `get_state()` 函式。傳遞給 Agent 的只有一個 20 元素的特徵元組，包含四個方向的近距危險、首個可見物體、距離分桶以及蘋果可見旗標。完全沒有傳遞蛇頭或蘋果的 $(x, y)$ 座標，嚴格遵守 Subject 的射線視野限制，絕對不會被判定 -42 分。」

### Q2：「為什麼選擇表格型 Q-learning 而非深度強化學習 (DQN)？」
**回答**：
「Subject 明確要求實作 Q-learning 並需提交 1, 10, 100, 1000 回合的模型文字檔。表格型 Q-learning 不需要肥大的 GPU 或深度學習框架，且能保證數值透明與迅速收斂。透過我們精心設計的 20 元素離散特徵，模型在 100 回合內就能穩定學會避障覓食，完全符合題目規格。」

### Q3：「如何驗證 1sess, 10sess, 100sess, 1000sess 的學習進展？」
**回答**：
- `models/1sess.txt`：剛開始隨機探索（$\epsilon \approx 1.0$），頻繁在前幾步撞牆死亡。
- `models/10sess.txt`：學會初步的避障反射，撞牆次數明顯減少，存活步數上升。
- `models/100sess.txt`：已建立清晰策略，穩定主動追蹤視野內的綠蘋果並避開紅蘋果。
- `models/1000sess.txt`：高度最佳化，長度穩定突破 10 格以上並能長時間存活。

### Q4：「如何執行單步檢驗模式？」
**回答**：
```bash
./snake -visual on -load models/100sess.txt -sessions 10 -dontlearn -step-by-step
```
「在此模式下，Pygame 視窗每一步都會暫停，按一下空白鍵推進一步。終端機同步輸出十字型視野與 Agent 所選動作，且權重完全凍結（`-dontlearn`），方便評審逐步檢驗決策是否合理。」
