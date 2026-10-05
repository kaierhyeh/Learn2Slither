<p align="right">
  <a href="#learn2slither-貪食蛇強化學習">
    <img src="https://img.shields.io/badge/中文-forestgreen?style=for-the-badge" />
  </a>
</p>

# Learn2Slither

An AI-driven Reinforcement Learning Snake game built with **Python** and **Pygame**, strictly following the **Subject**. The snake agent learns to navigate a 10×10 grid, consume green apples to grow, avoid red apples, and survive solely through **Q-learning** based on a restricted 4-direction ray-cast vision.
An AI-driven Reinforcement Learning Snake game built with **Python** and **Pygame**, strictly following the **42 Curriculum Subject**. The snake agent learns to navigate a 10×10 grid, consume green apples to grow, avoid red apples, and survive solely through **Q-learning** based on a restricted 4-direction ray-cast vision.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" />
  <img src="https://img.shields.io/badge/Algorithm-Q--Learning-orange.svg" />
  <img src="https://img.shields.io/badge/GUI-Pygame-green.svg" />
  <img src="https://img.shields.io/badge/Norm-flake8-lightgrey.svg" />
</p>

---

## ✨ Features

- **10×10 Board Environment**:
  - Dynamically spawned **2 Green Apples** (+1 length) and **1 Red Apple** (-1 length).
  - Starts with an initial snake of **3 contiguous cells** placed randomly.
  - Game terminates when hitting a wall, colliding with itself, or when length drops to 0.
- **Strict 4-Direction Ray-cast Vision**:
  - The snake only sees along 4 orthogonal ray casts (`UP`, `LEFT`, `DOWN`, `RIGHT`) from its head.
  - Cell representations: `W` (Wall), `H` (Head), `S` (Body segment), `G` (Green apple), `R` (Red apple), `0` (Empty).
  - **No board-level state leakage** (strictly complies with the `-42` penalty rule).
  - Displays the cross-shaped vision in the terminal before every decision.
- **Pure Q-Learning Architecture (IV.5)**:
  - Tabular Q-learning driven by the Bellman optimality equation:
    $$Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma \max_{a'} Q(s', a') - Q(s, a)]$$
  - $\epsilon$-greedy exploration-exploitation schedule.
  - Fully supports `-dontlearn` mode (pure exploitation without updating Q-values).
- **Decoupled Engine for Blazing-Fast Training**:
  - Headless mode (`-visual off`) decouples rendering and terminal output, simulating tens of thousands of steps per second.
- **Model Persistence**:
  - Full model export and restore (`-save <path>` and `-load <path>`).
  - Pre-trained models included for peer evaluation: `1sess.txt`, `10sess.txt`, `100sess.txt`, and `1000sess.txt`.
- **Interactive Pygame GUI**:
  - Step-by-step stepping mode (`-step-by-step`) using the spacebar.
  - Real-time score, session counter, duration, and speed controls.
  - Multi-language support (EN / ZH / FR) and anti-aliased grid rendering.

---

## 🏗️ Technical Architecture (Subject Part 6)

The project strictly follows the modular architecture defined in the subject:

```
       ┌───────────────────────────┐
       │        ENVIRONMENT        │
       │  (Board, Snake, Rules)    │
       └─────┬───────────────▲─────┘
             │               │
      (Observation)       (Action)
             │               │
             ▼               │
       ┌───────────┐         │
       │INTERPRETER│         │
       └─────┬─────┘         │
             │               │
     (State, Reward)         │
             │               │
             ▼               │
       ┌───────────┐         │
       │   AGENT   ├─────────┘
       │ (Q-table) │
       └───────────┘
```

- **`src/core/`**: Board mechanics, snake body manipulation, and environment step execution.
- **`src/interpreter/`**: Ray-cast line-of-sight vision extraction, terminal cross rendering, and reward calculation.
- **`src/ai/`**: Q-table state mapping, Bellman updates, and file serialization.
- **`src/ui/`**: Pygame graphical display with step-by-step controls.
- **`src/config/`**: Hyperparameters ($\alpha, \gamma, \epsilon$), board constants, and colors.

---

## 🚀 Installation & Compilation

```bash
# Clone the repository
git clone <repo-url> Learn2Slither
cd Learn2Slither

# Install dependencies and build ./snake executable
make
```

### Makefile Rules
- `make all`: Installs dependencies (`pygame`, `flake8`) and generates `./snake`.
- `make clean`: Removes Python cache (`__pycache__`, `*.pyc`).
- `make fclean`: Cleans cache and deletes the `./snake` binary.
- `make re`: Full clean and rebuild.
- `make lint`: Checks code standard compliance using `flake8`.
- `make norm` / `make lint`: Checks code standard compliance using `flake8`.
- `make test`: Runs unit tests.

---

## 🛠️ Usage & Evaluation Commands

### 1. Fast Headless Training (e.g. 10 sessions)
```bash
./snake -sessions 10 -save models/10sess.txt -visual off
```
*Outputs standard completion statistics:*
```
...
Game over, max length = 4, max duration = 17
Save learning state in models/10sess.txt
```

### 2. Peer-Evaluation Verification (Step-by-Step Mode)
```bash
./snake -visual on -load models/100sess.txt -sessions 10 -dontlearn -step-by-step
```
*Loads pre-trained weights without altering the Q-table, stepping through each move with visual and terminal feedback.*

### 3. High-Performance Continuous Play
```bash
./snake -visual on -load models/1000sess.txt
```
*Demonstrates a well-trained snake reaching a length of $\ge 10$ and surviving extended durations.*

---

## 📋 Command-Line Arguments Reference

| Option | Type | Description |
| :--- | :--- | :--- |
| `-sessions <N>` | Integer | Number of training/evaluation games to execute. |
| `-save <path>` | Filepath | Export learned Q-values to target file upon completion. |
| `-load <path>` | Filepath | Import pre-trained model weights before starting. |
| `-visual <on\|off>` | String | Toggle Pygame graphical interface (`on` by default). |
| `-dontlearn` | Flag | Disable learning (pure exploitation, Q-table remains frozen). |
| `-step-by-step` | Flag | Advance simulation one step at a time via keyboard trigger. |

---

## 📚 Project Documentation
- [Implementation Checklist & Roadmap](docs/TODO.md): Full breakdown of subject rules, banned practices, refactoring steps, and bonus features.
- [Development Patch Notes & Changelog](docs/PATCH_NOTES.md): Milestone history from v0.1.0 onwards.
- [42 Official Subject](docs/en.subject.pdf): The original project specification.

<br/>

---

<br/>

<p align="right">
  <a href="#learn2slither">
    <img src="https://img.shields.io/badge/-TOP-forestgreen?style=for-the-badge" />
  </a>
</p>

# Learn2Slither (貪食蛇強化學習)

基於 **Python** 與 **Pygame** 開發之貪食蛇強化學習專案，嚴格遵循 **專案手冊**。AI 代理在 $10 \times 10$ 的網格棋盤中，僅依賴蛇頭四向射線視野，透過純粹的 **Q-learning** 演算法學會尋找綠蘋果增長、避開紅蘋果與牆壁，並維持生存。
基於 **Python** 與 **Pygame** 開發之貪食蛇強化學習專案，嚴格遵循 **42 官方規範手冊 (Subject)**。智能體（Agent）在 $10 \times 10$ 的網格棋盤中，僅依賴蛇頭四向射線視野，透過純粹的 **Q-learning** 演算法學會自主尋找綠蘋果增長、避開紅蘋果與牆壁，並達成持久存活。

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" />
  <img src="https://img.shields.io/badge/演算法-Q--Learning-orange.svg" />
  <img src="https://img.shields.io/badge/圖形介面-Pygame-green.svg" />
  <img src="https://img.shields.io/badge/代碼規範-flake8-lightgrey.svg" />
</p>

---

## ✨ 核心特點

- **$10 \times 10$ 棋盤環境**：
  - 盤面常駐 **2 顆綠蘋果**（長度 $+1$）與 **1 顆紅蘋果**（長度 $-1$）。
  - 起始蛇長度為 **3 格**，隨機且連續放置於棋盤空格。
  - 碰撞牆壁、碰撞自身軀幹、或長度降為 0 時判定遊戲結束。
  - 盤面常駐 **2 顆綠蘋果**（長度 $+1$）與 **1 顆紅蘋果**（長度 $-1$），隨機刷新於空格。
  - 起始蛇長度為 **3 格**，隨機且**連續**（contiguously）放置於棋盤空格。
  - 碰撞牆壁（Wall）、碰撞自身軀幹（Tail）、或長度萎縮至 0 時判定遊戲結束（Game Over）。
- **嚴格四向射線視野 (Ray-cast Vision)**：
  - 蛇只能看到蛇頭朝向四個方向（`UP`, `LEFT`, `DOWN`, `RIGHT`）直線射線上的物體。
  - 視野字元：`W` (牆)、`H` (蛇頭)、`S` (身體)、`G` (綠蘋果)、`R` (紅蘋果)、`0` (空格)。
  - **嚴禁洩漏棋盤全域座標**（杜絕任何導致 -42 分之違規）。
  - 決策前於終端機精準輸出十字型視野與採取之動作。
- **標準 Q-learning 演算法 (IV.5)**：
  - 基於 Bellman 最優方程式更新 Q-table。
  - $\epsilon$-greedy 探索與利用調度。
  - 完整支援 `-dontlearn` 模式（純利用不更新權重，供評審客觀檢視學習成效）。
- **解耦架構與極速訓練**：
  - 支援無介面模式 (`-visual off`)，徹底關閉渲染與終端刷新，每秒可運算數萬步，幾秒鐘即可完成數百至數千回合訓練。
- **模型持久化保存**：
  - 完整模型匯出與載入機制（`-save` 與 `-load`）。
  - 內建繳交必備之預訓練模型：`1sess.txt`、`10sess.txt`、`100sess.txt`、以及長效存活的 `1000sess.txt`。
  - 蛇只能看到從蛇頭朝向四個方向（`UP`, `LEFT`, `DOWN`, `RIGHT`）直線射線上的物體。
  - 視野字元：`W` (牆)、`H` (蛇頭)、`S` (身體節點)、`G` (綠蘋果)、`R` (紅蘋果)、`0` (空格)。
  - **嚴禁洩漏棋盤全域座標**（杜絕任何導致 -42 分之重大違規）。
  - 每次做決策前，於終端機精確輸出十字型視野與採取之動作。
- **純粹的 Q-learning 演算法架構 (Subject IV.5)**：
  - 基於 Bellman 最優方程式更新 Q-table：
    $$Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma \max_{a'} Q(s', a') - Q(s, a)]$$
  - 具備 $\epsilon$-greedy 探索與利用（Exploration vs Exploitation）動態調度。
  - 完整支援 `-dontlearn` 模式（純利用不更新權重，供評審客觀檢視各階段學習成效）。
- **環境與渲染解耦的極速訓練引擎**：
  - 支援無介面模式 (`-visual off`)，徹底關閉 GUI 渲染與終端刷新，每秒可運算數萬步，幾秒鐘即可完成數百至數千回合訓練。
- **模型持久化保存與還原**：
  - 完整模型儲存與載入機制（`-save <path>` 與 `-load <path>`）。
  - 內建繳交必備之預訓練模型檔案：`1sess.txt`、`10sess.txt`、`100sess.txt`、以及長效存活的 `1000sess.txt`。
- **Pygame 互動圖形介面**：
  - 支援單步步進模式 (`-step-by-step`)，可按空白鍵逐步前進。
  - 即時顯示目前長度、存活時間、回合數與動態速度調節。
  - 支援單步步進模式 (`-step-by-step`)，可按空白鍵逐步前進並觀察蛇的即時反應。
  - 即時顯示目前長度、存活時間、回合數、探索率與動態速度調節。
  - 支援多國語系切換（英文 EN / 繁體中文 ZH / 法文 FR）與平滑抗鋸齒網格繪圖。

---

## 🚀 安裝與執行
## 🏗️ 模組化架構藍圖 (Subject Part 6 Technical Structure)

本專案嚴格遵循 42 Subject 規範定義的三大模組分層架構：

```
       ┌───────────────────────────┐
       │        ENVIRONMENT        │
       │  (棋盤、蛇本體、碰撞規則)  │
       └─────┬───────────────▲─────┘
             │               │
       (環境觀察事件)       (動作 Action)
             │               │
             ▼               │
       ┌───────────┐         │
       │INTERPRETER│         │
       └─────┬─────┘         │
             │               │
     (狀態 State, 獎勵 Reward)│
             │               │
             ▼               │
       ┌───────────┐         │
       │   AGENT   ├─────────┘
       │ (Q-table) │
       └───────────┘
```

- **`src/core/`**：棋盤狀態管理、蛇身體隊列操作、蘋果刷新、以及 `step(action)` 環境執行引擎。
- **`src/interpreter/`**：四向射線視野提取、終端機十字型圖樣渲染、以及多層次獎勵計算（吃綠+、吃紅-、步數-、死亡--）。
- **`src/ai/`**：Q-table 狀態映射、Bellman 方程式更新、$\epsilon$-greedy 決策邏輯、以及純文字模型存取。
- **`src/ui/`**：Pygame 圖形視覺化介面、步進模式事件循環與速度控制。
- **`src/config/`**：全域超參數（$\alpha, \gamma, \epsilon$）、棋盤常數、配色方案與獎勵設定。

---

## 🚀 安裝與編譯構建

```bash
# 取得專案並編譯生成可執行檔 ./snake
# 取得專案原始碼
git clone <repo-url> Learn2Slither
cd Learn2Slither

# 安裝相依套件並編譯生成可執行檔 ./snake
make
```

### 評分與測試常用指令範例
### Makefile 核心目標說明
- `make` / `make all`：驗證相依套件（`pygame`, `flake8`）並生成 `./snake` 執行包裝腳本。
- `make clean`：清理 Python 快取目錄（`__pycache__`, `*.pyc`），具備平滑進度條反饋。
- `make fclean`：完整清除快取並移除可執行檔 `./snake`。
- `make re`：一鍵深度清理並重新構建。
- `make norm` / `make lint`：使用 `flake8` 檢驗程式碼是否符合 42 Python 代碼規範。
- `make test`：執行單元測試套件（驗證蛇行為與視野產生）。
- `make help`：顯示所有目標與題目規格指令範例。

1. **高速無介面訓練（訓練 10 回合並儲存）：**
   ```bash
   ./snake -sessions 10 -save models/10sess.txt -visual off
   ```
---

2. **評分檢驗模式（載入 100 回合模型、單步推進、不更新 Q 表）：**
   ```bash
   ./snake -visual on -load models/100sess.txt -sessions 10 -dontlearn -step-by-step
   ```
## 🛠️ 評分與測試常用指令範例

3. **高水平模型展示（長度達 10 以上並長效存活）：**
   ```bash
   ./snake -visual on -load models/1000sess.txt
   ```
### 1. 極速無介面訓練（訓練 10 回合並儲存模型）
```bash
./snake -sessions 10 -save models/10sess.txt -visual off
```
*終端輸出標準結尾統計格式（符合 Subject Page 13 規範）：*
```
...
Game over, max length = 4, max duration = 17
Save learning state in models/10sess.txt
```

### 2. 評審驗收檢驗模式（載入 100 回合模型、單步推進、凍結權重）
```bash
./snake -visual on -load models/100sess.txt -sessions 10 -dontlearn -step-by-step
```
*載入已訓練權重且不更新 Q 表，在圖形視窗中按空白鍵單步前進，終端同步輸出十字視野與決策動作。*

### 3. 高性能模型連續展示（長度達 10 以上並長效存活）
```bash
./snake -visual on -load models/1000sess.txt
```
*流暢展示高度訓練後的蛇自動避開紅蘋果與牆壁，迅速達到長度 10 格以上並長效存活。*

---

## 📚 相關文檔
- [實作任務清單與規範對照 (TODO.md)](docs/TODO.md)
- [42 官方規範手冊 (en.subject.pdf)](docs/en.subject.pdf)
## 📋 命令列參數完全手冊 (CLI Reference)

| 參數選項 | 類型 | 說明 |
| :--- | :--- | :--- |
| `-sessions <N>` | 整數 (Integer) | 指定要執行的訓練或評估回合數（Session 數量）。 |
| `-save <path>` | 檔案路徑 (Filepath) | 訓練結束後，將當前學習到的 Q-values 匯出至指定文字檔。 |
| `-load <path>` | 檔案路徑 (Filepath) | 啟動時先載入預訓練模型檔案，接續學習或進行純利用展示。 |
| `-visual <on\|off>` | 字串 (String) | 開啟或關閉 Pygame 圖形視覺化介面（預設為 `on`）。 |
| `-dontlearn` | 開關 (Flag) | 停用學習更新（純利用 Exploitation，不更新 Q-table 權重）。 |
| `-step-by-step` | 開關 (Flag) | 啟用單步步進模式，按鍵推進一步，方便逐手檢驗視野與決策。 |

---

## 📚 專案文檔索引
- [實作任務清單與規範對照 (TODO.md)](docs/TODO.md)：Subject 規範剖析、三大禁區警示、重構步驟與驗收清單。
- [版本演進與補丁歷程 (PATCH_NOTES.md)](docs/PATCH_NOTES.md)：從 v0.1.0 起的里程碑紀錄與詳細變更。
- [42 官方規範手冊 (en.subject.pdf)](docs/en.subject.pdf)：官方專案要求與評分標準原件。