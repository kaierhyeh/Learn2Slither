# Learn2Slither 實作與重構任務清單 (TODO)

本文件依據 42 官方規範手冊 [`docs/en.subject.pdf`] 與 5eyes 成功架構經驗，詳細規劃 Learn2Slither 專案之重構、移植、實作步驟與驗收標準。

---

## 📌 目錄
1. [42 Subject 核心規格與硬性規定](#1-42-subject-核心規格與硬性規定)
2. [三大考題禁區（零容忍紅線）](#2-三大考題禁區零容忍紅線)
3. [專案模組架構藍圖 (Technical Structure)](#3-專案模組架構藍圖-technical-structure)
4. [現有檔案去留清單 (From 5eyes to Learn2Slither)](#4-現有檔案去留清單-from-5eyes-to-learn2slither)
5. [實作階段性工作清單 (Implementation Checklist)](#5-實作階段性工作清單-implementation-checklist)
6. [CLI 參數與驗收標準 (Evaluation Checklist)](#6-cli-參數與驗收標準-evaluation-checklist)
7. [加分項目規劃 (Bonus Features)](#7-加分項目規劃-bonus-features)

---

## 1. 42 Subject 核心規格與硬性規定

- **環境與棋盤 (Board & Environment)**：
  - 棋盤大小固定為 $10 \times 10$ 格（Bonus 支援自訂維度）。
  - 盤面常態維持 **2 顆綠蘋果（Green Apples）** 與 **1 顆紅蘋果（Red Apple）**，位置隨機刷新於空格。
  - 蛇起始長度為 **3 格**，隨機且**連續**（contiguously）放置於棋盤空格。
- **移動與長度機制**：
  - 動作集合：`UP`, `LEFT`, `DOWN`, `RIGHT`。
  - 吃綠蘋果：蛇長度 $+1$，並在空位重新生成 1 顆綠蘋果。
  - 吃紅蘋果：蛇長度 $-1$，並在空位重新生成 1 顆紅蘋果。
- **回合結束條件 (Game Over)**：
  - 蛇頭撞牆（Hit a wall）。
  - 蛇頭撞自身（Collides with its own tail/body）。
  - 蛇長度歸零（Length drops to 0，因吃紅蘋果）。
- **目標 (Goal)**：
  - 在回合結束時達到長度至少 **10 格以上**（$\ge 10$ cells），且存活時間越長越好。

---

## 2. 三大考題禁區（零容忍紅線）

### 🚨 禁區 1：AI 演算法限制（違者 0 分）
- **規定**：`No other model is allowed; otherwise, a score of 0 will be given.`
- **要求**：**強制只能使用 Q-learning**（Q-values in a Q-table 或 Neural Network）。
- **禁忌**：嚴禁使用 5eyes 的 Minimax、Alpha-Beta 剪枝、A* 尋路演算法或單純貪婪啟發式規則作為決策主體。

### 🚨 禁區 2：狀態可見度限制（違者 -42 分）
- **規定**：`You can only provide to the agent the information visible to your snake. Providing more information will result in a penalty of -42.`
- **要求**：蛇只能看見從蛇頭出發、四個正交方向（上、下、左、右）直線射線上的資訊。
- **視野字元**：
  - `W`：牆壁 (Wall)
  - `H`：蛇頭 (Snake Head)
  - `S`：蛇身 (Snake body segment)
  - `G`：綠蘋果 (Green apple)
  - `R`：紅蘋果 (Red apple)
  - `0`：空格 (Empty space)
- **終端機輸出**：每次請求 Agent 決策前，必須在 Terminal 印出十字型視野與採取的 Action。
- **訓練加速**：在訓練模式（`-visual off`）下，允許關閉 UI 與終端機輸出以達成極速訓練（每秒數萬步）。

### 🚨 禁區 3：模型持久化與評分驗證
- **規定**：repository 內**必須至少提交 3 個已訓練模型**：
  - `models/1sess.txt`（訓練 1 個 session）
  - `models/10sess.txt`（訓練 10 個 sessions）
  - `models/100sess.txt`（訓練 100 個 sessions）
  - （可額外提供 `models/1000sess.txt` 以展示最佳性能）
- **要求**：必須支援 `-dontlearn`（純 exploitation，不更新 Q-table），供評分者載入模型測試表現。
- **終端機標準輸出格式**：
  ```
  Game over, max length = <int>, max duration = <int>
  Save learning state in <filepath>
  Load trained model from <filepath>
  ```

---

## 3. 專案模組架構藍圖 (Technical Structure)

符合 Subject Part 6 的架構分層：
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

### 實際模組目錄架構
```
Learn2Slither/
├── Makefile                # 生成可執行檔 ./snake、flake8 檢測 (norm)、依賴管理
├── requirements.txt        # 外部相依套件 (pygame, flake8)
├── README.md               # 專案說明與操作指南
├── models/                 # 預訓練模型 (1sess.txt, 10sess.txt, 100sess.txt, 1000sess.txt)
├── docs/                   # 題目文檔 (en.subject.pdf) 與任務清單 (TODO.md)
└── src/
    ├── main.py             # CLI 參數解析與進入點 (協調訓練/展示迴圈)
    ├── config/
    │   ├── __init__.py
    │   ├── game.py         # 棋盤大小 (10x10)、物件字元、方向常數 (DIRECTIONS)
    │   ├── ai.py           # Q-learning 超參數 (α, γ, ε)、獎勵常數
    │   └── ui.py           # Pygame 視窗尺寸、格線色彩、按鍵映射 (KEY_TO_DIR)
    ├── core/
    │   ├── __init__.py
    │   ├── snake.py        # 蛇本體行為 (連續生成、移動、增長、縮短、碰撞檢測)
    │   └── environment.py  # 環境引擎 (reset, step, 蘋果生成, 4向射線特徵提取, 十字視野渲染)
    ├── ai/
    │   ├── __init__.py
    │   └── q_table.py      # Q-learning 智能體 (QTableAgent: ε-greedy, Bellman 更新, 模型存讀檔)
    └── ui/
        ├── __init__.py
        └── gui.py          # Pygame 視覺化介面 (UI: 網格渲染、HUD、按鍵事件監聽)
```

---

## 4. 現有檔案去留與重構清單

### ✏️ 保留並改造完成檔案
- [x] **`Makefile`**：
  - 將目標可執行檔名改為 `snake`。
  - 調整執行腳本，保持專案根目錄為 Working Directory，確保 `./models/` 檔案定位正確。
  - 保留 `norm` 目標執行 flake8（符合 42 規範）。
- [x] **`requirements.txt`**：
  - 設定 `pygame>=2.5.0` 與 `flake8>=6.0.0`。
- [x] **`src/main.py`**：
  - 支援 `-sessions`, `-save`, `-load`, `-visual`, `-dontlearn`, `-step-by-step`, `-board-size`。
  - 協調訓練迴圈（極速無介面模式）與展示迴圈（Pygame UI + Terminal 視野）。
- [x] **`src/config/game.py`**：
  - 定義 10x10 棋盤、物件代碼、單一方向字典 `DIRECTIONS`。
- [x] **`src/config/ai.py`**：
  - 定義 Q-learning 超參數（$\alpha, \gamma, \epsilon$）與 Reward 配置。
- [x] **`src/config/ui.py`**：
  - 深色主題配置、格點大小、蛇身與蘋果色彩、Pygame 鍵盤按鍵映射 `KEY_TO_DIR`。
- [x] **`src/core/snake.py`**：
  - 蛇模型邏輯（隨機連續 3 格生成、移動、長度增減、碰撞檢測）。
- [x] **`src/core/environment.py`**：
  - Snake Environment，整合狀態追蹤、蘋果生成、4 向射線狀態特徵提取與十字視野生成。
- [x] **`src/ai/q_table.py`**：
  - Q-table 管理、Bellman 方程更新、$\epsilon$-greedy 決策與模型存讀檔。
- [x] **`src/ui/gui.py`** (`UI`)：
  - 10x10 網格渲染、抬頭 HUD 資訊、按鍵事件監聽（ESC, Space, WASD/方向鍵, 調速）。

---

## 5. 實作階段性工作清單 (Implementation Checklist)

### Phase 1: 代碼清理與環境初始化
- [x] 執行清理指令，刪除所有五子棋專屬檔案。
- [x] 修改 `Makefile`：產出可執行檔 `./snake`，整合 `flake8` 檢查（`make norm`）。
- [x] 設定 `requirements.txt`：記錄必要套件依賴。
- [x] 設定 `src/config/` 常數（`game.py`, `ai.py`, `ui.py`）與單一方向字典 `DIRECTIONS`、`KEY_TO_DIR`。

### Phase 2: 貪食蛇核心環境 (Environment & Snake)
- [x] 實作 `src/core/snake.py`：
  - 隨機連續 3 格初始化。
  - 依動作前進，正確處理頭尾長度變化與自我碰撞檢測。
- [x] 實作 `src/core/environment.py`：
  - 隨機生成 2 顆綠蘋果與 1 顆紅蘋果（不重疊）。
  - 執行動作 `step(action)`，返回 next_state, reward, done, info。
  - 碰撞判定：撞牆、撞身、長度為 0 結束。

### Phase 3: 視野解析與獎勵機制 (Interpreter)
- [x] 4 向射線掃描與十字視野渲染：
  - 向 UP, LEFT, DOWN, RIGHT 四方向發射射線直到撞牆。
  - 生成終端機要求的十字型圖樣（`W`, `H`, `S`, `G`, `R`, `0`）。
- [x] 緊湊狀態特徵提煉（`get_state()`）：
  - 提取 20 元素特徵元組（距離 1 危險、首個物體、距離分桶、綠/紅蘋果可見標記）。
  - 嚴格遵守僅限可見視野資訊，杜絕 -42 分懲罰。
- [x] 獎勵評估機制：
  - 吃綠蘋果：$+10$（正獎勵）。
  - 吃紅蘋果：$-10$（負獎勵）。
  - 存活移動：$-0.1$（微小負獎勵，避免無意義迴圈）。
  - 死亡或長度為 0：$-50$（巨大懲罰）。

### Phase 4: Q-learning 智能體與模型存取 (Agent & Storage)
- [x] 實作 `src/ai/q_table.py`：
  - Q-table 映射與 Bellman 更新公式：
    $$Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma \max_{a'} Q(s', a') - Q(s, a)]$$
  - $\epsilon$-greedy 行為選擇（支援 `--dontlearn` 時強制 $\epsilon = 0$ 且不更新權重）。
- [x] 模型序列化與反序列化（`.txt` / `.json` 存檔/載入）：
  - 輸出格式符合題目規格：`Save learning state in <file>` / `Load trained model from <file>`。

### Phase 5: 渲染與終端互動 (UI & CLI)
- [x] 改造 `src/ui/gui.py` (`UI`)：
  - 10x10 網格渲染，藍色蛇身、綠蘋果、紅蘋果。
  - 支援 `-step-by-step`（按空白鍵單步前進、按方向鍵/WASD 手動轉向）。
  - 支援動態速度調節（`+`/`-` 調整 FPS）。
- [x] 整合 `src/main.py`：
  - 完整支援所有命令行參數。
  - 在 `-visual off` 時達到每秒萬步的高速極速訓練。
  - 每局結束時輸出標準結算：`Game over, max length = X, max duration = Y`。

### Phase 6: 訓練產出、測試與規範驗收
- [x] 執行 `flake8` 檢驗（`make norm`），確保整個專案程式碼風格零錯誤、零警告。
- [x] 執行訓練並生成必備模型檔：
  - `models/1sess.txt`（訓練 1 個 session）
  - `models/10sess.txt`（訓練 10 個 sessions）
  - `models/100sess.txt`（訓練 100 個 sessions）
  - `models/1000sess.txt`（充分訓練，長度能達到 10 格以上並長效存活）
- [x] 撰寫單元測試 `tests/test_snake.py`，確認環境與規則 100% 正確。

---

## 6. CLI 參數與驗收標準 (Evaluation Checklist)

評審將依照以下指令與行為進行測試：

| 測試情境 | 命令行範例 | 預期行為 |
| :--- | :--- | :--- |
| **快速訓練 10 回合** | `./snake -sessions 10 -save models/10sess.txt -visual off` | 無 UI 視窗，高速跑完 10 回合並儲存模型，終端輸出 Game over 與 Save 訊息。 |
| **載入模型單步檢驗** | `./snake -visual on -load models/100sess.txt -sessions 10 -dontlearn -step-by-step` | 載入模型，不進行學習更新；UI 彈出，每次按 Space 前進一步，終端印出十字視野與決策動作。 |
| **連續播放長度驗收** | `./snake -visual on -load models/1000sess.txt` | UI 流暢播放蛇自動尋找綠蘋果、閃避紅蘋果與牆壁，最終長度達 10 以上。 |

---

## 7. 加分項目規劃 (Bonus Features)

- [x] **高長度挑戰**：蛇在結束時長度突破 15、20、25 甚至 30 格。
- [x] **精緻視覺化介面**：即時長度與存活時間儀表板、探索率與 Q-value 顯示、歷史最佳紀錄統計。
- [x] **動態棋盤尺寸**：支援 `-board-size <N>` 參數，並驗證已訓練的射線模型可泛化適應不同棋盤大小。
