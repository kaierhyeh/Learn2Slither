# Learn2Slither 實作與重構任務清單 (TODO)

本文件依據 42 官方規範手冊 [`docs/en.subject.pdf`](file:///home/kyeh/Learn2Slither/docs/en.subject.pdf) 與 5eyes 成功架構經驗，詳細規劃 Learn2Slither 專案之重構、移植、實作步驟與驗收標準。

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
- **訓練加速**：在訓練模式（`-visual off`）下，允許關閉 GUI 與終端機輸出以達成極速訓練（每秒數萬步）。

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

### 推薦目錄架構
```
Learn2Slither/
├── Makefile                # 編譯生成可執行檔 ./snake、flake8 檢測、依賴管理
├── README.md               # 專案說明與操作指南
├── models/                 # 預訓練模型 (1sess.txt, 10sess.txt, 100sess.txt, 1000sess.txt)
├── docs/                   # 題目文檔 (en.subject.pdf) 與任務清單 (TODO.md)
├── assets/                 # 評分字型與資源
│   └── fonts/              # NotoSansTC-Regular.otf, NotoEmoji-Regular.ttf
├── tests/                  # 單元測試 (環境規則、射線視野、Q-learning)
└── src/
    ├── main.py             # CLI 參數解析與進入點
    ├── config/
    │   ├── __init__.py
    │   ├── game.py         # 棋盤大小 (10x10)、物件編號、動作常數
    │   ├── ai.py           # Q-learning 超參數 (α, γ, ε)、獎勵常數
    │   └── ui.py           # Pygame 視窗尺寸、格線、配色、FPS 設定
    ├── core/
    │   ├── __init__.py
    │   ├── state.py        # 貪食蛇狀態資料類別 (座標、佇列、長度、時間)
    │   ├── snake.py        # 蛇本體行為 (移動、增長、縮短、碰撞檢測)
    │   └── environment.py  # 環境引擎 (reset, step, 蘋果生成, 遊戲結束判定)
    ├── interpreter/
    │   ├── __init__.py
    │   ├── vision.py       # 四向射線掃描、終端機十字視野字串、狀態特徵提取
    │   └── reward.py       # 獎勵計算函式 (吃綠+、吃紅-、步數-、死亡--)
    ├── ai/
    │   ├── __init__.py
    │   ├── q_table.py      # Q-learning 核心 (Q-table, ε-greedy, Bellman 更新)
    │   └── storage.py      # 模型序列化與反序列化 (.txt 存檔/載入)
    └── ui/
        ├── __init__.py
        ├── gui.py          # Pygame 視覺化 (抗鋸齒繪圖、步進模式、速度控制)
        └── i18n.py         # 介面多語言支援 (EN / ZH / FR - Bonus 友善)
```

---

## 4. 現有檔案去留清單 (From 5eyes to Learn2Slither)

### ✏️ 需保留並改造檔案
- [ ] **`Makefile`**：
  - 將目標可執行檔名 `Gomoku` 改為 `snake`。
  - 調整執行腳本，保持專案根目錄為 Working Directory，確保 `./models/` 檔案定位正確。
  - 加入 `lint` 目標執行 `flake8`。
- [ ] **`src/main.py`**：
  - 改寫 CLI 參數解析，支援 `-sessions`, `-save`, `-load`, `-visual`, `-dontlearn`, `-step-by-step`。
  - 控制訓練迴圈（極速模式）與展示迴圈（Pygame GUI + Terminal 視野）。
- [ ] **`src/config/game.py`**：
  - 改為 10x10 棋盤、物件代碼、動作常數。
- [ ] **`src/config/ai.py`**：
  - 改為 Q-learning 超參數（$\alpha, \gamma, \epsilon$）與 Reward 配置。
- [ ] **`src/config/ui.py`**：
  - 保留深色主題，調整格點大小與蛇身（藍）、綠蘋果（綠）、紅蘋果（紅）色彩。
- [ ] **`src/core/state.py`**：
  - 改造為貪食蛇專屬狀態資料結構（蛇頭、身體 deque、蘋果座標）。
- [ ] **`src/core/game.py` -> `src/core/environment.py`**：
  - 改造為 Snake Environment，支援 `reset()` 與 `step(action)`。
- [ ] **`src/ui/gui.py`**：
  - 改造為 10x10 網格渲染，加入 step-by-step 手動單步推進與動態速度調節。
- [ ] **`src/ui/i18n.py`** 與 **`assets/fonts/`**：
  - 保留字型與多語系支援，做為 UI 豐富度 Bonus。

### ➕ 需全新建立之檔案
- [ ] **`src/core/snake.py`**：蛇模型邏輯（移動、長度增減、連續初始生成）。
- [ ] **`src/interpreter/vision.py`**：四向射線掃描、終端機十字視野渲染、特徵狀態轉換。
- [ ] **`src/interpreter/reward.py`**：狀態獎勵評估器。
- [ ] **`src/ai/q_table.py`**：Q-table 管理與更新。
- [ ] **`src/ai/storage.py`**：模型檔案讀寫儲存器。
- [ ] **`models/`**：建立預先訓練模型資料夾（1sess.txt, 10sess.txt, 100sess.txt, 1000sess.txt）。
- [ ] **`tests/test_snake.py`**：環境與視野測試腳本。

---

## 5. 實作階段性工作清單 (Implementation Checklist)

### Phase 1: 代碼清理與環境初始化
- [ ] 執行清理指令，刪除所有五子棋專屬檔案。
- [ ] 修改 `Makefile`：產出可執行檔 `./snake`，整合 `flake8` 檢查。
- [ ] 設定 `src/config/` 常數（`game.py`, `ai.py`, `ui.py`）。

### Phase 2: 貪食蛇核心環境 (Environment & Snake)
- [ ] 實作 `src/core/state.py`：儲存蛇位置、身體隊列、蘋果位置、存活時間。
- [ ] 實作 `src/core/snake.py`：
  - 隨機連續 3 格初始化。
  - 依動作前進，正確處理頭尾長度變化。
- [ ] 實作 `src/core/environment.py`：
  - 隨機生成 2 顆綠蘋果與 1 顆紅蘋果（不重疊）。
  - 執行動作 `step(action)`，返回 next_state, reward, done, info。
  - 碰撞判定：撞牆、撞身、長度為 0 結束。

### Phase 3: 視野解析與獎勵機制 (Interpreter)
- [ ] 實作 `src/interpreter/vision.py`：
  - 向 UP, LEFT, DOWN, RIGHT 四方向發射射線，直到撞牆。
  - 生成終端機要求的十字型圖樣（`W`, `H`, `S`, `G`, `R`, `0`）。
  - 提煉緊湊的狀態特徵供 Agent 學習（嚴格遵守僅限可見視野資訊，杜絕 -42 分懲罰）。
- [ ] 實作 `src/interpreter/reward.py`：
  - 吃綠蘋果：$+10$（正獎勵）。
  - 吃紅蘋果：$-10$（負獎勵）。
  - 存活移動：$-0.1$（微小負獎勵，避免無意義繞圈）。
  - 死亡或長度為 0：$-50$（巨大懲罰）。

### Phase 4: Q-learning 智能體與模型存取 (Agent & Storage)
- [ ] 實作 `src/ai/q_table.py`：
  - Q-table 映射與 Bellman 更新公式：
    $$Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma \max_{a'} Q(s', a') - Q(s, a)]$$
  - $\epsilon$-greedy 行為選擇（支援 `--dontlearn` 時強制 $\epsilon = 0$ 且不更新權重）。
- [ ] 實作 `src/ai/storage.py`：
  - 序列化 Q-table 至純文字檔（支援 `-save` 與 `-load`）。
  - 輸出格式符合題目規格：`Save learning state in <file>` / `Load trained model from <file>`。

### Phase 5: 渲染與終端互動 (UI & CLI)
- [ ] 改造 `src/ui/gui.py`：
  - 10x10 網格渲染，藍色蛇身、綠蘋果、紅蘋果。
  - 支援 `-step-by-step`（按空白鍵單步前進）。
  - 支援速度調節（按鍵加速/減速）。
- [ ] 整合 `src/main.py`：
  - 完整支援所有命令行參數。
  - 在 `-visual off` 時達到每秒萬步的高速訓練。
  - 回合結束時輸出：`Game over, max length = X, max duration = Y`。

### Phase 6: 訓練產出、測試與規範驗收
- [ ] 撰寫單元測試 `tests/test_snake.py`，確認規則 100% 正確。
- [ ] 執行 `flake8` 檢驗，確保程式碼風格零 warning。
- [ ] 執行訓練並生成必備模型檔：
  - `models/1sess.txt`
  - `models/10sess.txt`
  - `models/100sess.txt`
  - `models/1000sess.txt`（證明長度能達到 10 格以上並長效存活）。

---

## 6. CLI 參數與驗收標準 (Evaluation Checklist)

評審將依照以下指令與行為進行測試：

| 測試情境 | 命令行範例 | 預期行為 |
| :--- | :--- | :--- |
| **快速訓練 10 回合** | `./snake -sessions 10 -save models/10sess.txt -visual off` | 無 GUI 視窗，高速跑完 10 回合並儲存模型，終端輸出 Game over 與 Save 訊息。 |
| **載入模型單步檢驗** | `./snake -visual on -load models/100sess.txt -sessions 10 -dontlearn -step-by-step` | 載入模型，不進行學習更新；GUI 彈出，每次按鍵前進一步，終端印出十字視野與決策動作。 |
| **連續播放長度驗收** | `./snake -visual on -load models/1000sess.txt` | GUI 流暢播放蛇自動尋找綠蘋果、閃避紅蘋果與牆壁，最終長度達 10 以上。 |

---

## 7. 加分項目規劃 (Bonus Features)

- [ ] **高長度挑戰**：蛇在結束時長度突破 15、20、25 甚至 30 格。
- [ ] **精緻視覺化介面**：即時長度與存活時間儀表板、探索率與 Q-value 顯示、歷史最佳紀錄統計。
- [ ] **動態棋盤尺寸**：支援 `-board-size <N>` 參數，並驗證已訓練的射線模型可泛化適應不同棋盤大小。
