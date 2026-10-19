# Learn2Slither 專案版本演進與補丁歷程 (Patch Notes)

本文檔記錄 **Learn2Slither**（貪食蛇強化學習 AI）專案從 5eyes 核心架構奠基、規範解構、工程基礎設施改造、到後續強化學習核心實作的完整軟體工程生命週期。

---

## 專案里程碑總覽 (Version Timeline)

| 版本標籤 | 發布日期 | 核心主題 | 關鍵成果與亮點 |
|---|---|---|---|
| **v0.2.0** | 2026-10-09 | **代碼庫重構、清理與 UI 統一 (Refactoring & UI Standardization)** | 刪除 39MB 舊字型與五子棋殘留模組、整併狀態管理、全面將 GUI 統一為 UI、通過 `make norm` (flake8 0 warnings)、重寫 `README.md`、`TODO.md` 與 `walkthrough.md`。 |
| **v0.1.0** | 2026-10-05 | **專案工程基礎設施與規範奠基 (Foundation & Toolchain)** | 深度解析 42 Subject 規範手冊、建立 `TODO.md` 六階段藍圖、全面重寫 `README.md`、完成標準 `./snake` 包裝腳本、實現共用平滑動態進度條（`DRAW_BAR`）、整合 `flake8` 與 42 規範。 |

---

## 歷次版本詳細變更記錄 (Detailed Changelog)

### v0.2.0 (2026-10-09) - 代碼庫重構、清理與 UI 統一 (Refactoring & UI Standardization)

#### 1. 徹底清理五子棋遺留模組與字型資源
- **刪除 `assets/fonts/`（~39MB）**：移除 `NotoSansTC-Regular.otf` 與 `NotoEmoji-Regular.ttf`，42 Subject 僅需標準 ASCII 終端輸出與 Pygame 內建等寬字體，無需笨重的 CJK 字型包袱。
- **清理舊模組**：刪除舊五子棋專屬的 `src/core/game.py`、`src/ai/ai.py` (Minimax)、`src/ai/heuristic.py`、`src/ui/i18n.py`。
- **整併狀態管理**：刪除冗餘的 `src/core/state.py`，將所有狀態追蹤、射線視野特徵提取與終端十字渲染整併至 `src/core/environment.py` 與 `src/core/snake.py`。

#### 2. GUI 全面統一為 UI (Naming & Design Consistency)
- 將 `src/ui/gui.py` 中的核心類別重構為 `class UI:`。
- `src/ui/__init__.py` 導出 `from ui.gui import UI`。
- 更新 `src/main.py` 與 `src/config/ui.py`，保持全專案 `UI` 命名一致。

#### 3. 代碼規範檢查與文檔同步升級
- 全專案通過 `make norm`（`flake8 --max-line-length=100`），達成 0 錯誤、0 警告。
- 全面改寫並重構 [`README.md`]、[`docs/TODO.md`] 與 [`docs/walkthrough.md`]，完全轉向 Learn2Slither 規格與答辯指引。

---

### v0.1.0 (2026-10-05) - 專案工程基礎設施與規範奠基 (Foundation & Toolchain)

#### 1. 42 官方規範手冊 (Subject) 深度解析與考題紅線確立
- **全面研讀 `docs/en.subject.pdf`（共 16 頁全文）**，確立三條零容忍的評分紅線：
  - 🚨 **AI 演算法唯一性（IV.5）**：強制限定只能採用 **Q-learning**（Q-table 或 Neural Network），嚴禁任何 Minimax、Alpha-Beta、A* 或純貪婪啟發式規則（違者 0 分）。
  - 🚨 **狀態視野限制（IV.2）**：蛇僅具備從頭部出發之 4 向射線視野（`W`, `H`, `S`, `G`, `R`, `0`），嚴格禁止將整盤全域座標傳遞給 Agent（違者 -42 分）。
  - 🚨 **模型持久化驗收（Chapter V）**：必須支援 `-save`, `-load`, `-dontlearn`，且提交時需內建 1、10、100 sessions 等不同階段的預訓練模型檔案供評分檢視。
- **確認 Part 6 Technical Structure 架構要求**：
  - 官方確立 `ENVIRONMENT` ⇄ `AGENT` 並由 `INTERPRETER` 提取 `STATE` 與 `REWARD` 的三角分層架構。
  - 明確確認貪食蛇規則（撞牆、撞身、吃紅蘋果長度縮為 0）輕量且屬於環境狀態轉移的一環，無須過度設計獨立的 `rules.py`。

#### 2. 規劃文件與專案規範確立
- **建立實作藍圖 [`docs/TODO.md`]**：
  - 整理完整的規格要求、三大禁區警示、模組化架構藍圖。
  - 列出 5eyes 檔案去留與重構清單（標記刪除五子棋專用檔案、保留修改核心設定與 UI、新增 Interpreter 與 Q-learning 模組）。
  - 規劃 Phase 1 至 Phase 6 的循序開發與驗收 Checklist。
- **全面改寫 [`README.md`]**：
  - 提供專業的中英雙語專案說明文檔。
  - 包含 Q-learning 數學公式、四向十字視野說明、模組架構圖。
  - 詳述評審專用的命令行範例（`-sessions`, `-save`, `-load`, `-visual`, `-dontlearn`, `-step-by-step`）。

#### 3. 構建系統與 Makefile 全面重構
- **生成標準 `./snake` 執行包裝腳本**：
  - 根除舊版 5eyes 中 `cd src` 導致相對路徑被破壞的致命 Bug（避免評審輸入 `-save models/10sess.txt` 時引發 `FileNotFoundError`）。
  - 改用 `BASE_DIR="$(cd "$(dirname "$0")" && pwd)"` 與 `export PYTHONPATH="$BASE_DIR/src:$PYTHONPATH"`，確保任何執行目錄下皆能精確定位專案與讀寫 `models/`。
- **抽出通用平滑進度條巨集 (`DRAW_BAR`)**：
  - 抽取可重用的 Bash 繪圖函式，消除複製 5 遍 `printf + sleep` 的髒代碼。
  - 支援完整自訂：文字、顏色（ANSI）、圖示（Emoji）與延遲速度。
  - 提供 20% ➔ 40% ➔ 60% ➔ 80% ➔ 100% 的原地（In-place `\033[2K\r`）流暢動畫回饋。
  - 全面套用於 `all`, `install`, `$(NAME)`, `clean`, `fclean`。
- **依賴與規範管理**：
  - 新增 [`requirements.txt`]（包含 `pygame` 與 `flake8`）。
  - 支援 `make norm` / `make lint` 執行 `flake8` 程式碼規範審查。
  - 支援 `make test` 執行單元測試套件。
  - 支援 `make help` 輸出完整命令列說明手冊。
