# Learn2Slither 專案版本演進與補丁歷程 (Patch Notes)

本文檔記錄 **Learn2Slither**（貪食蛇強化學習 AI）專案從 5eyes 核心架構奠基、規範解構、工程基礎設施改造、到後續強化學習核心實作的完整軟體工程生命週期。

---

## 專案里程碑總覽 (Version Timeline)

| 版本標籤 | 發布日期 | 核心主題 | 關鍵成果與亮點 |
|---|---|---|---|
| **v0.1.0** | 2026-10-05 | **專案工程基礎設施與規範奠基 (Foundation & Toolchain)** | 深度解析 42 Subject 規範手冊、建立 `TODO.md` 六階段藍圖、全面重寫 `README.md`、完成標準 `./snake` 包裝腳本、實現共用平滑動態進度條（`DRAW_BAR`）、整合 `flake8` 與 42 規範。 |
| *(Heritage)* **v1.3.3** | 2026-09-17 | *5eyes 介面易讀性優化與文檔校準* | （5eyes 繼承架構）英／法雙行提示排版、雙色按鍵高亮、視窗標題統一、測試 20/20 通過。 |

---

## 歷次版本詳細變更記錄 (Detailed Changelog)

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
- **建立實作藍圖 [`docs/TODO.md`](file:///home/kyeh/Learn2Slither/docs/TODO.md)**：
  - 整理完整的規格要求、三大禁區警示、模組化架構藍圖。
  - 列出 5eyes 檔案去留與重構清單（標記刪除五子棋專用檔案、保留修改核心設定與 UI、新增 Interpreter 與 Q-learning 模組）。
  - 規劃 Phase 1 至 Phase 6 的循序開發與驗收 Checklist。
- **全面改寫 [`README.md`](file:///home/kyeh/Learn2Slither/README.md)**：
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
  - 新增 [`requirements.txt`](file:///home/kyeh/Learn2Slither/requirements.txt)（包含 `pygame` 與 `flake8`）。
  - 支援 `make norm` / `make lint` 執行 `flake8` 程式碼規範審查。
  - 支援 `make test` 執行單元測試套件。
  - 支援 `make help` 輸出完整命令列說明手冊。

---

### *(Heritage from 5eyes)* v1.3.3 (2026-09-17) - 介面易讀性優化、指南校準與標題統一

#### 1. 右下角快捷鍵提示詞雙行自適應排版
- 中文維持單行 3 欄並列，英／法文自動啟用雙行居中與邊角對齊，徹底消除文字下半部超出螢幕底緣被截切問題。
- 引入 `pygame.transform.smoothscale` 進行視窗窄邊界縮放保護。

#### 2. 按鍵雙色層次高亮 (Key Highlight)
- 括弧快捷鍵部分採用金色 `COLOR_ACCENT` 高亮，功能文字保持柔和米白，提升易讀性。

#### 3. 流星模式指南說明精確化
- 更正不精確描述：星石每 3 回合自變黑白顏色（不影響周圍棋子），同步校正三國語言。

#### 4. 視窗標題列規範化
- 統一為 `set_caption("Gomoku")`，避免動態標題與面板資訊冗餘脫節。

---

### *(Heritage from 5eyes)* v1.3.2 (2026-09-17) - AI 戰術評估權重倒掛修復與安全剪枝
- 修復連五分數倒掛破綻，建立己方成五（10,000,000+）> 防守成五（5,000,000）的分級體系，移除 `1.05` 係數。
- 修正 Minimax 單步剪枝，保留所有防守連五候選步。
