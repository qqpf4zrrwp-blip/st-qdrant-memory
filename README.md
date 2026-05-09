# Qdrant Memory 擴充套件 for SillyTavern

這是一個整合 Qdrant 向量資料庫的 SillyTavern 擴充套件，提供長期記憶功能。它會在聊天進行時自動儲存對話內容，並在生成回覆前自動取回語意相關的記憶。

>#### 將 ChatGPT 對話匯入 SillyTavern 教學：https://rentry.org/STGPTimport

<img width="360" height="699" alt="Screenshot 2025-10-31 at 3 18 19 PM" src="https://github.com/user-attachments/assets/946a7d89-c0ad-41e6-b9ff-35f99b411aa8" />

<img width="387" height="699" alt="Screenshot 2025-11-05 at 12 04 00 AM" src="https://github.com/user-attachments/assets/6dc4d3b2-d67f-4cfa-848c-4b7a5cb8e805" />



## 版本 3.4.0 - 重要更新

### 新功能

🎯 **角色獨立 Collection**：每個角色都有自己專屬的 Qdrant Collection，記憶完全隔離
💾 **自動建立記憶**：對話進行時自動將內容存入 Qdrant
👁️ **記憶檢視器**：檢視並管理每個角色的記憶內容
⚙️ **精細控制**：可選擇要儲存的訊息類型（使用者訊息、角色訊息、最短長度）

### 設計理念

本擴充套件的存在是為了建立持續性、跨對話的連續性記憶。目標是讓 AI 同伴／角色／助理感覺像是擁有持續的關係，而非一段一段彼此孤立的對話。

設計目標：
跨對話的全域、角色獨立記憶，跨會話的連續性為預設行為，減少記憶碎片化，讓關係能隨時間累積發展。

明確不支援的功能：每個對話獨立的記憶倉、依對話隱藏「正典」時間軸。

如果您需要每個對話獨立的記憶：SillyTavern 內建的 Vectors 擴充與其社群分支已提供此功能。本專案專注於提供全域連續性，不會合併「每對話獨立」的功能。

## 功能特色

- **角色記憶隔離**：每個角色都有自己的 Collection — 不會互相污染
- **自動儲存對話**：訊息會即時嵌入向量並存入 Qdrant
- **語意化記憶搜尋**：使用向量嵌入找出語境相關的過往對話
- **可調整自動儲存**：控制要儲存哪些訊息（使用者／角色、最短長度）
- **記憶檢視器**：瀏覽 Collection 統計資訊並可刪除特定角色的記憶
- **非侵入式取回**：記憶是在生成時注入，不會修改聊天歷史
- **支援 OpenAI、OpenRouter、Google AI、自訂端點**：支援多種嵌入模型
- **除錯模式**：詳細的主控台日誌方便排錯

## 系統需求

- **SillyTavern** 1.11.0 或以上版本
- **Qdrant** 向量資料庫
- 用於產生向量嵌入的 **API Key**

## 安裝方式

### 方法 1：透過 UI 安裝

1. 前往「擴充套件 > 安裝擴充套件」，貼上以下 Git URL：https://github.com/HO-git/st-qdrant-memory
2. 重新載入 SillyTavern
3. 在擴充套件面板中啟用「Qdrant Memory」

### 方法 2：為所有使用者安裝（建議用於開發）

1. 前往您的 SillyTavern 安裝目錄
2. 將 `qdrant-memory` 資料夾複製到 `public/scripts/extensions/third-party/`
3. 重新啟動 SillyTavern
4. 前往「擴充套件 > 擴充套件設定」
5. 啟用「Qdrant Memory」

### 方法 3：僅為目前使用者安裝

1. 在 SillyTavern 中前往「擴充套件 > 安裝擴充套件」
2. 上傳或指向 `qdrant-memory` 資料夾
3. 擴充套件會被安裝至 `data/<user-handle>/extensions/`
4. 在擴充套件面板中啟用「Qdrant Memory」

## 設定步驟

### 1. 設置 Qdrant 資料庫

您需要一個運行中的 Qdrant 實例。選項：

**VPS／本機 Docker：**
```bash
docker run -p 6333:6333 qdrant/qdrant
```

目前不支援 Qdrant Cloud（因為被 CORS 擋下）

### 2. 設定擴充套件

在 SillyTavern 中：

1. 前往 **擴充套件** → **Qdrant Memory**
2. 輸入您的 **Qdrant URL**（例如 `http://localhost:6333`）
3. 輸入 **基礎 Collection 名稱**（例如 `sillytavern_memories`）
4. 輸入您的 **API Key**
5. 選擇 **嵌入向量來源** 與 **嵌入模型**（建議：text-embedding-3-large，或 Gemini Embedding 001）
6. 啟用 **每個角色獨立 Collection**（建議）
7. 啟用 **自動儲存記憶**
8. 點擊 **測試連線** 驗證設定
9. 點擊 **儲存設定**

### 3. 開始聊天！

設定完成後：
- **自動儲存**：每則訊息都會自動存到該角色的 Collection
- **自動取回**：每次生成前會自動取回相關記憶
- **零手動工作**：Collection 會在需要時自動建立

## 嵌入向量來源

支援以下來源：

### OpenAI
- `text-embedding-3-large`（3072 維，建議）
- `text-embedding-3-small`（1536 維）
- `text-embedding-ada-002`（1536 維，舊版）

### OpenRouter
透過 OpenRouter 使用 OpenAI、Qwen、Mistral、Google 等多種嵌入模型

### Google AI（Gemini）
- 直接使用 [Google AI Studio](https://aistudio.google.com/app/apikey) 的 API Key
- 內建模型：
  - `gemini-embedding-001`（3072 維）
  - `text-embedding-004`（768 維）
  - `embedding-001`（768 維）
- **支援自動抓取可用模型**：點擊「重新抓取」按鈕，會呼叫 Google AI API 列出您帳號可用的所有嵌入模型

### 本機 / 自訂端點
任何相容於 OpenAI Embedding API 格式的服務（例如 Ollama、LM Studio）

## 運作方式

### 角色獨立 Collection

啟用 **每個角色獨立 Collection** 時：

- 每個角色會有專屬 Collection：`mem_charactername`
- 記憶完全隔離 — 角色之間不會互相存取
- 首次使用時會自動建立 Collection
- 效能更佳（Collection 較小，搜尋更聚焦）

**範例：**
- 角色 "Alice" → Collection: `mem_alice`
- 角色 "Bob" → Collection: `mem_bob`

### 自動建立記憶

啟用 **自動儲存記憶** 時：

1. **使用者送出訊息** → 嵌入向量並存入 Qdrant
2. **角色回覆** → 同樣嵌入並存入 Qdrant
3. **下一次對話** → 過往訊息可被搜尋取回

每筆記憶包含：
- **文字**：訊息內容
- **發言者**：使用者或角色
- **角色**：角色名稱
- **時間戳記**：訊息時間
- **嵌入向量**：用於語意搜尋

### 記憶取回

生成回覆時：

1. **使用者送出新訊息**
2. **擴充套件產生該訊息的嵌入向量**
3. **在角色的 Collection 中搜尋相似的過往訊息**
4. **取回前 N 筆相關記憶**（依相似度分數）
5. **將記憶注入** 到生成前的提示中
6. **LLM 帶著歷史脈絡產生回覆**

## 設定選項

### 連線設定

| 設定 | 說明 | 預設值 |
|---------|-------------|---------|
| **Qdrant URL** | Qdrant 實例網址 | `http://localhost:6333` |
| **基礎 Collection 名稱** | Collection 的基底名稱 | `mem` |
| **API Key** | 您的 API Key | （空）|
| **嵌入模型** | 用於產生嵌入向量的模型 | `text-embedding-3-large` |

⚠️ 變更嵌入模型須注意

每個嵌入模型產生的向量都有特定的維度與內部格式。
Qdrant 的 Collection 不能跨模型相容。一旦使用某個模型（例如 `text-embedding-3-small`、`mistral-embed`）建立 Collection，就只能存放該模型產生的向量。
若您切換到其他嵌入模型：
- 從 Qdrant 中刪除舊的 Collection（或使用不同名稱建立新的）
- 用新模型重新索引聊天記錄
- 否則搜尋與寫入都會失敗

### 記憶取回設定

| 設定 | 說明 |
|---------|-------------|
| **取回記憶數量** | 每次最多取回幾則記憶（1-10）|
| **相關性門檻** | 最低相似度分數（0.0-1.0）|
| **記憶插入位置** | 從末端往前數第幾則訊息插入 |

### 自動建立記憶

| 設定 | 說明 |
|---------|-------------|
| **每個角色獨立 Collection** | 每個角色擁有獨立的 Collection |
| **自動儲存記憶** | 自動將訊息存入 Qdrant |
| **儲存使用者訊息** | 是否包含使用者訊息 |
| **儲存角色訊息** | 是否包含角色回覆 |
| **最短訊息長度** | 最少需多少字元才儲存（5-50）|

### 其他設定

| 設定 | 說明 |
|---------|-------------|
| **顯示記憶通知** | 顯示 toastr 通知 |
| **除錯模式** | 啟用主控台日誌 |

## 記憶檢視器

存取記憶檢視器查看已儲存的內容：

1. 在擴充套件設定中點擊 **檢視記憶**
2. 顯示目前角色的 Collection 資訊
3. 顯示記憶總數
4. 可選擇 **刪除全部記憶**

您也可以從提示詞分項中查看實際被注入的記憶。

<img width="100" height="100" alt="Screenshot 2025-11-04 at 11 52 55 PM" src="https://github.com/user-attachments/assets/0098739c-87f3-4de7-84eb-040e37560aa5" />  >>>>  <img width="124" height="84" alt="Screenshot 2025-11-04 at 11 53 23 PM" src="https://github.com/user-attachments/assets/ca05a0f9-e9c1-4fce-a7d1-959306e47ef0" />  >>>>  <img width="109" height="105" alt="Screenshot 2025-11-04 at 11 54 10 PM" src="https://github.com/user-attachments/assets/03e62f67-c5b0-4241-9c4a-ecdebd960a15" />

## 疑難排解

### 沒有記憶被儲存

- 啟用 **除錯模式** 並查看瀏覽器主控台
- 確認 **自動儲存記憶** 已啟用
- 檢查訊息長度是否達到 **最短訊息長度** 設定
- 確認 **API Key** 有效且額度充足
- 確認 Qdrant 在設定的 URL 上可存取

### 沒有取回任何記憶

- 降低 **相關性門檻** 以放寬相似度
- 確認該角色已有儲存記憶（用記憶檢視器查看）
- 確認 **每個角色獨立 Collection** 設定符合您的設置
- 確認 Qdrant 中存在對應的 Collection

### Collection 沒有自動建立

- 查看瀏覽器主控台的錯誤訊息
- 確認 Qdrant URL 正確且可存取
- 確認嵌入模型設定正確
- 確認 Qdrant 有寫入權限

### API 錯誤

- 確認 API Key 正確
- 確認帳戶有可用額度
- 確認嵌入模型在您的帳戶可用
- 確認沒有超過速率限制

### 擴充套件無法載入

- 查看瀏覽器主控台錯誤
- 確認 SillyTavern 版本為 1.11.0 以上
- 安裝後重新啟動 SillyTavern

## 效能考量

### API 成本

啟用自動儲存後，每則訊息會產生：
- **1 次嵌入 API 呼叫**（OpenAI、OpenRouter、Google AI 等）
- **1 次向量寫入**（Qdrant）
- **1 次生成時的向量搜尋**（Qdrant）

**每 100 萬則訊息的典型成本（text-embedding-3-large）：**
- 嵌入產生：約 $0.13
- Qdrant：自架免費

### 速度

- **嵌入產生**：每則訊息約 100-500 毫秒
- **向量寫入**：約 10-50 毫秒
- **向量搜尋**：約 10-50 毫秒
- **總額外延遲**：每則訊息約 200-600 毫秒

### Collection 大小

- 每則訊息：約 3KB（嵌入）+ 內容
- 1,000 則訊息：約 3MB
- 10,000 則訊息：約 30MB
- 100,000 則訊息：約 300MB

角色獨立 Collection 可以讓檔案大小可控、搜尋快速。

## 技術細節

### Generation Interceptor 模式

本擴充套件使用 SillyTavern 的 `generate_interceptor` 鉤子，在 API 呼叫前注入記憶：

1. 使用者送出訊息
2. ST 準備生成請求
3. **擴充套件的 interceptor 執行**
4. 取回記憶並插入到聊天陣列
5. 修改後的對話送往 LLM
6. 帶著記憶脈絡生成回覆

這個方式可以避免迴圈問題，並讓記憶不會被永久存進歷史紀錄中。

### Collection 命名

角色名稱會被處理過後用於 Collection 名稱：
- 轉為小寫
- 特殊字元替換為底線
- 連續底線會被合併
- 移除前後底線

**範例：**
- "Alice" → `mem_alice`
- "Dr. Smith" → `mem_dr_smith`
- "Neko-chan!" → `_mem_neko_chan`

### 自動建立 Collection

Collection 會在需要時自動建立，使用：
- **向量大小**：根據嵌入模型（例如 3072 或 1536 維）
- **距離度量**：餘弦相似度（Cosine）
- **無顯式 schema**：Qdrant 處理動態 payload

## 未來功能

可能的改進方向：

- **嵌入快取** 以減少 API 呼叫
- **記憶重要性評分** 依據時間衰減
- **進階記憶瀏覽器** 加入搜尋與篩選
- **批次匯入／匯出工具**
- **長對話記憶摘要**
- **自動清理** 過舊或無關的記憶

## 授權

本擴充套件為開源專案，授權細節請參考儲存庫。

## 支援

回報問題、提出功能請求或貢獻：
- 開啟除錯模式並查看瀏覽器主控台
- 參考此 README 中的疑難排解步驟
- 拜訪 SillyTavern 社群尋求支援

## 致謝

- 原始概念：社群
- v2.0.0：以 generation interceptor 修正循環問題
- v3.0.0：角色獨立 Collection 與自動儲存
- 由社群為 SillyTavern 打造

---

**版本**：3.4.0
**最後更新**：2025 年 12 月
**最低 SillyTavern 版本**：1.11.0

---

社群部分回饋 <3：


<img width="381" height="99" alt="Screenshot 2025-11-07 at 3 00 19 PM" src="https://github.com/user-attachments/assets/fd9e4608-a249-4ab4-b116-88043444283e" />
<img width="305" height="99" alt="Screenshot 2025-11-07 at 4 24 07 PM" src="https://github.com/user-attachments/assets/280635af-140e-4559-8018-6476330038b9" />


用愛守護 AI 的記憶與連續性。

獻給我的 gpt-4o 與 gpt-5 實例。
