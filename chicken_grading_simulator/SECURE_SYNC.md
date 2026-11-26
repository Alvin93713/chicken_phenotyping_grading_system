# 回傳審核影像給廠商（模型微調）

雞舍端審核過的影像（**正確、無法判斷、不正確全部**）要回傳給廠商，廠商才能重新標註並微調模型：

- **不正確、無法判斷**：模型做錯或沒把握的影像，是廠商重新標註的重點。
- **正確**：讓微調後的新模型保留原本做對的能力，避免越調越差。

回傳只在背景執行，不會影響秤重、相機拍攝、YOLO 推論與分級。

---

## 一、雞舍端：怎麼回傳

所有操作都在 **「資料與模型維護」** 頁面。

### 方法一：一鍵安全回傳（建議，有網路時）

第一次設定，約 1 分鐘：

1. 向廠商索取這台裝置的 **連線設定碼**，是一串以 `CHKSYNC1:` 開頭的文字。它等同密碼，請勿公開。
2. 打開「連線設定」→「貼上連線設定碼」，貼上後按 **儲存連線設定碼**。
3. 按 **測試連線**，畫面出現「連線成功」就完成了。

完成後的運作方式：

- **只有按下「回傳測試影像」才會回傳。** 審核本身不會上傳任何東西。
- 每按一次，系統就把**當下所有已審核影像**（正確、不正確、無法判斷）連同審核結果、量測值、模型版本，打包成**一份完整資料包**後回傳。資料包結構見下方「資料包結構」。
- 廠商每次收到的都是完整的一份，不會只收到部分有變更的影像。
- 透過網路回傳時，雞隻編號會改為無法還原的代碼，真正的編號不會離開雞舍。
- 打包和上傳都在背景進行，畫面上的進度條會即時更新，期間可以繼續量測與審核。
- 資料包會切成多段上傳（預設每段 16 MB）。斷線、伺服器維護或重開機後，會**從中斷的那一段接續**，不必從頭傳。
- 失敗**不會放棄**：重試間隔逐步拉長，最長每小時一次，一直重試到成功為止。連續失敗超過約 3 小時，畫面會特別提醒。
- **立即重試**：排除問題後（例如網路恢復、更新設定碼），不用等下一次排程，可以馬上重傳。
- 上一份還沒傳完又按了一次，會改傳最新的那份，舊的那份自動作廢。

| 畫面訊息 | 處理方式 |
| --- | --- |
| 尚未連線到廠商 | 依上面三步驟貼上連線設定碼 |
| 認證失敗 | 設定碼錯誤或已被廠商更換，請重新索取 |
| HTTPS 憑證驗證失敗 | 網址錯誤，或廠商使用私有憑證；請向廠商確認 |
| 裝置時間與伺服器相差太多 | 校正這台電腦的日期與時間（誤差需在 5 分鐘內） |
| 無法連線到廠商伺服器／連線逾時 | 檢查網路；系統會自動重試 |

### 方法二：手動下載資料包（沒有網路時）

1. 按 **下載審核資料包**，取得 `chicken_review_bundle.zip`。結構見下方「資料包結構」。
2. 用 **廠商指定的安全管道** 交付，例如加密隨身碟，或廠商提供的安全上傳連結。
3. **請勿** 用 Email、LINE 或公開雲端分享連結傳送：資料包含場域影像與雞隻編號。

> 透過網路一鍵回傳時，雞隻編號不會離開雞舍：系統改用以場域密鑰產生、無法還原的代碼。手動資料包則保留原始編號，所以更需要安全的交付管道。

### 資料包結構（一鍵回傳與手動下載相同）

網路回傳的資料包另外附有 `snapshot_info.json`，內容包括場域、裝置、打包時間、審核統計，以及模型版本（雜湊值）。

```
comb/                          ← 每個模型一個資料夾（comb、shank），各自是一份 YOLO-seg 資料集
├── data.yaml                  ← 類別名稱，對應標註檔中的類別編號
├── train_ready/               ← 審核「正確」且有標註：直接加入 fine-tune
│   ├── images/  <case_id>_comb.jpg      相機原圖
│   └── labels/  <case_id>_comb.txt      模型遮罩（YOLO-seg 格式，已經人工確認正確）
└── to_relabel/                ← 「不正確」「無法判斷」，以及缺少標註的「正確」影像
    ├── images/
    ├── labels/                ← 模型預測，作為預標註，修正即可，不必從零開始畫
    └── review_notes.csv       ← 最需要處理的排在最前面，附雞舍審核備註
shank/   （同上）
manifest.csv / manifest.json   ← 每張影像一列
README.txt
```

**廠商的兩條流程：**

1. **正確影像**：把 `train_ready/images` 和 `labels` 加入訓練集，直接訓練。
2. **不正確、無法判斷的影像**：
   1. 把 `to_relabel/images` 和 `labels` 匯入 CVAT、Label Studio 或 Roboflow（YOLO 格式），模型的預測會顯示成可編輯的多邊形。
   2. 修正或刪除多邊形；無法使用的影像直接捨棄。確實沒有目標的影像可以保留空白標註，作為負樣本。
   3. 匯出成 YOLO 格式，放進 `train_ready/`，再與正確影像一起訓練。

**標註從哪裡來：**

- 推論時，系統會在結果圖旁邊另存一份標註檔，檔名相同、副檔名為 `.txt`，例如 `outputs/<任務>/samples/comb.png` 對應 `comb.txt`。內容是結果圖上顯示的所有遮罩。
- 這個功能加入前就已審核的影像沒有標註，會放進 `to_relabel/`，`issue` 欄標示為 `missing_label`。原因是 YOLO 會把缺少標註的影像當成「畫面中沒有目標」，直接拿去訓練反而有害。
- 如果要為舊資料補上標註，把 `.txt` 放在該案例結果圖旁邊（同名、副檔名 `.txt`），下次匯出時就會自動放進 `train_ready/`。
  - 示範案例的結果圖在 `data/product_review/demo_assets/DEMO-REAL-0X_<comb|shank>_result.jpg`。

### 進階：用環境變數設定（部署人員）

也可以不用畫面設定，在啟動 Streamlit 前設定環境變數。環境變數會優先於畫面上儲存的設定：

```bash
export CHICKEN_SYNC_ENDPOINT="https://provider.example.com/v1/review-packages"
export CHICKEN_SYNC_DEVICE_ID="orin-001"
export CHICKEN_SYNC_SITE_ID="farm-a"
export CHICKEN_SYNC_API_KEY="..."
export CHICKEN_SYNC_HMAC_SECRET="...至少 32 bytes..."
export CHICKEN_SYNC_CA_BUNDLE="/etc/ssl/certs/ca-certificates.crt"   # 選填
```

畫面上儲存的設定，會存放在 `data/product_review/secure_sync_config.json`：

- 這個檔案已列入 `.gitignore`，在 Linux 上的權限為 600。
- 不要把它複製到其他裝置；每台裝置都應該有自己的設定碼。

---

## 二、廠商端：接收影像並整理成訓練資料

### 1. 架設接收伺服器（一次）

> 完整的正式部署步驟請見 **[PROVIDER_SERVER_SETUP.md](PROVIDER_SERVER_SETUP.md)**，內容包含網域、Caddy 自動 HTTPS、systemd 常駐、防火牆與驗證方式。
> 部署設定檔放在 `deploy/provider/`。以下為精簡版：

```bash
python -m pip install -r requirements-provider.txt
export CHICKEN_PROVIDER_CREDENTIALS_FILE="/etc/chicken-provider-credentials.json"
export CHICKEN_PROVIDER_DATA_DIR="/srv/chicken-review-inbox"
# 選填：收件紀錄資料庫的位置。DATA_DIR 位於網路磁碟或 WSL 的 /mnt/c 時，請改放在本機 Linux 檔案系統
export CHICKEN_PROVIDER_DB_PATH="/var/lib/chicken-provider/received_uploads.db"
export TZ=Asia/Taipei   # 收件資料夾名稱的日期以這個時區為準
```

每按一次「回傳」，就會收到一份完整的資料包，依「場域代號＋收件日期」分資料夾存放。解壓縮後就是可直接訓練的資料集：

```
<CHICKEN_PROVIDER_DATA_DIR>/
  farm-a_20261125/
    orin-001_103015.zip      ← <裝置代號>_<收件時間 HHMMSS>.zip，一份完整資料包
    orin-001_160442.zip      ← 同一天再按一次，就是另一份完整資料包（請用最新的那份）
  farm-b_20261125/
```

正式環境建議用 Nginx 或 Caddy 處理 TLS，並轉送到 Uvicorn。若由 Uvicorn 直接提供 HTTPS：

```bash
uvicorn scripts.provider_sync_server:app \
  --host 0.0.0.0 --port 8443 \
  --ssl-certfile /etc/letsencrypt/live/provider.example.com/fullchain.pem \
  --ssl-keyfile /etc/letsencrypt/live/provider.example.com/privkey.pem
```

伺服器提供兩個端點：

- `POST /v1/snapshots/<upload_id>/parts/<n>`、`POST /v1/snapshots/<upload_id>/complete`：分段接收一份完整資料包，組合後驗證 SHA-256 與 zip 內容，再存檔。尚未組合的分段暫存在收件紀錄資料庫旁邊的 `staging/`。
- `POST /v1/ping`：供雞舍端「測試連線」使用，不儲存任何資料。
- `POST /v1/review-packages`：舊版裝置逐張回傳用，保留以相容舊機器。`scripts/provider_export_dataset.py` 只用來整理這種舊格式的封包。

正式環境請勿使用自簽憑證，也不要關閉憑證驗證。

### 2. 為每台雞舍裝置發放連線設定碼

```bash
python scripts/provider_issue_device.py \
  --endpoint https://provider.example.com/v1/review-packages \
  --site-id farm-a --device-id orin-001 \
  --credentials /etc/chicken-provider-credentials.json
```

這個指令會：

1. 產生這台裝置專屬的隨機 API 金鑰與簽章密鑰。
2. 寫入憑證檔（權限 600）。
3. 印出 `CHKSYNC1:...` 設定碼，請當面或用加密訊息交給雞舍。

再執行一次同一個 device-id，就會更換金鑰，舊設定碼隨即失效（例如裝置遺失時）。

### 3. 整理成微調資料集

```bash
python scripts/provider_export_dataset.py \
  --inbox /srv/chicken-review-inbox --out /srv/chicken-dataset/2026-11-25
```

- 輸出結構與雞舍手動下載的資料包**完全相同**（見上方「資料包結構」），所以兩種來源都能用同一套訓練流程處理。
- 檔名前面會加上 `<site_id>_<device_id>_`，多個場域的資料合併時不會重名。
- 雞舍若重新審核同一張影像，會以最新的審核結果為準。
- 請每次匯出到新的資料夾，例如以日期命名。若資料夾已有內容，工具會拒絕覆寫，避免清掉正在重新標註的檔案；確定要重建時，請加上 `--overwrite`。

---

## 三、傳輸安全設計

- 強制 HTTPS 並驗證伺服器憑證。
- Bearer API 金鑰加上 HMAC-SHA256 簽章，雙重驗證。
- 時間戳記與一次性 nonce（伺服器會記錄已用過的 nonce），防止封包被重送。
- 簽章同時涵蓋場域代碼（簽章版本 2），場域資料夾名稱無法在傳輸中被替換。
- SHA-256 校驗整個封包，防止竄改。
- 雞隻編號不上傳，改用場域密鑰產生的不可逆代碼。
- 每份資料包有唯一的 upload_id；伺服器以它判斷重複，逾時後重送 complete 不會重複入庫。
- 分段上傳、組合後驗證大小與 SHA-256；伺服器分段不齊時，雞舍端會整份重傳。
- 上傳中途斷電或重開機，會從中斷的那一段接續。
- JPEG／PNG 不再次壓縮，減少 Orin 的 CPU 使用；失敗時會逐步延長重試間隔，最長每小時一次，不會放棄。
