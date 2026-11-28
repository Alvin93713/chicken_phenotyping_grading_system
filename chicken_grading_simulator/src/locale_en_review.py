"""English UI text for product_review_ui.py, product_review.py and secure_sync.py.

Keys are the Chinese source strings (templates keep their {name} placeholders).
"""

EN = {
    # --- model table / quality dashboard -------------------------------------------------
    "模型": "Model",
    "檔案": "File",
    "版本雜湊": "Version hash",
    "大小 (MB)": "Size (MB)",
    "狀態": "Status",
    "已載入": "Loaded",
    "缺少檔案": "File missing",
    "模型品質監控": "Model Quality",
    "集中追蹤低信心、推論失敗與人工確認結果。":
        "Track low-confidence results, inference failures and review outcomes in one place.",
    "確認正確": "Confirmed correct",
    "確認錯誤": "Confirmed incorrect",
    "總案例": "Total cases",
    "目前模型": "Current models",
    "最近品質事件": "Recent quality events",
    "尚無品質事件。": "No quality events yet.",
    "優先度": "Priority",
    "原因": "Reason",
    "審核狀態": "Review status",
    "最低信心": "Lowest confidence",
    "建立時間": "Created",
    # stored case values (product_review.py), translated for display only
    "高": "High",
    "一般": "Normal",
    "推論失敗": "Inference failed",
    "缺少信心分數": "Confidence score missing",
    "例行品質抽樣": "Routine quality sample",
    "最低信心 {value}": "Lowest confidence {value}",
    "兩腳差異 {value}": "Legs differ by {value}",
    "{role} 執行格式：{runtime}（{file}）": "{role} runs on {runtime} ({file})",
    "已停用 TensorRT": "TensorRT disabled",
    "TensorRT 載入或推論失敗，已改用 PT": "TensorRT failed to load or run; using PT",
    "TensorRT 檔比 PT 舊，等待重建": "TensorRT engine is older than PT; waiting for rebuild",
    "comb 未偵測到有效遮罩": "No valid comb mask detected",
    # --- case review ---------------------------------------------------------------------
    "上一張：{chicken_id} · {role} → 判定為「{decision}」。選錯了可以復原。":
        "Previous image: {chicken_id} · {role} → marked \"{decision}\". Made a mistake? You can undo it.",
    "↩ 復原上一張": "↩ Undo last",
    "已復原：{chicken_id} · {role} 回到待審核，請重新判定。":
        "Undone: {chicken_id} · {role} is back in Pending. Please review it again.",
    "修改判定": "Change decision",
    "改為": "Change to",
    "退回待審核（重新判定）": "Back to Pending (review again)",
    "備註": "Note",
    "儲存修改": "Save change",
    "沒有變更。": "No changes.",
    "{chicken_id} · {role} 已退回待審核，可在「待審核」重新判定。":
        "{chicken_id} · {role} was sent back to Pending; review it again under \"Pending\".",
    "{chicken_id} · {role} 已由「{current}」改為「{choice}」。":
        "{chicken_id} · {role} changed from \"{current}\" to \"{choice}\".",
    "已無待審核影像。": "No images pending review.",
    "影像類型": "Image type",
    "信心": "Confidence",
    "剩餘待審核": "Pending left",
    "影像不存在": "Image not found",
    "選錯了？下一張出現後可按「↩ 復原上一張」；更早的判定可到其他分頁按「修改判定」。":
        "Wrong choice? Press \"↩ Undo last\" once the next image appears; for earlier decisions, "
        "use \"Change decision\" in the other tabs.",
    "目前沒有符合條件的案例影像。": "No case images match this filter.",
    "頁數": "Page",
    "第 {page} / {page_count} 頁": "Page {page} of {page_count}",
    "共 {count} 張 {role} 案例影像，每頁最多 {page_size} 張":
        "{count} {role} case images, up to {page_size} per page",
    "**判定：{state}**": "**Decision: {state}**",
    "信心：{value}": "Confidence: {value}",
    "案例審核": "Case Review",
    "COMB 與 SHANK 採獨立審核佇列。": "COMB and SHANK images are reviewed in separate queues.",
    "審核篩選": "Review filter",
    # --- maintenance: connection settings ------------------------------------------------
    "（未設定）": "(not set)",
    "連線設定（向廠商索取連線設定碼後貼上）": "Connection settings (paste the setup code from your vendor)",
    "注意：系統環境變數 {names} 已設定，會優先於此處儲存的設定。":
        "Note: the environment variables {names} are set and take priority over the settings saved here.",
    "貼上連線設定碼（建議）": "Paste setup code (recommended)",
    "手動輸入": "Enter manually",
    "連線設定碼": "Setup code",
    "CHKSYNC1:...（由廠商提供，請視同密碼保管）": "CHKSYNC1:... (from your vendor; keep it as safe as a password)",
    "儲存連線設定碼": "Save setup code",
    "已儲存連線設定。請按「測試連線」確認，之後即可按「回傳測試影像」。":
        "Connection settings saved. Press \"Test connection\" to check them, then you can press \"Send test images\".",
    "上傳網址（https://…）": "Upload URL (https://…)",
    "場域代號": "Site ID",
    "裝置代號": "Device ID",
    "API 金鑰": "API key",
    "簽章密鑰": "Signing secret",
    "上傳網址": "Upload URL",
    "CA 憑證檔路徑（選填，私有憑證才需要）": "CA certificate file path (optional, only for private certificates)",
    "儲存手動設定": "Save manual settings",
    "已儲存連線設定。請按「測試連線」確認。": "Connection settings saved. Press \"Test connection\" to check them.",
    "目前設定：{endpoint} · 場域 {site_id} · 裝置 {device_id} · API 金鑰 {api_key} · 簽章密鑰 {hmac_secret}":
        "Current settings: {endpoint} · site {site_id} · device {device_id} · "
        "API key {api_key} · signing secret {hmac_secret}",
    "清除已儲存的連線設定": "Clear saved connection settings",
    "已清除連線設定，之後無法透過網路回傳。": "Connection settings cleared; images can no longer be sent over the network.",
    # --- maintenance: upload progress ----------------------------------------------------
    "回傳暫時失敗：{reason}。下次自動重試：{next_attempt}":
        "Sending failed for now: {reason}. Next automatic retry: {next_attempt}",
    "（已連續失敗超過約 3 小時，請依原因排除問題）":
        " (failing for about 3 hours in a row; please fix the cause above)",
    "正在打包所有已審核影像…": "Packing all reviewed images…",
    "正在回傳：{sent} / {total}（{images} 張影像）": "Sending: {sent} / {total} ({images} images)",
    "回傳在背景進行，可以繼續量測與審核；斷線或重開機後會從中斷處接續。":
        "Sending runs in the background, so you can keep measuring and reviewing; "
        "after a network drop or restart it resumes where it stopped.",
    "最近一次回傳完成：{completed_at} · {images} 張影像（可直接訓練 {train_ready}、需重新標註 {to_relabel}） · {size}":
        "Last send completed: {completed_at} · {images} images "
        "({train_ready} ready for training, {to_relabel} to relabel) · {size}",
    "廠商端檔案：{name}": "Vendor file: {name}",
    "尚未回傳過。完成審核後，按下方按鈕把所有已審核影像回傳給廠商。":
        "Nothing sent yet. After reviewing, press the button below to send all reviewed images to the vendor.",
    "已開始打包並回傳 {count} 張已審核影像，進度顯示於上方。":
        "Started packing and sending {count} reviewed images; progress is shown above.",
    # --- maintenance: one-click upload panel ---------------------------------------------
    "方法一：一鍵安全回傳（建議）": "Option 1: One-click secure upload (recommended)",
    "已設定廠商連線：場域 {site_id} · 裝置 {device_id}": "Vendor connection set up: site {site_id} · device {device_id}",
    "尚未連線到廠商。": "Not connected to the vendor.",
    "**第一次設定（約 1 分鐘）：** ① 向廠商索取這台裝置的「連線設定碼」（CHKSYNC1: 開頭）"
    "　② 貼到下方「連線設定」並按「儲存連線設定碼」　③ 按「測試連線」出現「連線成功」即完成。":
        "**First-time setup (about 1 minute):** ① Ask your vendor for this device's setup code (starts with CHKSYNC1:)"
        " ② Paste it into \"Connection settings\" below and press \"Save setup code\""
        " ③ Press \"Test connection\"; when it says \"Connected\", you are done.",
    "技術細節：安全同步尚未啟用：{error}": "Technical details: secure sync is not enabled: {error}",
    "按下「回傳測試影像」，系統會把**當下所有已審核影像**（正確、不正確、無法判斷）"
    "連同審核結果與量測資訊打包成一份完整資料包，加密簽章後回傳。雞隻編號會改為無法還原的代碼。":
        "Press \"Send test images\" to pack **all images reviewed so far** (Correct, Incorrect, Unsure) "
        "together with the review results and measurements into one complete bundle, then encrypt, sign and send it. "
        "Chicken IDs are replaced with codes that cannot be traced back.",
    "回傳測試影像": "Send test images",
    "測試連線": "Test connection",
    "正在測試與廠商伺服器的連線...": "Testing the connection to the vendor server...",
    "立即重試": "Retry now",
    "已立即重新嘗試回傳。": "Retrying the upload now.",
    # --- maintenance page ----------------------------------------------------------------
    "資料與模型維護": "Data & Model Maintenance",
    "把所有已審核影像回傳給廠商，由廠商重新標註並微調模型。":
        "Send all reviewed images to the vendor, who relabels them and fine-tunes the models.",
    "可匯出正確影像": "Correct images to export",
    "可匯出不確定影像": "Unsure images to export",
    "可匯出錯誤影像": "Incorrect images to export",
    "待審核案例": "Cases pending review",
    "**回傳影像給廠商微調模型**　回傳內容＝所有已審核的 COMB 與 SHANK 影像（正確、無法判斷、不正確都要回傳）。"
    "「不正確」與「無法判斷」是廠商重新標註的重點；「正確」讓新模型保留原本做對的能力。"
    "先到「案例審核」完成審核，再用下列任一方法回傳。":
        "**Send images to the vendor to fine-tune the models**  What is sent: all reviewed COMB and SHANK images "
        "(Correct, Unsure and Incorrect are all sent). \"Incorrect\" and \"Unsure\" are what the vendor focuses on "
        "when relabeling; \"Correct\" helps the new model keep what it already gets right. "
        "Finish reviewing in \"Case Review\" first, then send using either option below.",
    "方法二：手動下載資料包（無網路時）": "Option 2: Download the bundle manually (no network)",
    "1. 按下方按鈕下載資料包（zip）。\n"
    "2. 透過**廠商指定的安全管道**交付：加密隨身碟，或廠商提供的安全上傳連結。\n"
    "3. 請勿用 Email、LINE 或公開雲端分享連結傳送，資料包含場域影像與雞隻編號。":
        "1. Press the button below to download the bundle (zip).\n"
        "2. Deliver it through a **secure channel chosen by the vendor**: an encrypted USB drive, "
        "or the secure upload link the vendor provides.\n"
        "3. Do not send it by email, LINE or public cloud share links; the bundle contains site images and chicken IDs.",
    "下載審核資料包": "Download review bundle",
    "（其中 {count} 張缺標註）": " ({count} of them without labels)",
    "包含所有已審核的 COMB 與 SHANK 影像：可直接訓練 {train_ready} 張，"
    "需人工重新標註 {to_relabel} 張{missing}。"
    "資料包依模型分成 comb／shank，各含 train_ready（正確，附標註）與 to_relabel（不正確、無法判斷，附模型預標註）。":
        "Contains all reviewed COMB and SHANK images: {train_ready} ready for training, "
        "{to_relabel} to relabel by hand{missing}. "
        "The bundle is split by model into comb / shank, each with train_ready (Correct, with labels) "
        "and to_relabel (Incorrect and Unsure, with model pre-labels).",
    "流程展示": "Workflow demo",
    "載入五筆真實測試案例": "Load five real test cases",
    "發布中的模型": "Models in production",
    "模型訓練與發布由廠商管理；雞舍端只負責確認案例。影像回傳於背景執行，不影響量測、推論與分級。":
        "The vendor manages model training and releases; the farm only reviews cases. "
        "Images are sent in the background without affecting measurement, inference or grading.",
    # --- secure_sync.py (messages shown in the UI) ---------------------------------------
    "請貼上廠商提供的連線設定碼": "Please paste the setup code from your vendor",
    "連線設定碼格式不正確，請確認是否完整複製（應以 CHKSYNC1: 開頭）":
        "The setup code is not valid. Check that you copied all of it (it should start with CHKSYNC1:)",
    "連線設定碼格式不正確": "The setup code is not valid",
    "連線設定碼缺少：{names}": "The setup code is missing: {names}",
    "尚未設定：{names}": "Not set: {names}",
    "CHICKEN_SYNC_ENDPOINT 必須是有效的 HTTPS 網址": "CHICKEN_SYNC_ENDPOINT must be a valid HTTPS URL",
    "CHICKEN_SYNC_HMAC_SECRET 至少需要 32 bytes": "CHICKEN_SYNC_HMAC_SECRET must be at least 32 bytes",
    "尚未連線到廠商：請貼上廠商提供的連線設定碼。": "Not connected to the vendor: please paste the setup code from your vendor.",
    "連線設定不完整，缺少：{names}。": "Connection settings are incomplete. Missing: {names}.",
    "上傳網址必須以 https:// 開頭，請向廠商確認。": "The upload URL must start with https://. Please check with your vendor.",
    "簽章密鑰長度不足，請向廠商重新索取連線設定碼。":
        "The signing secret is too short. Please ask your vendor for a new setup code.",
    "HTTPS 憑證驗證失敗：請確認上傳網址正確，或向廠商索取憑證設定":
        "HTTPS certificate check failed: check the upload URL, or ask your vendor for the certificate settings",
    "連線逾時：請檢查網路或稍後再試": "Connection timed out: check the network or try again later",
    "無法連線到廠商伺服器：請檢查網路連線與上傳網址":
        "Cannot reach the vendor server: check the network connection and the upload URL",
    "連線測試失敗：{error}": "Connection test failed: {error}",
    "連線成功：網址、憑證與金鑰皆正確，可以按「回傳測試影像」。":
        "Connected: the URL, certificate and keys are all correct. You can now press \"Send test images\".",
    "裝置時間與伺服器相差太多：請校正這台電腦的日期與時間":
        "This device's clock is too far off from the server: please correct this computer's date and time",
    "認證失敗：API 金鑰或簽章密鑰不正確，請向廠商重新索取連線設定碼":
        "Authentication failed: the API key or signing secret is wrong. Please ask your vendor for a new setup code",
    "找不到廠商的連線測試端點（404）：請向廠商確認上傳網址":
        "The vendor's connection test endpoint was not found (404): please check the upload URL with your vendor",
    "廠商伺服器回應異常（HTTP {status}）": "Unexpected response from the vendor server (HTTP {status})",
    "資料包超過廠商伺服器的大小上限：請聯絡廠商調整":
        "The bundle is larger than the vendor server allows: please contact your vendor",
    "這台裝置的硬碟空間不足，無法打包：請清理空間後重試":
        "Not enough disk space on this device to pack the bundle: free up space and try again",
    "連線逾時：網路不穩或伺服器忙碌，系統會自動重試":
        "Connection timed out: the network is unstable or the server is busy; it will retry automatically",
    "無法連線到廠商伺服器：請檢查網路，系統會自動重試":
        "Cannot reach the vendor server: check the network; it will retry automatically",
    "上傳網址錯誤（404），或廠商伺服器尚未支援整包回傳：請向廠商確認":
        "Wrong upload URL (404), or the vendor server does not support bundle uploads yet: please check with your vendor",
    "廠商伺服器暫時異常，系統會自動重試": "The vendor server has a temporary problem; it will retry automatically",
    "未知錯誤": "Unknown error",
    # model version control
    "模型版本": "Model versions",
    "使用中": "In use",
    "雞舍": "Farm",
    "更新日期": "Updated",
    "找不到任何模型檔": "No model files found",
    "使用中的版本": "Version in use",
    "自動（最新日期）": "Automatic (newest date)",
    "{role} 模型已切換為 {name}": "{role} model switched to {name}",
    "把目前的模型加入版本清單": "Add the current models to the version list",
    "已加入版本清單": "Added to the version list",
    "加入失敗：{error}": "Could not add: {error}",
    "匯入新模型": "Import a new model",
    "模型檔 (.pt)": "Model file (.pt)",
    "雞舍名稱": "Farm name",
    "模型更新日期": "Model update date",
    "匯入": "Import",
    "請選擇模型檔": "Choose a model file",
    "驗證並儲存模型中…": "Checking and saving the model…",
    "已匯入 {name}": "Imported {name}",
    "匯入失敗：{error}": "Import failed: {error}",
    "檔名：`best_<模型>_<雞舍名稱>_<日期>.pt`；舊版本保留，可隨時切回。": "File name: `best_<model>_<farm>_<date>.pt`. Older versions are kept so you can switch back any time.",
    "請輸入雞舍名稱": "Enter the farm name",
    "請先完成連線設定（場域代號），或用右側「匯入新模型」並填寫雞舍名稱": "Set up the connection (site ID) first, or use \"Import a new model\" and enter the farm name",
    # trimmed UI text
    '共 {count} 張': '{count} images',
    '連線設定': 'Connection settings',
    '已儲存，請按「測試連線」確認。': 'Saved. Press "Test connection" to check.',
    '尚未回傳過': 'Nothing sent yet',
    '開始回傳 {count} 張影像': 'Sending {count} images',
    '一鍵回傳': 'One-click upload',
    '場域 {site_id} · 裝置 {device_id}': 'Site {site_id} · Device {device_id}',
    '向廠商索取連線設定碼 → 貼到下方「連線設定」並儲存 → 按「測試連線」': 'Get a setup code from the vendor → paste it in "Connection settings" below and save → press "Test connection"',
    '手動下載資料包': 'Manual download',
    '含雞隻編號，請用加密隨身碟等安全管道交付，勿用 Email／LINE。': 'Contains chicken IDs: deliver via a secure channel such as an encrypted USB drive, not email or LINE.',
    '可直接訓練 {train_ready} 張 · 需重新標註 {to_relabel} 張{missing}': '{train_ready} ready for training · {to_relabel} to relabel{missing}',
}
