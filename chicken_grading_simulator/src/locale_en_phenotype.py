"""English UI text for the phenotype measurement / dual-camera pages (dual_camera_ui.py,
dual_camera.py, inference_queue.py messages, scale_control.py, qr_utils.py).
Keys are the Chinese source strings."""

EN = {
    # camera labels / settings
    "Comb 雞冠": "Comb",
    "Shank 腳脛": "Shank",
    "{label} camera ID": "{label} camera ID",
    "{label} cm/pixel": "{label} cm/pixel",
    "{label} 比例尺長度 (cm)": "{label} scale length (cm)",
    "兩個表型必須使用不同相機 ID": "Comb and Shank must use different camera IDs",
    "未校正": "Not calibrated",
    # adjustment / calibration pages
    "雙相機調整": "Dual camera adjustment",
    "雞冠與腳脛分別使用獨立相機。": "Comb and shank each use their own camera.",
    "雙相機比例尺校正": "Dual camera scale calibration",
    "分別設定雞冠與腳脛相機的比例尺。": "Set the scale for the comb and shank cameras separately.",
    "拍攝雙相機校正照片": "Capture calibration photos",
    "確定雙相機校正": "Apply calibration",
    "拍攝後已更改相機，請重新拍攝校正照片": "The camera changed after capture. Please capture the calibration photos again.",
    "{role} 兩個校正點不可重疊": "{role}: the two calibration points must not overlap",
    "雙相機校正已套用": "Dual camera calibration applied",
    "已選取 {count}/2 個校正點": "{count}/2 calibration points selected",
    "清除校正點": "Clear points",
    # phenotype page
    "雙相機表型量測": "Dual camera phenotype measurement",
    "Comb 與 Shank 各拍一張照片，合併量測與分級結果。":
        "Take one Comb photo and one Shank photo to measure and grade together.",
    "使用照片測試模式（不開啟雙相機）": "Photo test mode (cameras stay off)",
    "YOLO confidence 門檻": "YOLO confidence threshold",
    "照片模式不會開啟 Camera 0/1。cm/pixel 會直接影響面積、寬度與長度結果。":
        "Photo mode does not open Camera 0/1. cm/pixel directly affects the area, width and length results.",
    "上傳 {label} 照片": "Upload {label} photo",
    "即時 YOLO segmentation 預覽（背景推論目標 15 FPS；正式量測仍使用原始單張照片）":
        "Live YOLO segmentation preview (background inference targets 15 FPS; measurements still use the original single photo)",
    "找不到 {role} 模型：{path}": "{role} model not found: {path}",
    "**Chicken ID：{chicken_id}**": "**Chicken ID: {chicken_id}**",
    "手動重量備援 (g)": "Manual backup weight (g)",
    "計算雞冠及腳脛數據": "Measure comb and shank",
    "請先上傳雞冠與腳脛兩張照片": "Please upload both the comb and shank photos first",
    "請先完成兩台相機的比例尺校正": "Please calibrate the scale for both cameras first",
    "正在收集 2 秒平均重量，完成後將停止秤重並拍攝雙相機照片...":
        "Collecting a 2-second average weight, then weighing stops and both cameras take photos...",
    "未取得有效秤重值，使用手動重量備援": "No valid scale reading; using the manual backup weight",
    "最近建立任務：{job_id}": "Latest job: {job_id}",
    "請上傳雞冠與腳脛兩張照片，並設定兩組 cm/pixel": "Upload both comb and shank photos and set both cm/pixel values",
    "照片模式的 cm/pixel 必須大於 0": "In photo mode, cm/pixel must be greater than 0",
    "無法讀取上傳的照片，請使用 JPG、PNG 或 BMP": "Cannot read the uploaded photo. Please use JPG, PNG or BMP",
    # snapshot results
    "**目前任務：{chicken_id} · {status} · {message}**": "**Current job: {chicken_id} · {status} · {message}**",
    "**最近完成：{chicken_id} · {grade}**": "**Latest result: {chicken_id} · {grade}**",
    "已鎖定重量 (g)": "Locked weight (g)",
    "重量 (g)": "Weight (g)",
    "雞冠面積 (cm²)": "Comb area (cm²)",
    "腳脛寬度 (cm)": "Shank width (cm)",
    "腳脛長度 (cm)": "Shank length (cm)",
    # job table columns (inference_queue.list_jobs keys)
    "狀態": "Status",
    "進度": "Progress",
    "訊息": "Message",
    "體重(g)": "Weight (g)",
    "雞冠面積(cm²)": "Comb area (cm²)",
    "腳脛寬度(cm)": "Shank width (cm)",
    "腳脛長度(cm)": "Shank length (cm)",
    "初步分級": "Preliminary grade",
    "初步燈號": "Preliminary light",
    "錯誤": "Error",
    "更新時間": "Updated",
    "影片": "Video",
    "建立時間": "Created",
    # job messages written in Chinese by the background worker (translated at display)
    "等待背景推論": "Waiting for background inference",
    "正在推論雙相機照片": "Inferring dual camera photos",
    "正在推論 {role} 單張照片": "Inferring {role} photo",
    "兩張照片推論完成": "Both photos inferred",
    "雙相機推論完成，結果已寫入資料庫": "Dual camera inference done; results saved to the database",
    "正在錄製 3 秒影片": "Recording a 3-second video",
    "正在載入 YOLO segmentation 模型": "Loading YOLO segmentation model",
    "推論完成，結果已寫入資料庫並完成初步篩選": "Inference done; results saved and preliminary screening completed",
    "推論完成，結果已寫入資料庫；未執行初步亮燈": "Inference done; results saved; preliminary light not run",
    "任務失敗": "Job failed",
    # dual_camera.py (script-thread errors)
    "相機啟動中": "Camera starting",
    "無法開啟相機 {index}": "Cannot open camera {index}",
    "相機 {index} 無法讀取畫面": "Camera {index} cannot read frames",
    "相機 {index} 畫面已過期": "Camera {index} frame is stale",
    "Comb 與 Shank 必須選擇不同相機": "Comb and Shank must use different cameras",
    "等待兩台相機的新畫面": "Waiting for new frames from both cameras",
    "雙相機拍攝失敗：{error}；未建立推論任務": "Dual camera capture failed: {error}; no inference job was created",
    "無法啟動相機預覽串流 port {port}：{error}": "Cannot start the camera preview stream on port {port}: {error}",
    "無法儲存 {role} 照片": "Cannot save the {role} photo",
    "相機尚未校正，或相機已更換，請重新校正": "Camera not calibrated or changed. Please calibrate again",
    "相機解析度與校正照片不同，請重新校正": "Camera resolution differs from the calibration photo. Please calibrate again",
    "校正比例必須為正數": "The calibration scale must be positive",
    # scale_control.py
    "非 ASCII 封包：{payload}": "Non-ASCII packet: {payload}",
    "秤重設備回傳異常碼 9999": "Scale returned error code 9999",
    "秤重封包不是純數字：{payload}": "Scale packet is not numeric: {payload}",
    "請先選擇秤重設備 serial port。": "Please select the scale serial port first.",
    # qr_utils.py
    "chicken_id 不可空白，無法產生 QR code。": "chicken_id cannot be empty; cannot generate a QR code.",
    "OpenCV 解析器不可用：{error}": "OpenCV decoder unavailable: {error}",
    "OpenCV 無法讀取此圖片格式。": "OpenCV cannot read this image format.",
    "OpenCV 解析失敗：{error}": "OpenCV decoding failed: {error}",
    "OpenCV 未在圖片中偵測到 QR code。": "OpenCV found no QR code in the image.",
    "pyzbar 解析器不可用：{error}": "pyzbar decoder unavailable: {error}",
    "pyzbar 解析失敗：{error}": "pyzbar decoding failed: {error}",
    "pyzbar 未在圖片中偵測到 QR code。": "pyzbar found no QR code in the image.",
    "QR code 解析失敗。請確認圖片清晰且 QR code 完整，或改用手動輸入 chicken_id。"
    "\nOpenCV：{opencv_error}"
    "\npyzbar：{pyzbar_error}":
        "Could not read the QR code. Make sure the image is sharp and the whole QR code is visible, "
        "or enter the chicken_id manually."
        "\nOpenCV: {opencv_error}"
        "\npyzbar: {pyzbar_error}",
    "已鎖定體重 (g)": "Locked weight (g)",
    "鎖定體重": "Weight locked",
    # measurement progress ring
    "計算中": "Measuring",
    "計算進度": "Measurement progress",
    "鎖定重量": "Weight locked",
    "拍攝雙相機": "Capture both cameras",
    "推論雞冠面積": "Comb area inference",
    "推論腳脛長寬": "Shank size inference",
    "寫入並分級": "Save and grade",
    # weight flow (retry instead of silent manual fallback)
    "手動重量 (g)": "Manual weight (g)",
    "請輸入體重": "Enter weight",
    # calibrate each camera on its own
    "拍攝 {label} 校正照片": "Take {label} calibration photo",
    "確定 {label} 校正": "Confirm {label} calibration",
    "{label} 校正已套用": "{label} calibration applied",
    "雞冠與腳脛相機可以分別校正，各自拍照、點兩點、確定即可。": "Calibrate the comb and shank cameras separately: take a photo, pick two points, confirm.",
    "等待相機的新畫面": "waiting for a new camera frame",
    "相機 {index} 拍攝失敗：{error}": "Camera {index} capture failed: {error}",
    # calibration history
    "校正歷史": "Calibration history",
    "尚無校正紀錄": "No calibrations recorded yet",
    "校正時間": "Calibrated at",
    "量尺長度 (cm)": "Ruler length (cm)",
    "像素距離": "Pixel distance",
    "相機": "Camera",
    "與上次相比": "Change vs previous",
    "{label} 的 cm/pixel 與上次校正相差 {change}，請確認相機是否被移動或量尺點選是否正確": "{label} cm/pixel changed {change} from the previous calibration; check whether the camera moved or the ruler points were picked correctly",
    "請輸入手動重量 (g)": "Enter the manual weight (g)",
    "使用手動重量": "Use manual weight",
    "拍攝中…": "Capturing…",
    "秤重設備未取得有效讀值，請確認後再按一次，或勾選「使用手動重量」":
        "No valid scale reading. Check the scale and press again, or tick \"Use manual weight\".",
    "重量尚未穩定，請等雞隻靜止後再按一次": "Weight not stable yet. Wait until the chicken is still, then press again.",
    "重量 {weight} g 低於 {minimum} g，請確認雞隻已在秤上":
        "Weight {weight} g is below {minimum} g. Make sure the chicken is on the scale.",
    # trimmed UI text
    '收集 2 秒平均重量中…': 'Collecting 2 s average weight…',
    # ArUco calibration
    "雞冠與腳脛相機可以分別校正：ArUco 標記自動校正，或拍照點兩點手動校正。":
        "Calibrate each camera on its own: automatically with the ArUco marker, or by clicking two points.",
    "校正方式": "Method",
    "ArUco 自動": "ArUco (auto)",
    "手動點選": "Manual points",
    "方式": "Method",
    "參考長度 (cm)": "Reference length (cm)",
    "{label} 標記邊長 (cm)": "{label} marker side (cm)",
    "黑色外框的邊長。列印尺寸不準時，請填用尺量到的實際邊長。":
        "Edge of the black frame. If the print was scaled, enter the side length measured with a ruler.",
    "找到 ID {marker_id} 標記：邊長 {side} px，比例 {scale} cm/像素":
        "Found marker ID {marker_id}: side {side} px, scale {scale} cm/pixel",
    "找不到可用的標記，請重新拍攝或改用手動點選": "No usable marker; retake the photo or use manual points",
    "無法讀取校正照片，請重新拍攝": "Cannot read the calibration photo; please retake it",
    "可調整標記後重新拍攝，或在上方改用「手動點選」": "Adjust the marker and retake, or switch to manual points above",
    "找不到 ID {marker_id} 標記，請確認標記在畫面中、沒有被遮住或反光":
        "Marker ID {marker_id} not found; make sure it is in view, not covered and without glare",
    "畫面中只找到 ID {found} 標記，這台相機應使用 ID {marker_id}":
        "Only marker ID {found} is in view; this camera uses ID {marker_id}",
    "畫面中有 {count} 個 ID {marker_id} 標記，請只留一個": "{count} markers with ID {marker_id} are in view; keep only one",
    "標記在畫面中太小（邊長 {side:.0f} px，至少需 {minimum} px），請改用較大的標記或靠近相機":
        "The marker is too small ({side:.0f} px, at least {minimum} px); use a bigger marker or move it closer",
    "標記太斜（最長邊／最短邊 = {ratio:.2f}），請讓標記正對相機":
        "The marker is too tilted (longest / shortest side = {ratio:.2f}); face it to the camera",
}
