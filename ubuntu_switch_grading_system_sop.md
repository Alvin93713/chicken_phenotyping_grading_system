# Windows 轉 Ubuntu 22.04 並順利運行雞隻分級系統 SOP

本文給接手的 agent 使用，目標是把目前在 Windows 上開發的雞隻表型分級系統，搬到一台原本 Windows 電腦改裝的 Ubuntu 22.04 x86_64 系統上，並以瀏覽器開啟 Streamlit UI 正常操作。

## 目前版本

- Git commit：`691e0db Rename simulator outputs to grading system`
- 可搬移 bundle：`chicken_grading_system_ubuntu_691e0db.bundle`
- Windows 原始位置：`C:\Users\蘇晨崴\Documents\種公雞分級模擬系統`
- Ubuntu 目標位置建議：`~/chicken_grading_system/chicken_grading_simulator`
- UI 目標網址：`http://localhost:8501`

## 重要前提

- 這是一般 x86_64 Ubuntu 22.04 電腦，不是 Jetson Orin。
- 不需要包成 exe，正式運行方式是本機啟動 Streamlit，再用瀏覽器開 localhost。
- 系統正式運行不需要外網；只有安裝套件、複製 Git/bundle、更新依賴時需要網路。
- QR 掃描槍使用 USB HID，等同鍵盤輸入。
- Arduino Leonardo 使用 serial port，Ubuntu 上通常是 `/dev/ttyACM0`。
- USB 相機通常是 `/dev/video0`。

## 1. 從 Windows 準備搬移檔案

在 Windows 端確認 bundle 存在：

```powershell
Get-Item "C:\Users\蘇晨崴\Documents\種公雞分級模擬系統\chicken_grading_system_ubuntu_691e0db.bundle"
```

把 bundle 複製到 Ubuntu 電腦，例如放在 Ubuntu 的 `~/Downloads/`。

可用方式：

- USB 隨身碟
- 區域網路檔案分享
- `scp`

範例：

```powershell
scp "C:\Users\蘇晨崴\Documents\種公雞分級模擬系統\chicken_grading_system_ubuntu_691e0db.bundle" user@UBUNTU_IP:~/Downloads/
```

## 2. Ubuntu 安裝系統套件

在 Ubuntu 22.04 執行：

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip git libgl1 libglib2.0-0 libzbar0 v4l-utils usbutils
sudo usermod -aG dialout $USER
sudo reboot
```

重開機後再繼續。`dialout` 是 Arduino serial port 權限需要。

## 3. Clone 專案

```bash
cd ~
git clone ~/Downloads/chicken_grading_system_ubuntu_691e0db.bundle chicken_grading_system
cd ~/chicken_grading_system/chicken_grading_simulator
```

確認版本：

```bash
git log --oneline -3
```

應看到：

```text
691e0db Rename simulator outputs to grading system
fe2d092 Improve scanner and calibration workflow
d7388b0 Enable CUDA inference on Jetson when available
```

## 4. 建立 Python 虛擬環境

```bash
cd ~/chicken_grading_system/chicken_grading_simulator
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel setuptools
```

## 5. 安裝 Python 套件

### CPU 版優先，先求系統跑通

```bash
pip install -r requirements.txt
```

若安裝 `opencv-python` 或 `torch` 很慢，正常；等它完成。

### 如果 Ubuntu 電腦有 NVIDIA GPU

先確認 driver：

```bash
nvidia-smi
```

如果 `nvidia-smi` 正常，之後可以改裝 CUDA 版 PyTorch；但第一輪建議先用 CPU 版跑通 UI、DB、掃描槍、Arduino、相機流程。

## 6. 檢查必要檔案

在專案目錄：

```bash
cd ~/chicken_grading_system/chicken_grading_simulator
ls -lh app.py best.pt data/chicken_phenotype.db
ls qrcodes | head
```

必要檔案：

- `app.py`
- `best.pt`
- `data/chicken_phenotype.db`
- `qrcodes/C001.png` 到 `qrcodes/C020.png`

## 7. 檢查硬體

### 相機

```bash
v4l2-ctl --list-devices
ls /dev/video*
```

常見：

```text
/dev/video0
```

系統 UI 裡的 `camera ID` 通常填 `0`。

### Arduino Leonardo

插上 Arduino 後：

```bash
ls /dev/ttyACM* /dev/ttyUSB*
```

常見：

```text
/dev/ttyACM0
```

如果看得到 port 但 Streamlit 不能連線，確認使用者在 `dialout`：

```bash
groups
```

應包含：

```text
dialout
```

若沒有，重新執行：

```bash
sudo usermod -aG dialout $USER
sudo reboot
```

### USB QR 掃描槍

掃描槍通常是 HID keyboard。測試：

```bash
gedit
```

或任何文字輸入框。掃 QR code 後應輸入類似：

```text
C001
```

並自動 Enter 換行。

## 8. 啟動系統

```bash
cd ~/chicken_grading_system/chicken_grading_simulator
source .venv/bin/activate
streamlit run app.py --server.address=0.0.0.0 --server.port=8501 --browser.gatherUsageStats=false
```

Ubuntu 本機瀏覽器開：

```text
http://localhost:8501
```

同區域網路其他電腦開：

```text
http://UBUNTU_IP:8501
```

## 9. 首次驗證流程

依序測：

1. 打開 `http://localhost:8501`
2. 進入「硬體連接設定」
3. 掃碼器測試欄位直接掃 `C001`
4. 選 Arduino serial port，例如 `/dev/ttyACM0`
5. 按「測試燈號」
6. 進入「比例尺校正」
7. 拍攝校正照片
8. 點兩個校正點
9. 按 D3 實體按鈕，確認校正
10. 進入「掃描 QR code 與表型輸入」
11. 掃 `C001`
12. 輸入重量
13. 按「計算雞冠及腳脛數據」或用對應實體按鈕
14. 等背景推論完成
15. 進入「再次掃描與亮燈判定」
16. 連續掃 `C001`、`C002`、`C003` 測試第二階段連續掃描

## 10. 常見問題

### localhost 打不開

確認 Streamlit 還在跑：

```bash
ps aux | grep streamlit
```

確認 port：

```bash
ss -ltnp | grep 8501
```

重新啟動：

```bash
cd ~/chicken_grading_system/chicken_grading_simulator
source .venv/bin/activate
streamlit run app.py --server.address=0.0.0.0 --server.port=8501 --browser.gatherUsageStats=false
```

### Arduino permission denied

```bash
sudo usermod -aG dialout $USER
sudo reboot
```

### 找不到相機

```bash
v4l2-ctl --list-devices
ls /dev/video*
```

換 UI 裡的 `camera ID`，例如 `0`、`1`、`2`。

### 掃描槍無法連續掃

先在文字編輯器測試掃描槍是否會送 Enter：

```text
C001
C002
C003
```

若變成同一行：

```text
C001C002C003
```

代表掃描槍沒有設定 Enter suffix，需要依掃描槍手冊設定 CR/LF 或 Enter。

### YOLO 推論很慢

一般 CPU 版 Ubuntu 會慢，這是正常的。先確認流程正確；若電腦有 NVIDIA GPU，再安裝 CUDA 版 PyTorch。

## 11. 建議建立啟動腳本

建立 `run_grading_system.sh`：

```bash
cd ~/chicken_grading_system/chicken_grading_simulator
source .venv/bin/activate
streamlit run app.py --server.address=0.0.0.0 --server.port=8501 --browser.gatherUsageStats=false
```

給執行權限：

```bash
chmod +x ~/chicken_grading_system/chicken_grading_simulator/run_grading_system.sh
```

以後啟動：

```bash
~/chicken_grading_system/chicken_grading_simulator/run_grading_system.sh
```

## 12. 完成標準

切換成功的標準：

- Ubuntu 本機可以開 `http://localhost:8501`
- 掃描槍可直接輸入 chicken ID 並送 Enter
- Arduino 可由 UI 測試三燈
- 相機可拍攝校正照片
- 表型輸入頁可建立背景推論任務
- 推論完成後資料寫入 SQLite DB
- 再次掃描頁可連續掃描並亮燈判定


