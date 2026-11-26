# 架設廠商端 HTTPS 接收伺服器

雞舍端會把審核後的影像加密簽章後，透過 HTTPS 傳到這台伺服器。架設完成後，廠商為每台雞舍裝置發一組「連線設定碼」；雞舍在畫面上貼上，就會自動開始回傳。

```
雞舍裝置 ──HTTPS──▶ Caddy (443, 自動申請憑證) ──▶ Uvicorn 接收程式 (127.0.0.1:8000) ──▶ /srv/chicken-review-inbox
```

- **Caddy**：負責 HTTPS，會自動向 Let's Encrypt 申請並續約免費憑證。
- **接收程式**：`scripts/provider_sync_server.py`，負責驗證金鑰與簽章，並儲存封包。

---

## 0. 事前準備

| 項目 | 說明 |
| --- | --- |
| 一台 Linux 主機 | 建議 Ubuntu 22.04 或 24.04。雲端 VM（GCP、AWS、Azure、Linode 等）或公司機房都可以，1 vCPU、1 GB RAM 就夠。硬碟依影像量估算：每張約 0.2–1 MB，含原圖與結果圖。 |
| 固定的公開 IP | 雞舍要能從網際網路連到它。 |
| 一個網域名稱 | 例如 `provider.example.com`。Let's Encrypt 只發憑證給網域，不發給純 IP。 |
| 防火牆開放 80、443 | 443 用於上傳；80 只用於申請與續約憑證。 |

沒有公開 IP（例如學校內網）時，請見最後的「替代方案：Cloudflare Tunnel」。

---

## 1. 網域指向主機

在網域註冊商或 DNS 管理介面，新增一筆 **A 紀錄**：`provider.example.com` → 主機的公開 IP。

確認是否生效：

```bash
nslookup provider.example.com      # 應回傳主機 IP
```

## 2. 系統基本設定

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-venv ufw
sudo timedatectl set-ntp true            # 時間必須準確：簽章只接受誤差 5 分鐘內的請求
sudo ufw allow OpenSSH
sudo ufw allow 80,443/tcp
sudo ufw enable
```

如果是雲端 VM，還要在雲端主控台的安全群組（Security Group 或防火牆規則）開放 TCP 80 和 443。

## 3. 安裝 Caddy（自動 HTTPS）

```bash
sudo apt install -y debian-keyring debian-archive-keyring apt-transport-https curl
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list
sudo apt update && sudo apt install -y caddy
```

上面是 Caddy 官方的 apt 安裝方式。如果指令失敗，請以 <https://caddyserver.com/docs/install> 的最新說明為準。

## 4. 安裝接收程式

接收端只需要 `scripts/` 資料夾和 `requirements-provider.txt`，不需要雞舍端的模型與程式。

```bash
# 專用帳號，不能登入
sudo useradd --system --home /opt/chicken-provider --shell /usr/sbin/nologin chicken-provider

# 放程式（從本專案複製 scripts/ 與 requirements-provider.txt）
sudo mkdir -p /opt/chicken-provider
sudo cp -r scripts requirements-provider.txt /opt/chicken-provider/
sudo python3 -m venv /opt/chicken-provider/.venv
sudo /opt/chicken-provider/.venv/bin/pip install -r /opt/chicken-provider/requirements-provider.txt

# 資料夾與憑證檔，只有服務帳號可以讀寫
sudo mkdir -p /srv/chicken-review-inbox /srv/chicken-dataset /etc/chicken-provider
sudo chown -R chicken-provider:chicken-provider /opt/chicken-provider /srv/chicken-review-inbox /srv/chicken-dataset /etc/chicken-provider
sudo chmod 700 /srv/chicken-review-inbox /srv/chicken-dataset /etc/chicken-provider
```

## 5. 讓接收程式常駐（systemd）

```bash
sudo cp deploy/provider/chicken-provider.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now chicken-provider
sudo systemctl status chicken-provider      # 應顯示 active (running)
```

## 6. 設定 Caddy

先把 `deploy/provider/Caddyfile` 裡的 `provider.example.com` 換成你的網域，再套用：

```bash
sudo cp deploy/provider/Caddyfile /etc/caddy/Caddyfile
sudo mkdir -p /var/log/caddy && sudo chown caddy:caddy /var/log/caddy
sudo caddy validate --config /etc/caddy/Caddyfile
sudo systemctl reload caddy
```

Caddy 會在第一次有人連線時自動申請憑證，通常只要幾秒。若失敗，用 `sudo journalctl -u caddy -n 50` 查看原因；最常見的是 DNS 尚未生效，或 80 埠沒有開。

## 7. 確認伺服器可以從外部連到

在**任何一台其他電腦**上執行：

```bash
curl -i -X POST https://provider.example.com/v1/ping
```

預期結果：`HTTP/2 400`，內容為 `{"detail":"Missing header: X-Device-ID"}`。這代表 HTTPS 憑證、Caddy 轉送、接收程式三者都正常；回 400 是因為這個請求沒有帶金鑰，屬於正常情況。

如果少了 `-X POST`，會得到 `405 Method Not Allowed`。這也代表伺服器有在運作，只是這個端點只接受 POST。

| 看到的結果 | 原因 |
| --- | --- |
| 憑證錯誤或 SSL 錯誤 | Caddy 還沒成功申請到憑證（見第 6 步） |
| `502 Bad Gateway` | 接收程式沒在執行：`sudo systemctl status chicken-provider` |
| 連線逾時 | 防火牆或雲端安全群組沒開放 443 |

## 8. 為每台雞舍裝置發放連線設定碼

```bash
cd /opt/chicken-provider
sudo -u chicken-provider .venv/bin/python scripts/provider_issue_device.py \
  --endpoint https://provider.example.com/v1/review-packages \
  --site-id farm-a --device-id orin-001 \
  --credentials /etc/chicken-provider/credentials.json
```

- 指令會印出一行 `CHKSYNC1:...` 設定碼。請當面交付，或用加密訊息傳給雞舍，並把它視同密碼保管。
- 每個場域、每台裝置都要用不同的 `--site-id` 和 `--device-id`。
- 接收程式每次收到請求都會重新讀取憑證檔，所以發新的設定碼**不需要重啟服務**。
- 同一個 device-id 再執行一次，就會更換金鑰，舊設定碼立即失效（例如裝置遺失時）。

雞舍端的操作：「資料與模型維護」→「連線設定」→ 貼上設定碼 →「儲存連線設定碼」→「測試連線」，出現「連線成功」即完成。

## 9. 日常使用

**整理成微調資料集**：工具會同時讀取一鍵回傳的資料包（每台裝置只用最新的一份，因為每份都是完整集合）和舊版裝置的單張封包，同一張影像以最新的審核為準。結果會分成 comb、shank 兩份 YOLO-seg 資料集，各自包含 `train_ready/`（正確，可直接訓練）和 `to_relabel/`（需人工修正）。說明見 SECURE_SYNC.md 的「資料包結構」。

```bash
sudo -u chicken-provider /opt/chicken-provider/.venv/bin/python \
  /opt/chicken-provider/scripts/provider_export_dataset.py \
  --inbox /srv/chicken-review-inbox --out /srv/chicken-dataset/$(date +%F)
```

**查看收件狀況：**

```bash
sudo journalctl -u chicken-provider -f                       # 即時查看上傳紀錄
find /srv/chicken-review-inbox -name '*.zip' | wc -l         # 已收到的封包數
```

**備份**：定期備份 `/srv/chicken-review-inbox`（影像）和 `/etc/chicken-provider/credentials.json`（金鑰）。金鑰備份要另外加密保存。

**更新程式**：把新版 `scripts/` 複製到 `/opt/chicken-provider/scripts/`，再執行 `sudo systemctl restart chicken-provider`。

**簽章版本 2**：新版伺服器要求雞舍端的簽章同時涵蓋場域代碼（X-Site-ID），並記錄每個 nonce 防止重送。尚未更新的雞舍裝置會被拒絕（401「Signature version 2 required」）；更新期間若需暫時接受舊簽章，在服務的環境變數加上 `CHICKEN_PROVIDER_ACCEPT_LEGACY_SIGNATURES=1`，所有裝置更新後再移除。

---

## 替代方案：Cloudflare Tunnel（沒有公開 IP 時）

如果主機在學校或公司內網，沒有公開 IP，也不能開放 80／443，可以改用 Cloudflare Tunnel。

- **作法**：主機主動向 Cloudflare 建立加密通道，由 Cloudflare 對外提供 HTTPS 憑證。
- **條件**：網域需要託管在 Cloudflare。
- **步驟**：
  1. 照常完成第 2、4、5 步，接收程式一樣跑在 127.0.0.1:8000。
  2. 不用安裝 Caddy，也不用開放 80／443。
  3. 安裝 `cloudflared`，把 tunnel 的公開網址指向 `http://127.0.0.1:8000`。詳細步驟以 Cloudflare 官方文件為準。
  4. 第 7、8 步改用 tunnel 的網址即可。

## 安全提醒

- 不要使用自簽憑證，也不要要求雞舍關閉憑證驗證。
- `/etc/chicken-provider/credentials.json` 和設定碼都等同密碼：不要放進 git，也不要用 Email 或公開群組傳送。
- 伺服器只對外開放 `/v1/ping`、`/v1/snapshots/*`（一鍵回傳）和 `/v1/review-packages`（舊版裝置），其他路徑一律回 404（Caddyfile 已設定）。
