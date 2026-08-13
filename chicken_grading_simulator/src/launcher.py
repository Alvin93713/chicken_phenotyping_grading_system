"""Windows exe 入口：啟動 Streamlit 或執行內部 SAM2 pipeline。"""

from __future__ import annotations

import os
import runpy
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path


def app_root() -> Path:
    """取得 onedir exe 或原始碼模式下的 App 根目錄。"""
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent)).resolve()
    return Path(__file__).resolve().parents[1]


def run_internal_pipeline(root: Path) -> None:
    """由同一個 exe 執行 sam2_pipeline.py，供 Streamlit 子程序呼叫。"""
    script_path = root / "data" / "scripts" / "sam2_pipeline.py"
    if not script_path.exists():
        raise FileNotFoundError(f"找不到推論腳本：{script_path}")

    os.chdir(root)
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(script_path.parent))
    sys.argv = [str(script_path), *sys.argv[2:]]
    runpy.run_path(str(script_path), run_name="__main__")


def open_browser_later(url: str, delay_seconds: float = 2.5) -> None:
    """稍後開啟瀏覽器。"""
    time.sleep(delay_seconds)
    webbrowser.open(url)


def find_available_port(preferred_port: int = 8501, attempts: int = 50) -> int:
    """從 preferred_port 開始尋找可用 localhost port。"""
    for port in range(preferred_port, preferred_port + attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError("找不到可用的 localhost port。")


def run_streamlit(root: Path) -> None:
    """啟動 Streamlit App。"""
    app_path = root / "src" / "app.py"
    if not app_path.exists():
        app_path = root / "app.py"
    if not app_path.exists():
        raise FileNotFoundError(f"找不到 Streamlit 主程式：{app_path}")

    os.chdir(root)
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "src"))
    os.environ.setdefault("STREAMLIT_GLOBAL_DEVELOPMENT_MODE", "false")
    os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    os.environ.setdefault("STREAMLIT_SERVER_HEADLESS", "true")

    preferred_port = int(os.environ.get("CHICKEN_SIMULATOR_PORT", "8501"))
    port = str(find_available_port(preferred_port))
    url = f"http://localhost:{port}"
    print("=" * 72, flush=True)
    print("雞隻表型資料分級與亮燈篩選系統", flush=True)
    print(f"正在啟動本機服務：{url}", flush=True)
    print("若瀏覽器未自動開啟，請複製上方網址到瀏覽器。", flush=True)
    print("關閉此視窗即可停止系統。", flush=True)
    print("=" * 72, flush=True)
    threading.Thread(target=open_browser_later, args=(url,), daemon=True).start()

    from streamlit.web import cli as stcli

    sys.argv = [
        "streamlit",
        "run",
        str(app_path),
        "--server.headless=true",
        f"--server.port={port}",
        "--browser.gatherUsageStats=false",
    ]
    raise SystemExit(stcli.main())


def main() -> None:
    root = app_root()
    if len(sys.argv) > 1 and sys.argv[1] == "--run-sam2-pipeline":
        run_internal_pipeline(root)
        return
    run_streamlit(root)


if __name__ == "__main__":
    main()

