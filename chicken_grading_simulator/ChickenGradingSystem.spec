# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules, copy_metadata


block_cipher = None
root = Path.cwd()

datas = [
    (str(root / "app.py"), "."),
    (str(root / "database.py"), "."),
    (str(root / "grading.py"), "."),
    (str(root / "inference_utils.py"), "."),
    (str(root / "inference_queue.py"), "."),
    (str(root / "arduino_control.py"), "."),
    (str(root / "qr_utils.py"), "."),
    (str(root / "sample_data.py"), "."),
    (str(root / "README.md"), "."),
    (str(root / "requirements.txt"), "."),
    (str(root / "best.pt"), "."),
    (str(root / "data"), "data"),
    (str(root / "qrcodes"), "qrcodes"),
    (str(root / "arduino"), "arduino"),
    (str(root / "rendered_qr_docx"), "rendered_qr_docx"),
    (str(root / "Chicken_ID_QR_Codes_C001_C020.docx"), "."),
]

binaries = []
hiddenimports = [
    "altair",
    "cv2",
    "numpy",
    "pandas",
    "PIL",
    "pyarrow",
    "pyzbar",
    "qrcode",
    "streamlit",
    "streamlit_image_coordinates",
    "streamlit.web.cli",
    "serial",
    "serial.tools.list_ports",
    "torch",
    "torchvision",
    "tqdm",
    "ultralytics",
    "ultralytics.models.sam",
    "ultralytics.models.yolo",
    "ultralytics.nn.tasks",
]

hiddenimports += collect_submodules("streamlit")
hiddenimports += collect_submodules("ultralytics", filter=lambda name: ".tests" not in name)

binaries += collect_dynamic_libs("torch")
binaries += collect_dynamic_libs("torchvision")
binaries += collect_dynamic_libs("cv2")
binaries += collect_dynamic_libs("pyzbar")

datas += collect_data_files("streamlit", include_py_files=False)
datas += collect_data_files("streamlit_image_coordinates", include_py_files=False)
datas += collect_data_files("altair", include_py_files=False)
datas += collect_data_files("pyarrow", include_py_files=False)
datas += collect_data_files("ultralytics", include_py_files=False)

for metadata_package in [
    "streamlit",
    "streamlit-image-coordinates",
    "pyserial",
    "altair",
    "pyarrow",
    "pandas",
    "numpy",
    "pillow",
    "opencv-python",
    "torch",
    "torchvision",
    "ultralytics",
    "ultralytics-thop",
    "qrcode",
    "pyzbar",
]:
    datas += copy_metadata(metadata_package)

a = Analysis(
    ["launcher.py"],
    pathex=[str(root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="ChickenGradingSystem",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="ChickenGradingSystem",
)
