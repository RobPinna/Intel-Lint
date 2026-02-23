# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"


def _data_entry(src: Path, dest: str):
    return (str(src), dest)


datas = [
    _data_entry(ROOT_DIR / "frontend" / "dist", "webui_dist"),
]

optional_data_dirs = [
    (ROOT_DIR / "src" / "intel_lint" / "static", "static"),
    (ROOT_DIR / "src" / "intel_lint" / "templates", "templates"),
]
for source_dir, dest_dir in optional_data_dirs:
    if source_dir.exists() and source_dir.is_dir():
        datas.append(_data_entry(source_dir, dest_dir))


a = Analysis(
    ["src/intel_lint/packaged_app.py"],
    pathex=[str(SRC_DIR)],
    binaries=[],
    datas=datas,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="IntelLint",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
