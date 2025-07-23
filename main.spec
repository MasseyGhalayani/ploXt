# main.spec

# -*- mode: python ; coding: utf-8 -*-

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[
        # This copies the mmdetection folder for its configs
        ('mmdetection', 'mmdetection'),
        # --- FIX: This copies your model folder, including configs and weights ---
        ('model', 'model'),
        ('UI/resources/app_icon.png', './UI/resources')
    ],
    hiddenimports=[
        # This explicitly tells PyInstaller to find the missing extension.
        'mmcv._ext'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    # This tells PyInstaller to find and include all files from the mmcv package.
    collect_all=['mmcv'],
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='PloXt',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='UI/resources/app_icon.png',
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='PloXt',
)