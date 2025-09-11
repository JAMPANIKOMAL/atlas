# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

# a.datas: Add all the non-python files needed by the application
# We need to include the HTML file, the models, and the data directories
datas = [
    ('index.html', '.'),
    ('src', 'src'),
    ('models', 'models'),
    ('data', 'data')
]

a = Analysis(
    ['server.py'],
    pathex=['.'],
    binaries=[],
    datas=datas,
    hiddenimports=[],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# exe: Configure the executable
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='ATLAS',
    debug=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    console=True,  # This will open a console window to show server output
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='icon.ico',  # Assuming an icon file exists
)