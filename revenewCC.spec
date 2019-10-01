# -*- mode: python ; coding: utf-8 -*-

block_cipher = None


a = Analysis(['C:/DataScience/Projects/Revenew/revenewcc/revenewCC/ranking.py'],
             pathex=['C:\\DataScience\\Projects\\Revenew\\revenewcc'],
             binaries=[],
             datas=[
             ('C:/DataScience/Projects/Revenew/revenewcc/gooey/images', 'gooey/images'),
             ('C:/DataScience/Projects/Revenew/revenewcc/gooey/images/AvianaLogo.png', 'gooey/images/AvianaLogo.png'),
             ('C:/DataScience/Projects/Revenew/revenewcc/gooey/languages', 'gooey/languages')],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=True)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [('v', None, 'OPTION')],
          name='RevenewCC',
          debug=True,
          bootloader_ignore_signals=False,
          strip=False,
          upx=False,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=True )
