# -*- mode: python ; coding: utf-8 -*-

block_cipher = None


a = Analysis(['./ranking/ranking.py'],
             pathex=['C:/Users/MichaelJohnson/revenewcc'],
             datas=[
    	        ('./LICENSE', '.'),
            	('./MANIFEST.in', '.'),
             	('./ranking/inputdata/*.csv', './ranking/inputdata'),
             	('./ranking/inputdata/*.xlsx', './ranking/inputdata'),
             	('./ranking/inputdata/*.pkl', './ranking/inputdata'),
             	('./ranking/gooey/images/*.png', './ranking/gooey/images'),
             	('./ranking/gooey/images/*.ico', './ranking/gooey/images'),
             	('./ranking/gooey/images/*.gif', './ranking/gooey/images'),
             	('./ranking/gooey/languages/*.json', './ranking/gooey/languages'),
             ],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=False,
          name='ranking',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=True)
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='ranking')
