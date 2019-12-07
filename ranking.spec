# -*- mode: python ; coding: utf-8 -*-

block_cipher = None


a = Analysis(['revenewCC\\ranking.py'],
             pathex=['C:\\Users\\MichaelJohnson\\revenewcc'],
             datas=[
    	        ('LICENSE', '.'),
            	('MANIFEST.in', '.'),
             	('revenewCC/inputdata/*.csv', './revenewCC/inputdata'),
             	('revenewCC/inputdata/*.xlsx', './revenewCC/inputdata'),
             	('revenewCC/inputdata/*.pkl', './revenewCC/inputdata'),
             	('gooey/images/*.png', './gooey/images'),
             	('gooey/images/*.ico', './gooey/images'),
             	('gooey/images/*.gif', './gooey/images'),
             	('gooey/languages/*.json', './gooey/languages'),
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
          console=True )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='ranking')
