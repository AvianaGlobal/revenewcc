import os

from PyInstaller.building.api import EXE, PYZ, COLLECT
from PyInstaller.building.build_main import Analysis
from PyInstaller.building.datastruct import Tree

block_cipher = None

a = Analysis(['revenewCC/ranking.py'],
             pathex=['C:/Users/MichaelJohnson/revenewcc'],
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
             binaries=[
             	('C:/Users/MichaelJohnson/revenewcc/env', '.'),
             	('C:/Users/MichaelJohnson/revenewcc/env/Library', './Library'),
             	('C:/Users/MichaelJohnson/revenewcc/env/Library/bin', './Library/bin'),
             ],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             cipher=block_cipher)
pyz = PYZ(a.pure,
          a.zipped_data,
          cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          exclude_binaries=False,
          name='RevenewCC',
          debug=True)
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               name='RevenewCC')
