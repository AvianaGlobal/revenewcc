shebang

pyinstaller --windowed \
  --add-data='LICENSE:.' \
  --add-data='MANIFEST.in:.' \
  --add-data='gooey/languages/*.json:gooey/languages' \
  --add-data='gooey/images/*.png:gooey/images' \
  --add-data='gooey/images/*.ico:gooey/images' \
  --add-data='goeey/images/loading_icon.gif:gooey/images' \
  --upx-dir='/usr/local/bin/upx'
  revenewCC/ranking.py
