shebang

pyinstaller --windowed \
  --add-data='LICENSE:.' \
  --add-data='MANIFEST.in:.' \
  --add-data='gooey/languages/*.json:gooey/languages' \
  --add-data='gooey/tests/*.json:gooey/tests' \
  --add-data='gooey/images/*.png:gooey/images' \
  --add-data='gooey/images/*.ico:gooey/images' \
  --add-data='gooey/images/*.jpg:gooey/images' \
  --add-data='gooey/images/*.gif:gooey/images' \
  --add-binary='/Users/mj/opt/anaconda3/lib/libpython3.7m.dylib:.' \
  revenewCC/ranking.py


