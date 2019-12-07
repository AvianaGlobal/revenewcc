from ranking.gooey.gui.components.widgets import core
from ranking.gooey import TextContainer, BaseChooser


__ALL__ = [
    'FileChooser',
    'FileSaver',
    'DirChooser',
    'DateChooser'
]

class FileChooser(BaseChooser):
    # todo: allow wildcard from argparse
    widget_class = core.FileChooser


class MultiFileChooser(BaseChooser):
    # todo: allow wildcard from argparse
    widget_class = core.MultiFileChooser


class FileSaver(BaseChooser):
    # todo: allow wildcard
    widget_class = core.FileSaver


class DirChooser(BaseChooser):
    # todo: allow wildcard
    widget_class = core.DirChooser


class DateChooser(BaseChooser):
    # todo: allow wildcard
    widget_class = core.DateChooser

