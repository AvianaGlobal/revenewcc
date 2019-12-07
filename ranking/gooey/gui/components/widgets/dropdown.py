from ranking.gooey import TextContainer
import wx

from ranking.gooey.gui import formatters
from ranking.gooey import _


class Dropdown(TextContainer):

    def getWidget(self, parent, *args, **options):
        default = _('select_option')
        return wx.ComboBox(
            parent=parent,
            id=-1,
            value=default,
            choices=[default] + self._meta['choices'],
            style=wx.CB_DROPDOWN)

    def setOptions(self, options):
        prevSelection = self.widget.GetSelection()
        self.widget.Clear()
        for option in [_('select_option')] + options:
            self.widget.Append(option)
        self.widget.SetSelection(0)


    def setValue(self, value):
        ## +1 to offset the default placeholder value
        index = self._meta['choices'].index(value) + 1
        self.widget.SetSelection(index)

    def getWidgetValue(self):
        return self.widget.GetValue()

    def formatOutput(self, metadata, value):
        return formatters.dropdown(metadata, value)
