import wx
from ranking.gooey import TextContainer
from ranking.gooey.gui import events, formatters
from ranking.gooey import TextInput
from ranking.gooey import pub
from ranking.gooey import getin


class TextField(TextContainer):
    widget_class = TextInput

    def getWidgetValue(self):
        return self.widget.getValue()

    def setValue(self, value):
        self.widget.setValue(str(value))

    def formatOutput(self, metatdata, value):
        return formatters.general(metatdata, value)

