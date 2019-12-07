from ranking.gooey import PasswordInput
from ranking.gooey import TextField


__ALL__ = ('PasswordField',)

class PasswordField(TextField):
    widget_class = PasswordInput

    def __init__(self, *args, **kwargs):
        super(PasswordField, self).__init__(*args, **kwargs)

