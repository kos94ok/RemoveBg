from .models import Task
from .models import SaveImage
from django.forms import ModelForm

class TaskForm(ModelForm):
    class Meta:
        model = Task
        fields = ['textInput']


class SaveImageForm(ModelForm):
    class Meta:
        model = SaveImage
        fields = ['textInput', 'image']
