from django import forms
from django.core.validators import RegexValidator

# Used as a filename prefix (checkpoints/{run_name}_final_model.pth etc.), so keep it
# restricted to characters that are always safe in a path segment.
_RUN_NAME_VALIDATOR = RegexValidator(
    regex=r'^[A-Za-z0-9_\-]{1,40}$',
    message="Lettres, chiffres, tirets et underscores uniquement (1 à 40 caractères).",
)


class TrainingConfigForm(forms.Form):
    """Mirrors train.py's CLI arguments. Used both to render the config form
    and to validate the JSON payload received over the training WebSocket."""

    run_name = forms.CharField(required=False, initial='', validators=[_RUN_NAME_VALIDATOR])
    algo = forms.ChoiceField(choices=[('DQN', 'DQN'), ('DDQN', 'DDQN')], initial='DQN')
    obs = forms.ChoiceField(choices=[('vector', 'Vector'), ('image', 'Image')], initial='vector')
    episodes = forms.IntegerField(min_value=1, max_value=200000, initial=1000)
    max_steps = forms.IntegerField(min_value=1, max_value=100000, initial=500)
    batch_size = forms.IntegerField(min_value=1, max_value=2048, initial=32)
    lr = forms.FloatField(min_value=0.0000001, max_value=1.0, initial=0.00025)
    obstacles = forms.IntegerField(min_value=0, max_value=200, initial=10)
    no_food_steps = forms.IntegerField(min_value=0, max_value=100000, initial=0)
    step_penalty = forms.FloatField(min_value=0.0, max_value=10.0, initial=0.01)
    use_custom_obstacles = forms.BooleanField(required=False, initial=False)
    train_speed = forms.ChoiceField(
        choices=[('fast', 'Rapide'), ('slow', 'Lent (visualisation fluide)')], initial='fast'
    )
    resume = forms.BooleanField(required=False, initial=False)
    resume_model = forms.CharField(required=False, initial='final_model')
