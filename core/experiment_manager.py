import os
import ipywidgets as widgets
from IPython.display import display

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, "models", "experiments")

def get_train_widgets(default_name="baseline_model"):
    """
    Возвращает текстовое поле для ввода имени эксперимента.
    """
    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
    exp_widget = widgets.Text(value=default_name, description="Experiment:", layout=widgets.Layout(width='50%'))
    display(exp_widget)
    return exp_widget

def get_eval_widgets():
    """
    Возвращает выпадающий список для выбора эксперимента.
    """
    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
    experiments = [d for d in os.listdir(EXPERIMENTS_DIR) if os.path.isdir(os.path.join(EXPERIMENTS_DIR, d))]
    
    if not experiments:
        experiments = ["No experiments found"]
        
    dropdown = widgets.Dropdown(
        options=experiments,
        description='Experiment:',
        disabled=False,
        layout=widgets.Layout(width='50%')
    )
    display(dropdown)
    return dropdown

def get_experiment_paths(experiment_name):
    """
    Возвращает пути для сохранения/загрузки: (model_path, norm_path, tensorboard_log).
    """
    base_path = os.path.join(EXPERIMENTS_DIR, experiment_name)
    os.makedirs(base_path, exist_ok=True)
    
    model_path = os.path.join(base_path, "ppo_model")
    norm_path = os.path.join(base_path, "vec_normalize.pkl")
    tensorboard_log = os.path.join(PROJECT_ROOT, "tensorboard_logs", experiment_name)
    
    return model_path, norm_path, tensorboard_log
