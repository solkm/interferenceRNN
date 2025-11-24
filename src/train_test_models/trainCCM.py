import sys
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.self_history import RNN_SH2
from src.tasks import SelfHistoryTask
from src.train_test_models.train_model import TrainModel

#%% Train n_models new Correct Choice Models
# Note: Random seed selected randomly if not specified
n_models = 4
specified_model_names = None # Replace with list of desired model names
custom_task_kwargs = [{'K': k} for k in [2, 4, 6, 8]]

for i in range(n_models):
    if specified_model_names is not None:
        new_model_name = specified_model_names[i]
    else:
        datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_model_name = f'CCM_K{custom_task_kwargs[i]['K']}_{datetime_str}'
        print(f'Training model {i+1} of {n_models}: {new_model_name}')
    
    CorrectChoiceModel = TrainModel(
        model_name = new_model_name,
        parent_folder = 'correct_choice_model',
        model_class = RNN_SH2,
        task_class = SelfHistoryTask,
        task_kwargs = custom_task_kwargs[i],
        # modular_inputs = False,
        # modular_outputs = False
    )
    CorrectChoiceModel.train()