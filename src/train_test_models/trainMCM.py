#%%
import sys
from pathlib import Path
from datetime import datetime

project_root = Path('/Users/Sol/Desktop/CohenLab/DynamicTaskPerceptionProject/task-interference') #Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.tasks import MonkeyHistoryTask
from src.psychrnn.backend.models.basic import Basic
from src.train_test_models.train_model import TrainModel

#%% Train n_models new Monkey Choice Models
# Note: Random seed selected randomly if not specified
specified_model_names = None # Replace with list of desired model names
custom_task_kwargs = [{'K': k} for k in [2, 3, 5, 8]]
n_models = len(custom_task_kwargs)

for i in range(n_models):
    if specified_model_names is not None:
        new_model_name = specified_model_names[i]
    else:
        datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_model_name = f'MCM_K{custom_task_kwargs[i]['K']}_{datetime_str}'
    print(f'Training model {i+1} of {n_models}: {new_model_name}')

    MonkeyChoiceModel = TrainModel(
        model_name = new_model_name,
        parent_folder = 'monkey_choice_model',
        model_class = Basic,
        task_class = MonkeyHistoryTask,
        task_kwargs = custom_task_kwargs[i],
        # modular_inputs = False,
        # modular_outputs = False
    )
    MonkeyChoiceModel.train()
# %%
