from typing import Type, Dict


from rsna.task.base import BaseTorchModelTraining, load_torch_model_training_from_task_dir, \
    load_torch_model_training_from_task_id


def evaluate_torch_model_from_task_id(model_cls: Type[BaseTorchModelTraining], task_id: str) -> Dict[str, float]:
    model = load_torch_model_training_from_task_id(model_cls, task_id)
    return evaluate_torch_model(model)


def evaluate_torch_model_from_task_dir(model_cls: Type[BaseTorchModelTraining], task_dir: str) -> Dict[str, float]:
    model = load_torch_model_training_from_task_dir(model_cls, task_dir)
    return evaluate_torch_model(model)


def evaluate_torch_model(model: BaseTorchModelTraining) -> Dict[str, float]:
    _, _, test_generator = model.get_train_val_test_data_loader()

    torchbearer_model = model.create_torchbearer_model(model.get_trained_module())

    return torchbearer_model.evaluate_generator(test_generator)
