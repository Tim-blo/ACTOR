from ..datasets.get_dataset import get_datasets
from ..recognition.get_model import get_model as get_rec_model
from ..models.get_model import get_model as get_gen_model


def get_model_and_data_checkpointed(parameters):
    datasets = get_datasets(parameters)
    model = get_gen_model(parameters)
    print("Restore weights..")
    checkpoint = parameters["checkpoint"]
    checkpoint_path = os.path.join(parameters["folder"],
                                               'checkpoint_{:04d}.pth.tar'.format(checkpoint))
    state_dict = torch.load(checkpoint_path, map_location=parameters["device"])
    model.load_state_dict(state_dict)
    print("Checkpoint model loaded!")
    return model, datasets
