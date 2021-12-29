from src.parser.evaluation import parser
from .stgcn_eval import evaluate


def main():
    parameters, folder, checkpointname, epoch, niter = parser()
    dataset = parameters["dataset"]
    print(dataset)
    if dataset in ["ntu13", "humanact12"]:
        evaluate(parameters, folder, checkpointname, epoch, niter)
    else:
        raise NotImplementedError("This dataset is not supported.")


if __name__ == '__main__':
    main()
