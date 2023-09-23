import sys, os, argparse


gpu_id = 1
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

sys.path.append(os.path.split(os.path.dirname(os.path.realpath(__file__)))[0])
from framework.models_pytorch import *


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', type=str, required=True)
    args = parser.parse_args()

    model_type = args.model
    models = ['DNN', 'CNN', 'CNN_Transformer', 'DCNN_CaF', 'YAMNet', 'PANN', 'AST', 'DCNN_CaF_SSC']
    model_index = models.index(model_type)
    using_models = [DNN, CNN, CNN_Transformer, DCNN_CaF, YAMNet, PANN, ASTModel, DCNN_CaF_SSC]

    event_class = len(config.event_labels)
    model = using_models[model_index](event_class)

    params_num = count_parameters(model)
    print('Parameters num: {} M'.format(params_num / 1000 ** 2))




if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)















