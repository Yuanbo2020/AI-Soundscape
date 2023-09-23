import sys, os, argparse


gpu_id = 1
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

sys.path.append(os.path.split(os.path.dirname(os.path.realpath(__file__)))[0])
from framework.models_pytorch import *


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main(argv):
    using_models = [DNN, CNN, CNN_Transformer, DCNN_CaF, YAMNet, PANN, ASTModel, DCNN_CaF_SSC]

    event_class = len(config.event_labels)

    for each_model in using_models:
        model = each_model(event_class)
        params_num = count_parameters(model)
        print(str(each_model), ' Parameters num: {} M'.format(params_num / 1000 ** 2))




if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)















