import torch
import torch.nn.functional as F
import numpy as np
from framework.models_pytorch import move_data_to_gpu


def forward(model, generate_func, cuda):
    outputs = []
    outputs_event = []

    targets = []
    targets_event = []

    # Evaluate on mini-batch
    for data in generate_func:
        (batch_x, batch_x_rms, batch_y, batch_y_event) = data

        batch_x = move_data_to_gpu(batch_x, cuda)
        batch_x_rms = move_data_to_gpu(batch_x_rms, cuda)
        # print(batch_x.size())

        model.eval()
        with torch.no_grad():
            all_output = model(batch_x, batch_x_rms)
            batch_rate, batch_output_event = all_output[0], all_output[1]

            batch_output_event = F.sigmoid(batch_output_event)

            outputs.append(batch_rate.data.cpu().numpy())
            outputs_event.append(batch_output_event.data.cpu().numpy())

            targets.append(batch_y)
            targets_event.append(batch_y_event)

    dict = {}

    if len(outputs):
        outputs = np.concatenate(outputs, axis=0)
    dict['output'] = outputs

    if len(outputs_event):
        outputs_event = np.concatenate(outputs_event, axis=0)
    dict['outputs_event'] = outputs_event

    targets = np.concatenate(targets, axis=0)
    dict['target'] = targets
    targets_event = np.concatenate(targets_event, axis=0)
    dict['targets_event'] = targets_event
    return dict



def forward_SSC(model, generate_func, cuda):
    outputs_event = []
    targets_event = []

    # Evaluate on mini-batch
    for data in generate_func:
        (batch_x, batch_y_event) = data

        batch_x = move_data_to_gpu(batch_x, cuda)

        model.eval()
        with torch.no_grad():
            all_output = model(batch_x)
            batch_output_event = all_output

            batch_output_event = F.sigmoid(batch_output_event)
            outputs_event.append(batch_output_event.data.cpu().numpy())
            targets_event.append(batch_y_event)

    dict = {}

    if len(outputs_event):
        outputs_event = np.concatenate(outputs_event, axis=0)
    dict['outputs_event'] = outputs_event

    targets_event = np.concatenate(targets_event, axis=0)
    dict['targets_event'] = targets_event
    return dict



