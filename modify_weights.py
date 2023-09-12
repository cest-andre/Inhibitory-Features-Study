import os
import torch
import numpy as np
from numpy.random import default_rng
from PIL import Image
from torchvision import transforms, models
from thingsvision import get_extractor
import matplotlib.pyplot as plt


#   For all negative weights after first layer, change value to zero and return new model.
#   First layer is skipped as negative weights could be excitatory due to negative pixel values.
def ablate_all_inhibs(states):
    keys = list(states.keys())
    for i in range(len(keys)):
        k = keys[i]
        if 'weight' in k and i != 0:
            states[k] = torch.clamp(states[k], min=0)

    return states


def ablate_inhib_layer(states, layername):
    states[layername] = torch.clamp(states[layername], min=0)

    return states


def clamp_ablate_unit(states, layername, unit, min=None, max=None, bias_name=None):
    states[layername][unit] = torch.clamp(states[layername][unit], min=min, max=max)
    if bias_name is not None:
        states[bias_name][unit] = 0

    return states


def random_ablate_unit(states, layername, unit, perc=None, bias_name=None):
    weights = torch.flatten(states[layername][unit])
    rng = default_rng()

    if perc == 1:
        #   Leave one weight intact to avoid zero std.  Last one selected arbitrarily.
        weights[0:-1] = 0
    else:
        #   If p is not passed, we obtain p as the ratio of negative weights to all weights.
        #   This is done to control for inhib ablation (ablate the same number of weights but randomly).
        #   **I really should just pass the number of < 0 weights directly to rng.choice rather than multiply.
        if perc is None:
            weights[rng.choice(weights.shape[0], int(torch.nonzero(weights < 0).shape[0]), replace=False)] = 0
        else:
            weights[rng.choice(weights.shape[0], int(weights.shape[0] * perc), replace=False)] = 0

    states[layername][unit] = torch.reshape(weights, states[layername][unit].shape)

    if bias_name is not None:
        states[bias_name][unit] = 0

    return states


def channel_random_ablate_unit(states, layername, unit, perc, bias_name=None):
    weights = states[layername][unit]

    rng = default_rng()
    weights[rng.choice(weights.shape[0], int(weights.shape[0] * perc), replace=False)] = 0
    states[layername][unit] = weights

    if bias_name is not None:
        states[bias_name][unit] = 0

    return states


def binarize_unit(states, layername, unit):
    unit_weights = torch.flatten(states[layername][unit])
    unit_weights[torch.nonzero(unit_weights > 0, as_tuple=True)] = 1
    unit_weights[torch.nonzero(unit_weights <= 0, as_tuple=True)] = -1

    states[layername][unit] = torch.reshape(unit_weights, states[layername][unit].shape)

    return states


def shuffle_unit(states, layername, unit, mode='all'):
    unit_weights = states[layername][unit]
    unit_weights = torch.flatten(unit_weights)

    if mode == 'all':
        unit_weights = unit_weights[torch.randperm(unit_weights.shape[0])]
    elif mode == 'pos':
        print(unit_weights[:20])
        pos_idx = torch.nonzero(unit_weights > 0, as_tuple=True)
        unit_weights[pos_idx] = unit_weights[pos_idx[0][torch.randperm(pos_idx[0].shape[0])]]
        print(unit_weights[:20])
    elif mode == 'neg':
        print(unit_weights[:20])
        num_pos = torch.nonzero(unit_weights > 0).shape[0]
        neg_idx = torch.nonzero(unit_weights < 0, as_tuple=True)
        neg_idx = neg_idx[0][torch.randperm(neg_idx[0].shape[0])][:num_pos]
        unit_weights[neg_idx] = unit_weights[neg_idx[torch.randperm(neg_idx.shape[0])]]
        print(unit_weights[:20])

    states[layername][unit] = torch.reshape(unit_weights, states[layername][unit].shape)

    return states


#   Change the sign of weights in specified channel of layer.
def invert_weights(states, layer, unit):
    states[layer][unit] *= -1

    return states


if __name__ == "__main__":
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    norm_trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])

    batch_size = 9
    model_name = 'alexnet'
    source = 'torchvision'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    neuron_coord = None
    conv_layer = True
    if conv_layer:
        neuron_coord = 6
    module_name = 'features.10'
    unit_id = 25

    extractor = get_extractor(
        model_name=model_name,
        source=source,
        device=device,
        pretrained=True
    )

    # extractor.model.load_state_dict(
    #     ablate_unit(extractor.model.state_dict(), "features.10.weight", unit_id, min=0, max=None)
    # )

    scores = []
    img_folder = r"/home/andre/tuning_curves/inh_ablate/layer11_neuron17/max/exc"
    imgs = [Image.open(os.path.join(img_folder, file)) for file in os.listdir(img_folder)]

    #   Proto load.
    # exc_proto = Image.open(r"/home/andre/evolved_data/alexnet_.features.Conv2d10_unit25/Avg_Exc_Prototype_alexnet_.features.Conv2d10.png")
    # inh_proto = Image.open(r'/home/andre/evolved_data/alexnet_.features.Conv2d10_unit25/Avg_Inh_Prototype_alexnet_.features.Conv2d10.png')

    # exc_proto = np.array(exc_proto, dtype=np.float32)
    # inh_proto = np.array(inh_proto, dtype=np.float32)
    #
    # avg_proto = (exc_proto + inh_proto) / 2
    # avg_proto = exc_proto
    # avg_proto = Image.fromarray(avg_proto.astype('uint8'))
    # avg_proto.save("/home/andre/evolved_data/alexnet_.features.Conv2d10_unit25/mixed_avg_protos.png")
    # imgs = [avg_proto]

    for img in imgs:
        img = norm_trans(img)
        img = torch.unsqueeze(img, 0)
        img = torch.unsqueeze(img, 0)
        print(img.shape)

        features = extractor.extract_features(
            batches=img,
            module_name=module_name,
            flatten_acts=False
        )

        score = None
        if neuron_coord is not None:
            score = features[:, unit_id, neuron_coord, neuron_coord]
        else:
            score = features[:, unit_id]

        scores.append(score)

    mean_score = np.mean(np.array(scores))
    print(mean_score)

    all_ord_list = np.load(os.path.join(f'/home/andre/tuning_curves/intact/layer11_neuron{unit_id}', 'order_list.npy'))
    all_act_list = np.load(os.path.join(f'/home/andre/tuning_curves/intact/layer11_neuron{unit_id}', 'activations_list.npy'))

    plt.scatter(all_ord_list, all_act_list, color='b')
    plt.axhline(mean_score, color='g')
    plt.savefig(f'/home/andre/tuning_curves/intact/inh_ablate_top9_intact_tuning_curve_{unit_id}.png')
    plt.close()