import sys
import os
import copy
sys.path.append(r'/home/andre/evolve-code/circuit_toolkit')
sys.path.append(r'/home/andre/evolve-code/selectivity_codes')
import argparse
import torch
import torchvision
from torchvision import models, transforms
import matplotlib.pyplot as plt
import numpy as np
from thingsvision import get_extractor
from PIL import Image

from circuit_toolkit.insilico_rf_save import get_center_pos_and_rf
from circuit_toolkit.selectivity_utils import normalize, saveTopN, Sobel
from circuit_toolkit.selectivity import batch_selectivity
# from scatterplot import simple_scattplot
from imnet_val import validate_tuning_curve, validate_tuning_curve_thingsvision

from modify_weights import clamp_ablate_unit, clamp_ablate_layer, random_ablate_unit, channel_random_ablate_unit, binarize_unit
from transplant import get_activations


def get_max_stim_acts(extractor, imgs, module_name, selected_neuron=None, neuron_coord=None):
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    norm_trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])

    img_acts = []
    for img in imgs:
        img = norm_trans(img)
        act = get_activations(extractor, img, module_name, neuron_coord, selected_neuron)
        img_acts.append(act)

    return img_acts


def extract_ranks(extractor, topdir, botdir, protodir, antiprotodir, poslucentdir, neglucentdir,
                  layer, selected_layer, selected_neuron, module_name, neuron_coord, ablate, actsdir=None):

    # original_states = copy.deepcopy(extractor.model.state_dict())

    top_imgs = [Image.open(os.path.join(topdir, file)) for file in os.listdir(topdir)]
    bot_imgs = [Image.open(os.path.join(botdir, file)) for file in os.listdir(botdir)]
    proto_imgs = [Image.open(os.path.join(protodir, file)) for file in os.listdir(protodir)]
    antiproto_imgs = [Image.open(os.path.join(antiprotodir, file)) for file in os.listdir(antiprotodir)]

    # poslucent_imgs = [Image.open(os.path.join(poslucentdir, file)) for file in os.listdir(poslucentdir)]
    # neglucent_imgs = [Image.open(os.path.join(neglucentdir, file)) for file in os.listdir(neglucentdir)]

    top_ranks = []
    bot_ranks = []
    proto_ranks = []
    antiproto_ranks = []
    poslucent_ranks = []
    neglucent_ranks = []

    percentages = np.arange(10, 100, 10)
    percentages = np.append(percentages, 99)
    percentages = np.append(percentages, 100)

    iters = 1
    # if inh_abl:
    #     iters = 1
    # else:
    #     iters = percentages.shape[0]

    for i in range(iters):
        if ablate != "":
            min = 0 if ablate == "inh_abl" else None
            max = 0 if ablate == "exc_abl" else None
            extractor.model.load_state_dict(
                clamp_ablate_unit(extractor.model.state_dict(), module_name + '.weight', selected_neuron, min=min, max=max)
            )
        # else:
            # p = percentages[i]
            # extractor.model.load_state_dict(
            #     random_ablate_unit(extractor.model.state_dict(), "features.10.weight", selected_neuron, perc=p / 100)
            # )

        val_acts = None
        if actsdir is None:
            _, _, val_acts, _, _, _ = validate_tuning_curve(
                extractor.model, layer, selected_layer, selected_neuron
            )
            val_acts = np.array(val_acts)
        else:
            val_acts = np.load(actsdir)

        acts = get_max_stim_acts(extractor, top_imgs, module_name, selected_neuron, neuron_coord)
        top_ranks.append([
            torch.nonzero(torch.tensor(val_acts) > torch.tensor(act)).shape[0]
            for act in acts
        ])

        acts = get_max_stim_acts(extractor, bot_imgs, module_name, selected_neuron, neuron_coord)
        bot_ranks.append([
            torch.nonzero(torch.tensor(val_acts) > torch.tensor(act)).shape[0]
            for act in acts
        ])

        acts = get_max_stim_acts(extractor, proto_imgs, module_name, selected_neuron, neuron_coord)
        proto_ranks.append([
            torch.nonzero(torch.tensor(val_acts) > torch.tensor(act)).shape[0]
            for act in acts
        ])

        acts = get_max_stim_acts(extractor, antiproto_imgs, module_name, selected_neuron, neuron_coord)
        antiproto_ranks.append([
            torch.nonzero(torch.tensor(val_acts) > torch.tensor(act)).shape[0]
            for act in acts
        ])

        # acts = get_max_stim_acts(extractor, poslucent_imgs, module_name, selected_neuron, neuron_coord)
        # poslucent_ranks.append([
        #     torch.nonzero(torch.tensor(val_acts) > torch.tensor(act)).shape[0]
        #     for act in acts
        # ])
        #
        # acts = get_max_stim_acts(extractor, neglucent_imgs, module_name, selected_neuron, neuron_coord)
        # neglucent_ranks.append([
        #     torch.nonzero(torch.tensor(val_acts) > torch.tensor(act)).shape[0]
        #     for act in acts
        # ])

    return top_ranks, bot_ranks, proto_ranks, antiproto_ranks, poslucent_ranks, neglucent_ranks

#   For resnet, take batchnorm layer as inputs and use torch.clamp(inputs, min=0, max=None) to simulate relu
def extract_max_stim_inputs(extractor, topdir, botdir, protodir, antiprotodir, poslucentdir, neglucentdir,
                            selected_neuron, module_name, split=False, model_name=''):
    savedir = "/home/andre/max_stim_inputs"

    top_imgs = [Image.open(os.path.join(topdir, file)) for file in os.listdir(topdir)]
    bot_imgs = [Image.open(os.path.join(botdir, file)) for file in os.listdir(botdir)]
    proto_imgs = [Image.open(os.path.join(protodir, file)) for file in os.listdir(protodir)]
    antiproto_imgs = [Image.open(os.path.join(antiprotodir, file)) for file in os.listdir(antiprotodir)]

    poslucent_imgs = [Image.open(os.path.join(poslucentdir, file)) for file in os.listdir(poslucentdir)]
    neglucent_imgs = [Image.open(os.path.join(neglucentdir, file)) for file in os.listdir(neglucentdir)]

    top_inputs = torch.tensor(get_max_stim_acts(extractor, top_imgs, module_name))[:, :, :, 2:5, 2:5]
    bot_inputs = torch.tensor(get_max_stim_acts(extractor, bot_imgs, module_name))[:, :, :, 2:5, 2:5]
    proto_inputs = torch.tensor(get_max_stim_acts(extractor, proto_imgs, module_name))[:, :, :, 2:5, 2:5]
    antiproto_inputs = torch.tensor(get_max_stim_acts(extractor, antiproto_imgs, module_name))[:, :, :, 2:5, 2:5]

    poslucent_inputs = torch.tensor(get_max_stim_acts(extractor, poslucent_imgs, module_name))[:, :, :, 2:5, 2:5]
    neglucent_inputs = torch.tensor(get_max_stim_acts(extractor, neglucent_imgs, module_name))[:, :, :, 2:5, 2:5]

    top_inputs = torch.flatten(torch.clamp(top_inputs, min=0, max=None), start_dim=1, end_dim=-1)
    bot_inputs = torch.flatten(torch.clamp(bot_inputs, min=0, max=None), start_dim=1, end_dim=-1)
    proto_inputs = torch.flatten(torch.clamp(proto_inputs, min=0, max=None), start_dim=1, end_dim=-1)
    antiproto_inputs = torch.flatten(torch.clamp(antiproto_inputs, min=0, max=None), start_dim=1, end_dim=-1)

    poslucent_inputs = torch.flatten(torch.clamp(poslucent_inputs, min=0, max=None), start_dim=1, end_dim=-1)
    neglucent_inputs = torch.flatten(torch.clamp(neglucent_inputs, min=0, max=None), start_dim=1, end_dim=-1)

    if split:
        weights = extractor.model.state_dict()[module_name + '.weight'][selected_neuron]
        weights = torch.flatten(weights).cpu()

        #   Positive-weighted inputs only.
        top_split_inputs = top_inputs[:, torch.nonzero(weights > 0)]
        bot_split_inputs = bot_inputs[:, torch.nonzero(weights > 0)]
        proto_split_inputs = proto_inputs[:, torch.nonzero(weights > 0)]
        antiproto_split_inputs = antiproto_inputs[:, torch.nonzero(weights > 0)]
        poslucent_split_inputs = poslucent_inputs[:, torch.nonzero(weights > 0)]
        neglucent_split_inputs = neglucent_inputs[:, torch.nonzero(weights > 0)]

        top_norms = torch.linalg.vector_norm(top_split_inputs, ord=1, dim=1)
        bot_norms = torch.linalg.vector_norm(bot_split_inputs, ord=1, dim=1)
        proto_norms = torch.linalg.vector_norm(proto_split_inputs, ord=1, dim=1)
        antiproto_norms = torch.linalg.vector_norm(antiproto_split_inputs, ord=1, dim=1)

        poslucent_norms = torch.linalg.vector_norm(poslucent_split_inputs, ord=1, dim=1)
        neglucent_norms = torch.linalg.vector_norm(neglucent_split_inputs, ord=1, dim=1)

        np.save(os.path.join(savedir, 'split/pos', model_name, f"layer{selected_layer}_unit{selected_neuron}_top_norms.npy"),
                top_norms.numpy())
        np.save(os.path.join(savedir, 'split/pos', model_name, f"layer{selected_layer}_unit{selected_neuron}_proto_norms.npy"),
                proto_norms.numpy())
        np.save(os.path.join(savedir, 'split/pos', model_name, f"layer{selected_layer}_unit{selected_neuron}_bot_norms.npy"),
                bot_norms.numpy())
        np.save(os.path.join(savedir, 'split/pos', model_name, f"layer{selected_layer}_unit{selected_neuron}_antiproto_norms.npy"),
                antiproto_norms.numpy())

        np.save(os.path.join(savedir, 'split/pos', model_name, f"layer{selected_layer}_unit{selected_neuron}_poslucent_norms.npy"),
                poslucent_norms.numpy())
        np.save(os.path.join(savedir, 'split/pos', model_name, f"layer{selected_layer}_unit{selected_neuron}_neglucent_norms.npy"),
                neglucent_norms.numpy())

        #   Negative only.
        top_split_inputs = top_inputs[:, torch.nonzero(weights < 0)]
        bot_split_inputs = bot_inputs[:, torch.nonzero(weights < 0)]
        proto_split_inputs = proto_inputs[:, torch.nonzero(weights < 0)]
        antiproto_split_inputs = antiproto_inputs[:, torch.nonzero(weights < 0)]
        poslucent_split_inputs = poslucent_inputs[:, torch.nonzero(weights < 0)]
        neglucent_split_inputs = neglucent_inputs[:, torch.nonzero(weights < 0)]

        top_norms = torch.linalg.vector_norm(top_split_inputs, ord=1, dim=1)
        bot_norms = torch.linalg.vector_norm(bot_split_inputs, ord=1, dim=1)
        proto_norms = torch.linalg.vector_norm(proto_split_inputs, ord=1, dim=1)
        antiproto_norms = torch.linalg.vector_norm(antiproto_split_inputs, ord=1, dim=1)

        poslucent_norms = torch.linalg.vector_norm(poslucent_split_inputs, ord=1, dim=1)
        neglucent_norms = torch.linalg.vector_norm(neglucent_split_inputs, ord=1, dim=1)

        np.save(os.path.join(savedir, 'split/neg',model_name, f"layer{selected_layer}_unit{selected_neuron}_top_norms.npy"),
                top_norms.numpy())
        np.save(os.path.join(savedir, 'split/neg',model_name, f"layer{selected_layer}_unit{selected_neuron}_proto_norms.npy"),
                proto_norms.numpy())
        np.save(os.path.join(savedir, 'split/neg',model_name, f"layer{selected_layer}_unit{selected_neuron}_bot_norms.npy"),
                bot_norms.numpy())
        np.save(os.path.join(savedir, 'split/neg',model_name, f"layer{selected_layer}_unit{selected_neuron}_antiproto_norms.npy"),
                antiproto_norms.numpy())

        np.save(os.path.join(savedir, 'split/neg', model_name, f"layer{selected_layer}_unit{selected_neuron}_poslucent_norms.npy"),
                poslucent_norms.numpy())
        np.save(os.path.join(savedir, 'split/neg', model_name, f"layer{selected_layer}_unit{selected_neuron}_neglucent_norms.npy"),
                neglucent_norms.numpy())

    else:
        top_norms = torch.linalg.vector_norm(top_inputs, ord=1, dim=1)
        bot_norms = torch.linalg.vector_norm(bot_inputs, ord=1, dim=1)
        proto_norms = torch.linalg.vector_norm(proto_inputs, ord=1, dim=1)
        antiproto_norms = torch.linalg.vector_norm(antiproto_inputs, ord=1, dim=1)

        poslucent_norms = torch.linalg.vector_norm(poslucent_inputs, ord=1, dim=1)
        neglucent_norms = torch.linalg.vector_norm(neglucent_inputs, ord=1, dim=1)

        np.save(os.path.join(savedir, 'all', model_name, f"layer{selected_layer}_unit{selected_neuron}_top_norms.npy"),
                top_norms.numpy())
        np.save(os.path.join(savedir, 'all', model_name, f"layer{selected_layer}_unit{selected_neuron}_proto_norms.npy"),
                proto_norms.numpy())
        np.save(os.path.join(savedir, 'all', model_name, f"layer{selected_layer}_unit{selected_neuron}_bot_norms.npy"),
                bot_norms.numpy())
        np.save(os.path.join(savedir, 'all', model_name, f"layer{selected_layer}_unit{selected_neuron}_antiproto_norms.npy"),
                antiproto_norms.numpy())

        np.save(os.path.join(savedir, 'all', model_name, f"layer{selected_layer}_unit{selected_neuron}_poslucent_norms.npy"),
                poslucent_norms.numpy())
        np.save(os.path.join(savedir, 'all', model_name, f"layer{selected_layer}_unit{selected_neuron}_neglucent_norms.npy"),
                neglucent_norms.numpy())


def get_all_stim_acts(extractor, topdir, botdir, protodir, antiprotodir, poslucentdir, neglucentdir, module_name, neuron_coord):
    top_acts = None
    bot_acts = None
    proto_acts = None
    antiproto_acts = None
    poslucent_acts = None
    neglucent_acts = None

    if topdir is not None:
        top_imgs = [Image.open(os.path.join(topdir, file)) for file in os.listdir(topdir)]
        top_acts = torch.tensor(get_max_stim_acts(extractor, top_imgs, module_name, selected_neuron=None, neuron_coord=neuron_coord))
    
    if botdir is not None:
        bot_imgs = [Image.open(os.path.join(botdir, file)) for file in os.listdir(botdir)]
        bot_acts = torch.tensor(get_max_stim_acts(extractor, bot_imgs, module_name, selected_neuron=None, neuron_coord=neuron_coord))

    if protodir is not None:
        proto_imgs = [Image.open(os.path.join(protodir, file)) for file in os.listdir(protodir)]
        proto_acts = torch.tensor(get_max_stim_acts(extractor, proto_imgs, module_name, selected_neuron=None, neuron_coord=neuron_coord))

    if antiprotodir is not None:
        antiproto_imgs = [Image.open(os.path.join(antiprotodir, file)) for file in os.listdir(antiprotodir)]
        antiproto_acts = torch.tensor(get_max_stim_acts(extractor, antiproto_imgs, module_name, selected_neuron=None, neuron_coord=neuron_coord))

    if poslucentdir is not None:
        poslucent_imgs = [Image.open(os.path.join(poslucentdir, file)) for file in os.listdir(poslucentdir)]
        poslucent_acts = torch.tensor(get_max_stim_acts(extractor, poslucent_imgs, module_name, selected_neuron=None, neuron_coord=neuron_coord))

    if neglucentdir is not None:
        neglucent_imgs = [Image.open(os.path.join(neglucentdir, file)) for file in os.listdir(neglucentdir)]
        neglucent_acts = torch.tensor(get_max_stim_acts(extractor, neglucent_imgs, module_name, selected_neuron=None, neuron_coord=neuron_coord))

    return top_acts, bot_acts, proto_acts, antiproto_acts, poslucent_acts, neglucent_acts


def compare_tuning_curves(model, savedir, layer, selected_layer, selected_neuron, module_name,
                          ablate, extractor=None, neuron_coord=None):
    #   Intact tuning curve.
    if ablate != "":
        min = 0 if ablate == "inh_abl" else None
        max = 0 if ablate == "exc_abl" else None

        if selected_neuron is None:
            model.load_state_dict(clamp_ablate_unit(model.state_dict(), module_name + '.weight', selected_neuron, min=min, max=max))
        else:
            model.load_state_dict(clamp_ablate_layer(model.state_dict(), module_name + '.weight', min=min, max=max))

    all_images = act_list = unrolled_act = all_act_list = all_ord_sorted = None
    if extractor is None:
        all_images, act_list, unrolled_act, all_act_list, all_ord_sorted, _ = \
            validate_tuning_curve(model, layer, selected_layer, selected_neuron)
    else:
        all_images, act_list, unrolled_acts, all_act_list, all_ord_sorted, _ = \
            validate_tuning_curve_thingsvision(extractor, module_name, selected_neuron, neuron_coord, sort_acts=False)

    unrolled_acts = np.array(unrolled_acts)
    if len(unrolled_acts.shape) == 2:
        unrolled_acts = np.transpose(unrolled_acts, (1, 0))

        for i in range(unrolled_acts.shape[0]):
            selected_neuron = i

            unrolled_act = unrolled_acts[i].tolist()
            all_ord_list = np.arange(len(all_images)).tolist()
            all_act_list, all_ord_sorted = zip(*sorted(zip(unrolled_act, all_ord_list), reverse=True))

            if ablate != "":
                saveTopN(all_images, all_ord_sorted, f"{layer}_neuron{selected_neuron}", path=savedir)

                np.save(os.path.join(savedir, f"{layer}_unit{selected_neuron}_{ablate}_unrolled_act.npy"),
                        np.array(unrolled_act))
                np.save(os.path.join(savedir, f"{layer}_unit{selected_neuron}_{ablate}_all_act_list.npy"),
                        np.array(list(all_act_list)))
                np.save(os.path.join(savedir, f"{layer}_unit{selected_neuron}_{ablate}_all_ord_sorted.npy"),
                        np.array(list(all_ord_sorted)))
            else:
                saveTopN(all_images, all_ord_sorted, f"{layer}_neuron{selected_neuron}", path=savedir)

                np.save(os.path.join(savedir, f"{layer}_unit{selected_neuron}_intact_unrolled_act.npy"),
                        np.array(unrolled_act))
                np.save(os.path.join(savedir, f"{layer}_unit{selected_neuron}_intact_all_act_list.npy"),
                        np.array(list(all_act_list)))
                np.save(os.path.join(savedir, f"{layer}_unit{selected_neuron}_intact_all_ord_sorted.npy"),
                        np.array(list(all_ord_sorted)))


def get_max_stim_dirs(network, layer, selected_neuron, ablate_type="", mask="/no_mask", ablated="", load_states=False):

    model_name = neuron_coord = module_name = states = None
    curvedir = ranksdir = actsdir = None
    topdir = botdir = protodir = antiprotodir = poslucentdir = neglucentdir = None

    if network == 'alexnet':
        model_name = 'alexnet'
        neuron_coord = 6
        module_name = 'features.10'
        if ablate_type != "":
            curvedir = f'/home/andrelongon/Documents/inhibition_code/tuning_curves/alexnet/{layer}/{ablate_type}'
            ranksdir = f'/home/andrelongon/Documents/inhibition_code/rank_data/{ablate_type}/alexnet_trained'

            actsdir = os.path.join(curvedir, f'{layer}_unit{selected_neuron}_{ablate_type}_unrolled_act.npy')
        else:
            curvedir = f'/home/andrelongon/Documents/inhibition_code/tuning_curves/alexnet/{layer}/intact'
            ranksdir = '/home/andrelongon/Documents/inhibition_code/rank_data/intact/alexnet_trained'

            actsdir = os.path.join(curvedir, f'{layer}_unit{selected_neuron}_intact_unrolled_act.npy')

        topdir = f'/home/andrelongon/Documents/inhibition_code/tuning_curves/alexnet/{layer}/intact/{layer}_neuron{selected_neuron}/max/exc'
        botdir = f'/home/andrelongon/Documents/inhibition_code/tuning_curves/alexnet/{layer}/intact/{layer}_neuron{selected_neuron}/max/inh'
        protodir = f'/home/andrelongon/Documents/inhibition_code/evolved_data{ablated}/alexnet_.{layer}_unit{selected_neuron}/exc{mask}'
        antiprotodir = f'/home/andrelongon/Documents/inhibition_code/evolved_data{ablated}/alexnet_.{layer}_unit{selected_neuron}/inh{mask}'

        poslucentdir = f"/home/andrelongon/Documents/inhibition_code/lucent_imgs/alexnet_trained/{selected_neuron}/pos"
        neglucentdir = f"/home/andrelongon/Documents/inhibition_code/lucent_imgs/alexnet_trained/{selected_neuron}/neg"

    elif network == 'alexnet-untrained':
        model_name = 'alexnet'
        neuron_coord = 27
        module_name = 'features.0'
        states = torch.load("/home/andre/tuning_curves/untrained_alexnet/random_weights.pth")

        if ablate_type != "":
            curvedir = '/home/andre/tuning_curves/untrained_alexnet/inh_abl'
            ranksdir = '/home/andre/rank_data/inh_abl/alexnet_untrained'

            actsdir = os.path.join(curvedir, f'layer{selected_layer}_unit{selected_neuron}_inh_abl_unrolled_act.npy')
        else:
            curvedir = '/home/andre/tuning_curves/untrained_alexnet'
            ranksdir = '/home/andre/rank_data/intact/alexnet_untrained'

            actsdir = os.path.join(curvedir, f'layer{selected_layer}_unit{selected_neuron}_intact_unrolled_act.npy')

        topdir = f'/home/andre/tuning_curves/untrained_alexnet/layer{selected_layer}_neuron{selected_neuron}/max/exc'
        botdir = f'/home/andre/tuning_curves/untrained_alexnet/layer{selected_layer}_neuron{selected_neuron}/max/inh'
        protodir = f'/home/andre/evolved_data/alexnet_untrained_.features.Conv2d{selected_layer-1}_unit{selected_neuron}/exc{mask}'
        antiprotodir = f'/home/andre/evolved_data/alexnet_untrained_.features.Conv2d{selected_layer-1}_unit{selected_neuron}/inh{mask}'

        poslucentdir = f"/home/andre/lucent_imgs/alexnet_untrained/{selected_neuron}/pos"
        neglucentdir = f"/home/andre/lucent_imgs/alexnet_untrained/{selected_neuron}/neg"

    elif network == 'resnet18_trained':
        model_name = 'resnet18'
        neuron_coord = 7
        module_name = 'layer3.1.bn2'

        if ablate_type != "":
            curvedir = f'/home/andrelongon/Documents/inhibition_code/tuning_curves/resnet18/{layer}/{ablate_type}'
            ranksdir = f'/home/andrelongon/Documents/inhibition_code/rank_data/{ablate_type}/resnet18_trained'

            actsdir = os.path.join(curvedir, f'{layer}_unit{selected_neuron}_{ablate_type}_unrolled_act.npy')
        else:
            curvedir = f'/home/andrelongon/Documents/inhibition_code/tuning_curves/resnet18/{layer}/intact'
            ranksdir = '/home/andrelongon/Documents/inhibition_code/rank_data/intact/resnet18_trained'

            actsdir = os.path.join(curvedir, f'{layer}_unit{selected_neuron}_intact_unrolled_act.npy')

        topdir = f'/home/andrelongon/Documents/inhibition_code/tuning_curves/resnet18/{layer}/intact/{layer}_neuron{selected_neuron}/max/exc'
        botdir = f'/home/andrelongon/Documents/inhibition_code/tuning_curves/resnet18/{layer}/intact/{layer}_neuron{selected_neuron}/max/inh'
        protodir = f'/home/andrelongon/Documents/inhibition_code/evolved_data{ablated}/resnet18_.{layer}_unit{selected_neuron}/exc{mask}'
        antiprotodir = f'/home/andrelongon/Documents/inhibition_code/evolved_data{ablated}/resnet18_.{layer}_unit{selected_neuron}/inh{mask}'

        poslucentdir = f"/home/andre/lucent_imgs/resnet18_trained/{selected_neuron}/pos"
        neglucentdir = f"/home/andre/lucent_imgs/resnet18_trained/{selected_neuron}/neg"

    elif network == 'resnet18_untrained':
        model_name = 'resnet18'
        neuron_coord = 56
        module_name = 'conv1'
        states = torch.load("/home/andre/tuning_curves/untrained_resnet18/random_weights.pth", map_location="cuda:0")

        if ablate_type != "":
            curvedir = '/home/andre/tuning_curves/untrained_resnet18/inh_abl'
            ranksdir = '/home/andre/rank_data/inh_abl/resnet18_untrained'

            actsdir = os.path.join(curvedir, f'layer{selected_layer}_unit{selected_neuron}_inh_abl_unrolled_act.npy')
        else:
            curvedir = '/home/andre/tuning_curves/untrained_resnet18/intact'
            ranksdir = '/home/andre/rank_data/intact/resnet18_untrained'

            actsdir = os.path.join(curvedir, f'layer{selected_layer}_unit{selected_neuron}_intact_unrolled_act.npy')

        topdir = f'/home/andre/tuning_curves/untrained_resnet18/intact/layer{selected_layer}_neuron{selected_neuron}/max/exc'
        botdir = f'/home/andre/tuning_curves/untrained_resnet18/intact/layer{selected_layer}_neuron{selected_neuron}/max/inh'
        protodir = f'/home/andre/evolved_data/resnet18_untrained_.Conv2dconv1_unit{selected_neuron}/exc{mask}'
        antiprotodir = f'/home/andre/evolved_data/resnet18_untrained_.Conv2dconv1_unit{selected_neuron}/inh{mask}'

        poslucentdir = f"/home/andre/lucent_imgs/resnet18_untrained/{selected_neuron}/pos"
        neglucentdir = f"/home/andre/lucent_imgs/resnet18_untrained/{selected_neuron}/neg"

    elif network == 'resnet18_robust':
        model_name = 'resnet18'
        neuron_coord = 14
        module_name = 'layer2.1.bn1'
        if load_states:
            states = torch.load("/home/andrelongon/Documents/inhibition_code/weights/resnet-18-l2-eps3.pt", map_location="cuda:0")

        if ablate_type != "":
            curvedir = f'/home/andrelongon/Documents/inhibition_code/tuning_curves/resnet18_robust/{layer}/{ablate_type}'
            ranksdir = f'/home/andrelongon/Documents/inhibition_code/rank_data/{ablate_type}/resnet18_robust'

            actsdir = os.path.join(curvedir, f'{layer}_unit{selected_neuron}_{ablate_type}_unrolled_act.npy')
        else:
            curvedir = f'/home/andrelongon/Documents/inhibition_code/tuning_curves/resnet18_robust/{layer}/intact'
            ranksdir = '/home/andrelongon/Documents/inhibition_code/rank_data/intact/resnet18_robust'

            actsdir = os.path.join(curvedir, f'{layer}_unit{selected_neuron}_intact_unrolled_act.npy')

        topdir = f'/home/andrelongon/Documents/inhibition_code/tuning_curves/resnet18_robust/{layer}/intact/{layer}_neuron{selected_neuron}/max/exc'
        botdir = f'/home/andrelongon/Documents/inhibition_code/tuning_curves/resnet18_robust/{layer}/intact/{layer}_neuron{selected_neuron}/max/inh'
        protodir = f'/home/andrelongon/Documents/inhibition_code/evolved_data/resnet18_robust_.{layer}_unit{selected_neuron}/exc{mask}'
        antiprotodir = f'/home/andrelongon/Documents/inhibition_code/evolved_data/resnet18_robust_.{layer}_unit{selected_neuron}/inh{mask}'

        poslucentdir = f"/home/andre/lucent_imgs/resnet18_trained/{selected_neuron}/pos"
        neglucentdir = f"/home/andre/lucent_imgs/resnet18_trained/{selected_neuron}/neg"

    elif network == 'cornet-s':
        model_name = 'cornet-s'
        neuron_coord = None
        if layer == 'IT.conv_input' or layer == 'IT.norm1_0':
            neuron_coord = 7
        elif layer == 'IT.norm1_1' or layer == 'IT.norm2_0':
            neuron_coord = 3
        module_name = 'IT.norm2_1'

        if ablate_type != "":
            curvedir = f'/home/andrelongon/Documents/inhibition_code/tuning_curves/cornet-s/{layer}/{ablate_type}'
            ranksdir = f'/home/andrelongon/Documents/inhibition_code/rank_data/{ablate_type}/cornet-s'

            actsdir = os.path.join(curvedir, f'{layer}_unit{selected_neuron}_{ablate_type}_unrolled_act.npy')
        else:
            curvedir = f'/home/andrelongon/Documents/inhibition_code/tuning_curves/cornet-s/{layer}/intact'
            ranksdir = '/home/andrelongon/Documents/inhibition_code/rank_data/intact/cornet-s'

            actsdir = os.path.join(curvedir, f'{layer}_unit{selected_neuron}_intact_unrolled_act.npy')

        topdir = f'/home/andrelongon/Documents/inhibition_code/tuning_curves/cornet-s/{layer}/intact/{layer}_neuron{selected_neuron}/max/exc'
        botdir = f'/home/andrelongon/Documents/inhibition_code/tuning_curves/cornet-s/{layer}/intact/{layer}_neuron{selected_neuron}/max/inh'
        protodir = f'/home/andrelongon/Documents/inhibition_code/evolved_data/cornet-s_.{layer}_unit{selected_neuron}/exc{mask}'
        antiprotodir = f'/home/andrelongon/Documents/inhibition_code/evolved_data/cornet-s_.{layer}_unit{selected_neuron}/inh{mask}'

        poslucentdir = f"/home/andre/lucent_imgs/cornet-s/{selected_neuron}/pos"
        neglucentdir = f"/home/andre/lucent_imgs/cornet-s/{selected_neuron}/neg"

    elif network == 'resnet152_v1':
        model_name = 'resnet152'

    elif network == 'resnet50_v1':
        model_name = 'resnet50'

    elif network == 'resnext101_wsl':
        model_name = 'resnext101_32x8d'
        if load_states:
            states = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x8d_wsl').state_dict()

    #   TODO:  Simplify above code to make use of this catchall.
    if ablate_type != "":
        curvedir = f'/home/andrelongon/Documents/inhibition_code/tuning_curves/{network}/{layer}/{ablate_type}'
        ranksdir = f'/home/andrelongon/Documents/inhibition_code/rank_data/{ablate_type}/{network}'

        actsdir = os.path.join(curvedir, f'{layer}_unit{selected_neuron}_{ablate_type}_unrolled_act.npy')
    else:
        curvedir = f'/home/andrelongon/Documents/inhibition_code/tuning_curves/{network}/{layer}/intact'
        ranksdir = f'/home/andrelongon/Documents/inhibition_code/rank_data/intact/{network}'

        actsdir = os.path.join(curvedir, f'{layer}_unit{selected_neuron}_intact_unrolled_act.npy')

    topdir = f'/home/andrelongon/Documents/inhibition_code/tuning_curves/{network}/{layer}/intact/{layer}_neuron{selected_neuron}/max/exc'
    botdir = f'/home/andrelongon/Documents/inhibition_code/tuning_curves/{network}/{layer}/intact/{layer}_neuron{selected_neuron}/max/inh'
    protodir = f'/home/andrelongon/Documents/inhibition_code/evolved_data/{network}_.{layer}_unit{selected_neuron}/exc{mask}'
    antiprotodir = f'/home/andrelongon/Documents/inhibition_code/evolved_data/{network}_.{layer}_unit{selected_neuron}/inh{mask}'

    poslucentdir = f"/home/andre/lucent_imgs/{network}/{selected_neuron}/pos"
    neglucentdir = f"/home/andre/lucent_imgs/{network}/{selected_neuron}/neg"

    return model_name, neuron_coord, module_name, states, curvedir, ranksdir, actsdir, topdir, botdir, protodir, antiprotodir, poslucentdir, neglucentdir


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str, required=False)
    parser.add_argument('--neuron', type=int, required=False)
    parser.add_argument('--layer', type=str)
    parser.add_argument('--selected_layer', type=int, required=False)
    # parser.add_argument('--savedir', type=str)
    # parser.add_argument('--protodir', type=str)
    parser.add_argument('--ablate_type', type=str, default="")
    args = parser.parse_args()

    layer = args.layer
    selected_layer = args.selected_layer
    selected_neuron = args.neuron

    model_name, neuron_coord, module_name, states, \
    curvedir, ranksdir, actsdir, \
    topdir, botdir, protodir, antiprotodir, poslucentdir, neglucentdir = get_max_stim_dirs(args.network, args.layer, args.neuron, ablate_type=args.ablate_type, load_states=True)

    source_name = 'custom' if model_name == 'cornet-s' else 'torchvision'
    model_params = {'weights': 'IMAGENET1K_V1'} if model_name == 'resnet152' or model_name == 'resnet50' else None

    extractor = get_extractor(
        model_name=model_name,
        source=source_name,
        device='cuda',
        pretrained=True,
        model_parameters=model_params
    )
    
    states = torch.load(f"/home/andrelongon/Documents/inhibition_code/weights/overlap_finetune/alexnet_features.10_1_3ep.pth")
    if states is not None:
        extractor.model.load_state_dict(states)

    # print(extractor.show_model())
    # exit()

    # extract_max_stim_inputs(extractor, topdir, botdir, protodir, antiprotodir, poslucentdir, neglucentdir,
    #                         selected_neuron, 'layer4.1.bn1', split=True, model_name=args.network)
    # exit()

    compare_tuning_curves(extractor.model, curvedir, layer, selected_layer, selected_neuron, args.layer,
                          ablate=args.ablate_type, extractor=extractor, neuron_coord=neuron_coord)

    # top_ranks, bot_ranks, proto_ranks, antiproto_ranks, poslucent_ranks, neglucent_ranks = \
    #     extract_ranks(extractor, topdir, botdir, protodir, antiprotodir, poslucentdir, neglucentdir,
    #                   layer, selected_layer, selected_neuron, module_name, neuron_coord, ablate=args.ablate_type, actsdir=actsdir)

    # np.save(os.path.join(ranksdir, f"layer{selected_layer}_unit{selected_neuron}_top9_ranks.npy"), np.array(top_ranks))
    # np.save(os.path.join(ranksdir, f"layer{selected_layer}_unit{selected_neuron}_bot9_ranks.npy"), np.array(bot_ranks))
    # np.save(os.path.join(ranksdir, f"layer{selected_layer}_unit{selected_neuron}_proto_ranks.npy"),
    #         np.array(proto_ranks))
    # np.save(os.path.join(ranksdir, f"layer{selected_layer}_unit{selected_neuron}_antiproto_ranks.npy"),
    #         np.array(antiproto_ranks))

    # np.save(os.path.join(ranksdir, f"layer{selected_layer}_unit{selected_neuron}_poslucent_ranks.npy"), np.array(poslucent_ranks))
    # np.save(os.path.join(ranksdir, f"layer{selected_layer}_unit{selected_neuron}_neglucent_ranks.npy"), np.array(neglucent_ranks))


    # compare_tuning_curves(extractor.model, curvedir, layer, selected_layer, selected_neuron, inh_abl=args.inh_abl)

    # weights = torch.flatten(extractor.model.state_dict()['features.10.weight'][selected_neuron])
    # inh_perc = (torch.nonzero(weights < 0).shape[0] / weights.shape[0]) * 100
    # trials = 4
    # rank_trials = []
    # for i in range(trials):
    #     rank_trials.append(prototype_tuning_curves(extractor, protodir, savedir, layer, selected_layer, selected_neuron))
    #
    # np.save(os.path.join(savedir, f"unit{selected_neuron}_rand_abl_ranks.npy"), np.array(rank_trials))
    # rank_trials = np.mean(np.array(rank_trials), axis=0)
    # percentages = np.arange(10, 100, 10)
    # percentages = np.append(percentages, 99)
    # percentages = np.append(percentages, 100)

    # inh_ranks = []
    # for i in range(trials):
    #     inh_ranks.append(extract_ranks(extractor, protodir, savedir, layer, selected_layer, selected_neuron, inh_abl=True))

    # top_ranks, bot_ranks, proto_ranks, antiproto_ranks = extract_ranks(extractor, topdir, botdir, protodir, antiprotodir,
    #                                                                    layer, selected_layer, selected_neuron, module_name,
    #                                                                    inh_abl=args.inh_abl, actsdir=actsdir)
    #
    # np.save(os.path.join(ranksdir, f"layer{selected_layer}_unit{selected_neuron}_top9_ranks.npy"), np.array(top_ranks))
    # np.save(os.path.join(ranksdir, f"layer{selected_layer}_unit{selected_neuron}_bot9_ranks.npy"), np.array(bot_ranks))
    # np.save(os.path.join(ranksdir, f"layer{selected_layer}_unit{selected_neuron}_proto_ranks.npy"), np.array(proto_ranks))
    # np.save(os.path.join(ranksdir, f"layer{selected_layer}_unit{selected_neuron}_antiproto_ranks.npy"), np.array(antiproto_ranks))
    # inh_rank = np.mean(np.array(inh_ranks))
    #
    # plt.scatter(percentages, rank_trials, color='b', label='Rand Abl')
    # plt.plot(percentages, rank_trials, color='b')
    # plt.scatter(np.array([inh_perc]), np.array(inh_rank), color='r', label='Inh Abl')
    # plt.legend(loc='lower left')
    # plt.xlabel('Percent Ablated')
    # plt.ylabel('Tuning Curve Rank')
    # plt.title(f'Unit {selected_neuron} {args.network} Ablated Rank')
    # plt.savefig(os.path.join(savedir, f"unit{selected_neuron}_abl_antiproto_ranks.png"))