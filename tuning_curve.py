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

from insilico_RF_save import get_center_pos_and_rf
from utils import normalize, saveTopN, Sobel
from selectivity import batch_selectivity
from scatterplot import simple_scattplot
from imnet_val import validate_tuning_curve, validate_tuning_curve_thingsvision

from modify_weights import clamp_ablate_unit, random_ablate_unit, channel_random_ablate_unit, binarize_unit
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
                  layer, selected_layer, selected_neuron, module_name, neuron_coord, ablate=False, actsdir=None):

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
        if ablate:
            extractor.model.load_state_dict(
                clamp_ablate_unit(extractor.model.state_dict(), module_name + '.weight', selected_neuron, min=0, max=None)
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
        weights = extractor.model.state_dict()['layer4.1.conv2.weight'][selected_neuron]
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


def compare_tuning_curves(model, savedir, layer, selected_layer, selected_neuron, module_name,
                          ablate=True, extractor=None, neuron_coord=None):
    #   Intact tuning curve.
    if ablate:
        model.load_state_dict(clamp_ablate_unit(model.state_dict(), module_name + '.weight', selected_neuron, min=0, max=None))

    all_images = act_list = unrolled_act = all_act_list = all_ord_sorted = None
    if extractor is None:
        all_images, act_list, unrolled_act, all_act_list, all_ord_sorted, _ = \
            validate_tuning_curve(model, layer, selected_layer, selected_neuron)
    else:
        all_images, act_list, unrolled_act, all_act_list, all_ord_sorted, _ = \
            validate_tuning_curve_thingsvision(extractor, module_name, selected_neuron, neuron_coord)

    if ablate:
        # saveTopN(all_images, all_ord_sorted, f"layer{selected_layer}_neuron{selected_neuron}", path=savedir)

        np.save(os.path.join(savedir, f"layer{selected_layer}_unit{selected_neuron}_inh_abl_unrolled_act.npy"),
                np.array(unrolled_act))
        np.save(os.path.join(savedir, f"layer{selected_layer}_unit{selected_neuron}_inh_abl_all_act_list.npy"),
                np.array(list(all_act_list)))
        np.save(os.path.join(savedir, f"layer{selected_layer}_unit{selected_neuron}_inh_abl_all_ord_sorted.npy"),
                np.array(list(all_ord_sorted)))
    else:
        saveTopN(all_images, all_ord_sorted, f"layer{selected_layer}_neuron{selected_neuron}", path=savedir)

        np.save(os.path.join(savedir, f"layer{selected_layer}_unit{selected_neuron}_intact_unrolled_act.npy"),
                np.array(unrolled_act))
        np.save(os.path.join(savedir, f"layer{selected_layer}_unit{selected_neuron}_intact_all_act_list.npy"),
                np.array(list(all_act_list)))
        np.save(os.path.join(savedir, f"layer{selected_layer}_unit{selected_neuron}_intact_all_ord_sorted.npy"),
                np.array(list(all_ord_sorted)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str, required=False)
    parser.add_argument('--neuron', type=int)
    parser.add_argument('--layer', type=str, required=False)
    parser.add_argument('--selected_layer', type=int, required=False)
    parser.add_argument('--savedir', type=str)
    parser.add_argument('--protodir', type=str)
    parser.add_argument('--ablate', action='store_true', default=False)
    args = parser.parse_args()

    layer = args.layer
    selected_layer = args.selected_layer
    selected_neuron = args.neuron

    model_name = type = neuron_coord = module_name = states = None
    curvedir = ranksdir = actsdir = None
    topdir = botdir = protodir = antiprotodir = poslucentdir = neglucentdir = None

    if args.network == 'alexnet':
        model_name = 'alexnet'
        neuron_coord = 27
        module_name = 'features.0'
        if args.ablate:
            curvedir = '/home/andre/tuning_curves/alexnet/inh_abl'
            ranksdir = '/home/andre/rank_data/inh_abl/alexnet_trained'

            actsdir = os.path.join(curvedir, f'layer{selected_layer}_unit{selected_neuron}_inh_abl_unrolled_act.npy')
        else:
            curvedir = '/home/andre/tuning_curves/alexnet/intact'
            ranksdir = '/home/andre/rank_data/intact/alexnet_trained'

            actsdir = os.path.join(curvedir, f'layer{selected_layer}_unit{selected_neuron}_intact_unrolled_act.npy')

        topdir = f'/home/andre/tuning_curves/alexnet/intact/layer{selected_layer}_neuron{selected_neuron}/max/exc'
        botdir = f'/home/andre/tuning_curves/alexnet/intact/layer{selected_layer}_neuron{selected_neuron}/max/inh'
        protodir = f'/home/andre/evolved_data/alexnet_.features.Conv2d{selected_layer-1}_unit{selected_neuron}/exc/no_mask'
        antiprotodir = f'/home/andre/evolved_data/alexnet_.features.Conv2d{selected_layer-1}_unit{selected_neuron}/inh/no_mask'

        poslucentdir = f"/home/andre/lucent_imgs/alexnet_trained/{selected_neuron}/pos"
        neglucentdir = f"/home/andre/lucent_imgs/alexnet_trained/{selected_neuron}/neg"

    elif args.network == 'alexnet-untrained':
        model_name = 'alexnet'
        neuron_coord = 27
        module_name = 'features.0'
        states = torch.load("/home/andre/tuning_curves/untrained_alexnet/random_weights.pth")

        if args.ablate:
            curvedir = '/home/andre/tuning_curves/untrained_alexnet/inh_abl'
            ranksdir = '/home/andre/rank_data/inh_abl/alexnet_untrained'

            actsdir = os.path.join(curvedir, f'layer{selected_layer}_unit{selected_neuron}_inh_abl_unrolled_act.npy')
        else:
            curvedir = '/home/andre/tuning_curves/untrained_alexnet'
            ranksdir = '/home/andre/rank_data/intact/alexnet_untrained'

            actsdir = os.path.join(curvedir, f'layer{selected_layer}_unit{selected_neuron}_intact_unrolled_act.npy')

        topdir = f'/home/andre/tuning_curves/untrained_alexnet/layer{selected_layer}_neuron{selected_neuron}/max/exc'
        botdir = f'/home/andre/tuning_curves/untrained_alexnet/layer{selected_layer}_neuron{selected_neuron}/max/inh'
        protodir = f'/home/andre/evolved_data/alexnet_untrained_.features.Conv2d{selected_layer-1}_unit{selected_neuron}/exc/no_mask'
        antiprotodir = f'/home/andre/evolved_data/alexnet_untrained_.features.Conv2d{selected_layer-1}_unit{selected_neuron}/inh/no_mask'

        poslucentdir = f"/home/andre/lucent_imgs/alexnet_untrained/{selected_neuron}/pos"
        neglucentdir = f"/home/andre/lucent_imgs/alexnet_untrained/{selected_neuron}/neg"

    elif args.network == 'resnet18_trained':
        model_name = 'resnet18'
        neuron_coord = 56
        module_name = 'conv1'

        if args.ablate:
            curvedir = '/home/andre/tuning_curves/resnet18/inh_abl'
            ranksdir = '/home/andre/rank_data/inh_abl/resnet18_trained'

            actsdir = os.path.join(curvedir, f'layer{selected_layer}_unit{selected_neuron}_inh_abl_unrolled_act.npy')
        else:
            curvedir = '/home/andre/tuning_curves/resnet18/intact'
            ranksdir = '/home/andre/rank_data/intact/resnet18_trained'

            actsdir = os.path.join(curvedir, f'layer{selected_layer}_unit{selected_neuron}_intact_unrolled_act.npy')

        topdir = f'/home/andre/tuning_curves/resnet18/intact/layer{selected_layer}_neuron{selected_neuron}/max/exc'
        botdir = f'/home/andre/tuning_curves/resnet18/intact/layer{selected_layer}_neuron{selected_neuron}/max/inh'
        protodir = f'/home/andre/evolved_data/resnet18_.Conv2dconv1_unit{selected_neuron}/exc/no_mask'
        antiprotodir = f'/home/andre/evolved_data/resnet18_.Conv2dconv1_unit{selected_neuron}/inh/no_mask'

        poslucentdir = f"/home/andre/lucent_imgs/resnet18_trained/{selected_neuron}/pos"
        neglucentdir = f"/home/andre/lucent_imgs/resnet18_trained/{selected_neuron}/neg"

    elif args.network == 'resnet18_untrained':
        model_name = 'resnet18'
        neuron_coord = 56
        module_name = 'conv1'
        states = torch.load("/home/andre/tuning_curves/untrained_resnet18/random_weights.pth")

        if args.ablate:
            curvedir = '/home/andre/tuning_curves/untrained_resnet18/inh_abl'
            ranksdir = '/home/andre/rank_data/inh_abl/resnet18_untrained'

            actsdir = os.path.join(curvedir, f'layer{selected_layer}_unit{selected_neuron}_inh_abl_unrolled_act.npy')
        else:
            curvedir = '/home/andre/tuning_curves/untrained_resnet18/intact'
            ranksdir = '/home/andre/rank_data/intact/resnet18_untrained'

            actsdir = os.path.join(curvedir, f'layer{selected_layer}_unit{selected_neuron}_intact_unrolled_act.npy')

        topdir = f'/home/andre/tuning_curves/untrained_resnet18/intact/layer{selected_layer}_neuron{selected_neuron}/max/exc'
        botdir = f'/home/andre/tuning_curves/untrained_resnet18/intact/layer{selected_layer}_neuron{selected_neuron}/max/inh'
        protodir = f'/home/andre/evolved_data/resnet18_untrained_.Conv2dconv1_unit{selected_neuron}/exc/no_mask'
        antiprotodir = f'/home/andre/evolved_data/resnet18_untrained_.Conv2dconv1_unit{selected_neuron}/inh/no_mask'

        poslucentdir = f"/home/andre/lucent_imgs/resnet18_untrained/{selected_neuron}/pos"
        neglucentdir = f"/home/andre/lucent_imgs/resnet18_untrained/{selected_neuron}/neg"

    elif args.network == 'resnet18_robust':
        model_name = 'resnet18'
        neuron_coord = 56
        module_name = 'conv1'
        states = torch.load("/home/andre/model_weights/resnet-18-l2-eps3.pt")

        if args.ablate:
            curvedir = '/home/andre/tuning_curves/resnet18_robust/inh_abl'
            ranksdir = '/home/andre/rank_data/inh_abl/resnet18_robust'

            actsdir = os.path.join(curvedir, f'layer{selected_layer}_unit{selected_neuron}_inh_abl_unrolled_act.npy')
        else:
            curvedir = '/home/andre/tuning_curves/resnet18_robust/intact'
            ranksdir = '/home/andre/rank_data/intact/resnet18_robust'

            actsdir = os.path.join(curvedir, f'layer{selected_layer}_unit{selected_neuron}_intact_unrolled_act.npy')

        topdir = f'/home/andre/tuning_curves/resnet18_robust/intact/layer{selected_layer}_neuron{selected_neuron}/max/exc'
        botdir = f'/home/andre/tuning_curves/resnet18_robust/intact/layer{selected_layer}_neuron{selected_neuron}/max/inh'
        protodir = f'/home/andre/evolved_data/resnet18_robust_.Conv2dconv1_unit{selected_neuron}/exc/no_mask'
        antiprotodir = f'/home/andre/evolved_data/resnet18_robust_.Conv2dconv1_unit{selected_neuron}/inh/no_mask'

        poslucentdir = f"/home/andre/lucent_imgs/resnet18_robust/{selected_neuron}/pos"
        neglucentdir = f"/home/andre/lucent_imgs/resnet18_robust/{selected_neuron}/neg"

    extractor = get_extractor(
        model_name=model_name,
        source='torchvision',
        device='cuda',
        pretrained=True
    )

    if states is not None:
        extractor.model.load_state_dict(states)

    # extract_max_stim_inputs(extractor, topdir, botdir, protodir, antiprotodir, poslucentdir, neglucentdir,
    #                         selected_neuron, 'layer4.1.bn1', split=True, model_name=args.network)
    # exit()

    compare_tuning_curves(extractor.model, curvedir, layer, selected_layer, selected_neuron, module_name,
                          ablate=args.ablate, extractor=extractor, neuron_coord=neuron_coord)

    top_ranks, bot_ranks, proto_ranks, antiproto_ranks, poslucent_ranks, neglucent_ranks = \
        extract_ranks(extractor, topdir, botdir, protodir, antiprotodir, poslucentdir, neglucentdir,
                      layer, selected_layer, selected_neuron, module_name, neuron_coord, ablate=args.ablate, actsdir=actsdir)

    np.save(os.path.join(ranksdir, f"layer{selected_layer}_unit{selected_neuron}_top9_ranks.npy"), np.array(top_ranks))
    np.save(os.path.join(ranksdir, f"layer{selected_layer}_unit{selected_neuron}_bot9_ranks.npy"), np.array(bot_ranks))
    np.save(os.path.join(ranksdir, f"layer{selected_layer}_unit{selected_neuron}_proto_ranks.npy"),
            np.array(proto_ranks))
    np.save(os.path.join(ranksdir, f"layer{selected_layer}_unit{selected_neuron}_antiproto_ranks.npy"),
            np.array(antiproto_ranks))

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