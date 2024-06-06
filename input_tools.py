import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from thingsvision import get_extractor, get_extractor_from_model
from PIL import Image
import copy
import torchvision
from torchvision import transforms
import lpips
from scipy.stats import pearsonr, spearmanr, f_oneway
from scipy.spatial.distance import cosine

from circuit_toolkit.insilico_rf_save import get_center_pos_and_rf
from circuit_toolkit.selectivity_utils import normalize

from imnet_val import validate_tuning_curve_thingsvision
from tuning_curve import get_all_stim_acts, get_max_stim_dirs
from modify_weights import clamp_ablate_layer, clamp_ablate_unit
from plot_utils import input_proto_sim_plot, proto_anti_sim_plot
from misc_utils import stim_compare
from train_imnet import AlexNet


def get_layer_max_stim_acts(extractor, topdir, botdir, protodir, antiprotodir, poslucentdir, neglucentdir, module_name, neuron_coord):
    top_intact_acts, bot_intact_acts, proto_intact_acts, antiproto_intact_acts = \
        get_all_stim_acts(extractor, topdir, botdir, protodir, antiprotodir, poslucentdir, neglucentdir, module_name, neuron_coord)

    original_states = copy.deepcopy(extractor.model.state_dict())

    #   TODO:  Perform ablation here and run get_all twice for different ablation conditions.
    extractor.model.load_state_dict(clamp_ablate_layer(extractor.model.state_dict(), module_name + '.weight', min=0, max=None))

    #   po: positive-only (negative weights ablated)
    #   no: negative-only (positive weights ablated)
    top_po_acts, bot_po_acts, proto_po_acts, antiproto_po_acts = get_all_stim_acts(extractor, topdir, botdir, protodir, antiprotodir, poslucentdir, neglucentdir, module_name, neuron_coord)

    extractor.model.load_state_dict(original_states)
    extractor.model.load_state_dict(clamp_ablate_layer(extractor.model.state_dict(), module_name + '.weight', min=None, max=0))

    top_no_acts, bot_no_acts, proto_no_acts, antiproto_no_acts = get_all_stim_acts(extractor, topdir, botdir, protodir, antiprotodir, poslucentdir, neglucentdir, module_name, neuron_coord)

    print("Intact Acts")
    print(torch.mean(top_intact_acts, 0)[0, :10])
    print(torch.mean(proto_intact_acts, 0)[0, :10])
    print("Positive-Only Acts")
    print(torch.mean(top_po_acts, 0)[0, :10])
    print(torch.mean(proto_po_acts, 0)[0, :10])
    print("Negative-Only Acts")
    print(torch.mean(top_no_acts, 0)[0, :10])
    print(torch.mean(proto_no_acts, 0)[0, :10])


#   Same neuron's prototype/antiproto lpips.  Control can be proto/antiproto compare from different random neurons.
def proto_antiproto_compare(metric, model_name, layer, neuron):
    model_id, neuron_coord, module_name, states, \
    curvedir, ranksdir, actsdir, \
    topdir, botdir, protodir, antiprotodir, poslucentdir, neglucentdir = get_max_stim_dirs(model_name, layer, neuron, mask="", ablated="_ablated")

    protos = [Image.open(os.path.join(protodir, file)) for file in os.listdir(protodir) if ".png" in file]
    antiprotos = [Image.open(os.path.join(antiprotodir, file)) for file in os.listdir(antiprotodir) if ".png" in file]

    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    norm_trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])

    proto_anti_d = []
    for i in range(len(protos)):
        #   Use inner for j in range(start=j, stop=len(min_protos)) to perform all comparison permutations.
        for j in range(i, len(protos)):
            proto = norm_trans(protos[i])
            antiproto = norm_trans(antiprotos[j])

            d = metric(proto.cuda(), antiproto.cuda())
            proto_anti_d.append(d)

    neuron_exclude_list = [neuron]
    all_rand_proto_anti_d = []
    for i in range(3):
        rand_proto_anti_d = []

        rand_neuron1 = np.random.randint(0, 256)
        rand_neuron2 = np.random.randint(0, 256)

        while rand_neuron1 in neuron_exclude_list or rand_neuron2 in neuron_exclude_list or rand_neuron1 == rand_neuron2:
            rand_neuron1 = np.random.randint(0, 256)
            rand_neuron2 = np.random.randint(0, 256)

        neuron_exclude_list.append(rand_neuron1)
        neuron_exclude_list.append(rand_neuron2)

        model_id, neuron_coord, module_name, states, \
        curvedir, ranksdir, actsdir, \
        topdir, botdir, protodir, antiprotodir, poslucentdir, neglucentdir = get_max_stim_dirs(model_name, layer, rand_neuron1, mask="", ablated="_ablated")

        rand1_protos = [Image.open(os.path.join(protodir, file)) for file in os.listdir(protodir) if ".png" in file]

        model_id, neuron_coord, module_name, states, \
        curvedir, ranksdir, actsdir, \
        topdir, botdir, protodir, antiprotodir, poslucentdir, neglucentdir = get_max_stim_dirs(model_name, layer, rand_neuron2, mask="", ablated="_ablated")

        rand2_antiprotos = [Image.open(os.path.join(antiprotodir, file)) for file in os.listdir(antiprotodir) if ".png" in file]

        for j in range(len(rand1_protos)):
            for k in range(j, len(rand2_antiprotos)):
                rand_proto = norm_trans(rand1_protos[j])
                rand_antiproto = norm_trans(rand2_antiprotos[k])

                d = metric(rand_proto.cuda(), rand_antiproto.cuda())
                rand_proto_anti_d.append(d)
                # print(f"\nMin|Max distance:  {d}")

        all_rand_proto_anti_d.append(torch.mean(torch.Tensor(rand_proto_anti_d)))

    return torch.mean(torch.Tensor(proto_anti_d)), torch.mean(torch.Tensor(all_rand_proto_anti_d))


#   TODO:  While I currently taking only top and bot 1 channel, I could take top bot 10 to even out the statistics with 10 random pairs.  In that case, I need to sort.
#
#          If the min/max protos are more similar (lower lpips) than random, then this provides some evidence that a neuron's positive and negative weights
#          have some feature overlap.  Why would a neuron set negative weights on similar features it wants for excitation?  What is the avg weight mag
#          of these filters?
def max_input_proto_compare(extractor, topdir, botdir, protodir, antiprotodir, poslucentdir, neglucentdir,
                            input_layer_name, weight_name, neuron, model_name, metric, use_acts=False, compare_layer="layer4.1.BatchNorm2dbn1"):
    weights = extractor.model.state_dict()[weight_name][neuron]
    idx_offset = weights.shape[-1] // 2

    min_channels = None
    max_channels = None
    if use_acts:
        top_acts, bot_acts, proto_acts, antiproto_acts = \
            get_all_stim_acts(extractor, topdir, botdir, protodir, antiprotodir, poslucentdir, neglucentdir, input_layer_name, neuron_coord=None)

        mid_idx = torch.Tensor(top_acts).shape[-1] // 2

        # top_acts = torch.Tensor(top_acts)[:, :, :, mid_idx-idx_offset:mid_idx+idx_offset+1, mid_idx-idx_offset:mid_idx+idx_offset+1]
        # bot_acts = torch.Tensor(bot_acts)[:, :, :, mid_idx-idx_offset:mid_idx+idx_offset+1, mid_idx-idx_offset:mid_idx+idx_offset+1]
        proto_acts = torch.Tensor(proto_acts)[:, :, :, mid_idx-idx_offset:mid_idx+idx_offset+1, mid_idx-idx_offset:mid_idx+idx_offset+1]
        # antiproto_acts = torch.Tensor(antiproto_acts)[:, :, :, mid_idx-idx_offset:mid_idx+idx_offset+1, mid_idx-idx_offset:mid_idx+idx_offset+1]

        # top_acts = torch.mean(top_acts, 0)[0]
        # bot_acts = torch.mean(bot_acts, 0)[0]
        proto_acts = torch.mean(proto_acts, 0)[0]
        # antiproto_acts = torch.mean(antiproto_acts, 0)[0]

        channel_acts = torch.sum((weights * proto_acts), (1, 2))

        min_channels = torch.argmin(channel_acts)
        max_channels = torch.argmax(channel_acts)
    else:
        channel_weights = torch.sum(weights, (1, 2))

        #   GET TOP AND BOT 10
        all_ord_list = np.arange(channel_weights.shape[0]).tolist()
        _, all_ord_sorted = zip(*sorted(zip(channel_weights.tolist(), all_ord_list), reverse=True))

        min_channels = list(all_ord_sorted[-3:])
        max_channels = list(all_ord_sorted[:3])

        # print(f"Min channel: {min_channels[-1]}, Value: {channel_weights[min_channels[-1]]}")
        # print(f"Max channel: {max_channels[0]}, Value: {channel_weights[max_channels[0]]}")

    #   TODO:  Perform LPIPS on pairs of min and max protos (maybe just pair 0 with 0, 1 with 1, etc).  Then select two random neurons in same layer,
    #          get their protos and LPIPS compare.  Will min/max comparison be higher than random?
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    norm_trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])

    all_min_max_d = []
    #   NOTE:  Comparing top 1 with bot 1, top 2 with bot 2, etc.  No cross-comparisons such as top 1 with bot 8.
    for c in range(len(min_channels)):
        min_c = min_channels[-1*(c+1)]
        max_c = max_channels[c]

        # if min_c >= 398 or max_c >= 398:
        #     return None, None

        model_id, neuron_coord, module_name, states, \
        curvedir, ranksdir, actsdir, \
        topdir, botdir, protodir, antiprotodir, poslucentdir, neglucentdir = get_max_stim_dirs(model_name, compare_layer, min_c, mask="")

        min_protos = [Image.open(os.path.join(protodir, file)) for file in os.listdir(protodir) if ".png" in file]

        model_id, neuron_coord, module_name, states, \
        curvedir, ranksdir, actsdir, \
        topdir, botdir, protodir, antiprotodir, poslucentdir, neglucentdir = get_max_stim_dirs(model_name, compare_layer, max_c, mask="")

        max_protos = [Image.open(os.path.join(protodir, file)) for file in os.listdir(protodir) if ".png" in file]

        min_max_d = []
        for i in range(len(min_protos)):
            #   Use inner for j in range(start=j, stop=len(min_protos)) to perform all comparison permutations.
            for j in range(i, len(min_protos)):
                min_proto = norm_trans(min_protos[i])
                max_proto = norm_trans(max_protos[j])

                d = metric(min_proto.cuda(), max_proto.cuda())
                min_max_d.append(d)
                # print(f"\nMin|Max distance:  {d}")

        all_min_max_d.append(torch.mean(torch.Tensor(min_max_d)))

    neuron_exclude_list = min_channels + max_channels
    all_rand_d = []
    #   10 different random samplings.
    for i in range(3):
        # max_c = max_channels[i]

        rand_d = []

        rand_neuron1 = np.random.randint(0, 128)
        rand_neuron2 = np.random.randint(0, 128)

        while rand_neuron1 in neuron_exclude_list or rand_neuron2 in neuron_exclude_list:
            if rand_neuron1 in neuron_exclude_list:
                rand_neuron1 = np.random.randint(0, 128)

            if rand_neuron2 in neuron_exclude_list:
                rand_neuron2 = np.random.randint(0, 128)

        neuron_exclude_list.append(rand_neuron1)
        neuron_exclude_list.append(rand_neuron2)

        model_id, neuron_coord, module_name, states, \
        curvedir, ranksdir, actsdir, \
        topdir, botdir, protodir, antiprotodir, poslucentdir, neglucentdir = get_max_stim_dirs(model_name, compare_layer, rand_neuron1, mask="")

        rand1_protos = [Image.open(os.path.join(protodir, file)) for file in os.listdir(protodir) if ".png" in file]

        model_id, neuron_coord, module_name, states, \
        curvedir, ranksdir, actsdir, \
        topdir, botdir, protodir, antiprotodir, poslucentdir, neglucentdir = get_max_stim_dirs(model_name, compare_layer, rand_neuron2, mask="")

        rand2_protos = [Image.open(os.path.join(protodir, file)) for file in os.listdir(protodir) if ".png" in file]

        for j in range(len(rand1_protos)):
            for k in range(j, len(rand1_protos)):
                rand1_proto = norm_trans(rand1_protos[j])
                rand2_proto = norm_trans(rand2_protos[k])

                d = metric(rand1_proto.cuda(), rand2_proto.cuda())
                rand_d.append(d)
                # print(f"\nMin|Max distance:  {d}")

        all_rand_d.append(torch.mean(torch.Tensor(rand_d)))

    return torch.mean(torch.Tensor(all_min_max_d)), torch.mean(torch.Tensor(all_rand_d))


#   TODO:  get input acts for neuron's max stims.  Retrieve neuron's weights and extract input acts for pos and neg weights
#          separately.  Then average the inputs for these weights and compare.  Pos avg input mag should be higher
#          for max exc stim.  But is neg avg input higher for max inh stim?
def get_pos_neg_input_mag(extractor, topdir, botdir, protodir, antiprotodir, poslucentdir, neglucentdir, input_layer_name, weight_name, neuron):
    weights = extractor.model.state_dict()[weight_name]
    idx_offset = weights.shape[-1] // 2

    top_acts, bot_acts, proto_acts, antiproto_acts = \
        get_all_stim_acts(extractor, topdir, botdir, protodir, antiprotodir, poslucentdir, neglucentdir, input_layer_name, neuron_coord=None)

    mid_idx = torch.Tensor(top_acts).shape[-1] // 2

    top_acts = torch.Tensor(top_acts)[:, :, :, mid_idx-idx_offset:mid_idx+idx_offset+1, mid_idx-idx_offset:mid_idx+idx_offset+1]
    bot_acts = torch.Tensor(bot_acts)[:, :, :, mid_idx-idx_offset:mid_idx+idx_offset+1, mid_idx-idx_offset:mid_idx+idx_offset+1]
    proto_acts = torch.Tensor(proto_acts)[:, :, :, mid_idx-idx_offset:mid_idx+idx_offset+1, mid_idx-idx_offset:mid_idx+idx_offset+1]
    antiproto_acts = torch.Tensor(antiproto_acts)[:, :, :, mid_idx-idx_offset:mid_idx+idx_offset+1, mid_idx-idx_offset:mid_idx+idx_offset+1]

    top_acts = torch.flatten(torch.mean(top_acts, 0))
    bot_acts = torch.flatten(torch.mean(bot_acts, 0))
    proto_acts = torch.flatten(torch.mean(proto_acts, 0))
    antiproto_acts = torch.flatten(torch.mean(antiproto_acts, 0))

    top_acts = torch.clamp(top_acts, min=0, max=None)
    bot_acts = torch.clamp(bot_acts, min=0, max=None)
    proto_acts = torch.clamp(proto_acts, min=0, max=None)
    antiproto_acts = torch.clamp(antiproto_acts, min=0, max=None)

    top_pos_means = []
    bot_pos_means = []
    proto_pos_means = []
    antiproto_pos_means = []
    top_neg_means = []
    bot_neg_means = []
    proto_neg_means = []
    antiproto_neg_means = []
    for n in range(weights.shape[0]):
        w = torch.flatten(weights[n])

        top_pos_inputs = top_acts[torch.nonzero(w > 0)]
        bot_pos_inputs = bot_acts[torch.nonzero(w > 0)]
        proto_pos_inputs = proto_acts[torch.nonzero(w > 0)]
        antiproto_pos_inputs = antiproto_acts[torch.nonzero(w > 0)]

        top_neg_inputs = top_acts[torch.nonzero(w < 0)]
        bot_neg_inputs = bot_acts[torch.nonzero(w < 0)]
        proto_neg_inputs = proto_acts[torch.nonzero(w < 0)]
        antiproto_neg_inputs = antiproto_acts[torch.nonzero(w < 0)]

        if n == neuron:
            print("OWN NEURON")
            print(f"Neuron {n} avg top pos input mag: {torch.mean(top_pos_inputs)}")
            print(f"Neuron {n} avg top neg input mag: {torch.mean(top_neg_inputs)}\n\n")

            print(f"Neuron {n} avg bot pos input mag: {torch.mean(bot_pos_inputs)}")
            print(f"Neuron {n} avg bot neg input mag: {torch.mean(bot_neg_inputs)}\n\n")

            print(f"Neuron {n} avg proto pos input mag: {torch.mean(proto_pos_inputs)}")
            print(f"Neuron {n} avg proto neg input mag: {torch.mean(proto_neg_inputs)}\n\n")

            print(f"Neuron {n} avg antiproto pos input mag: {torch.mean(antiproto_pos_inputs)}")
            print(f"Neuron {n} avg antiproto neg input mag: {torch.mean(antiproto_neg_inputs)}\n\n")
        else:
            top_pos_means.append(torch.mean(top_pos_inputs))
            bot_pos_means.append(torch.mean(bot_pos_inputs))
            proto_pos_means.append(torch.mean(proto_pos_inputs))
            antiproto_pos_means.append(torch.mean(antiproto_pos_inputs))

            top_neg_means.append(torch.mean(top_neg_inputs))
            bot_neg_means.append(torch.mean(bot_neg_inputs))
            proto_neg_means.append(torch.mean(proto_neg_inputs))
            antiproto_neg_means.append(torch.mean(antiproto_neg_inputs))

    print("AVG OTHER NEURONS")
    print(f"Avg top pos input mag: {torch.mean(torch.Tensor(top_pos_means))}")
    print(f"Avg top neg input mag: {torch.mean(torch.Tensor(top_neg_means))}\n\n")

    print(f"Avg bot pos input mag: {torch.mean(torch.Tensor(bot_pos_means))}")
    print(f"Avg bot neg input mag: {torch.mean(torch.Tensor(bot_neg_means))}\n\n")

    print(f"Avg proto pos input mag: {torch.mean(torch.Tensor(proto_pos_means))}")
    print(f"Avg proto neg input mag: {torch.mean(torch.Tensor(proto_neg_means))}\n\n")

    print(f"Avg antiproto pos input mag: {torch.mean(torch.Tensor(antiproto_pos_means))}")
    print(f"Avg antiproto neg input mag: {torch.mean(torch.Tensor(antiproto_neg_means))}\n\n")


def intrachannel_top_stim(extractor, weight_name, neuron, model_name, target_layer, input_layer):
    weights = extractor.model.state_dict()[weight_name][neuron]
    idx_offset = weights.shape[-1] // 2
    # max_weights = torch.max(torch.flatten(weights, start_dim=1, end_dim=-1), -1)[0]
    summed_weights = torch.sum(weights, (1, 2))

    all_ord_list = np.arange(summed_weights.shape[0]).tolist()
    _, all_ord_sorted = zip(*sorted(zip(summed_weights.tolist(), all_ord_list), reverse=True))

    model_id, neuron_coord, module_name, states, \
    curvedir, ranksdir, actsdir, \
    topdir, botdir, protodir, antiprotodir, poslucentdir, neglucentdir = get_max_stim_dirs(model_name, target_layer, neuron, mask="/no_mask")

    top_acts, bot_acts, proto_acts, antiproto_acts, poslucent_acts, neglucent_acts = \
        get_all_stim_acts(extractor, None, None, protodir, None, None, None, input_layer, neuron_coord=None)

    mid_idx = torch.Tensor(proto_acts).shape[-1] // 2

    acts = torch.Tensor(proto_acts)[:, :, :, mid_idx-idx_offset:mid_idx+idx_offset+1, mid_idx-idx_offset:mid_idx+idx_offset+1]
    acts = torch.mean(acts, 0)[0]
    acts = torch.clamp(acts, min=0, max=None)

    mixed_neg_total = 0
    all_neg_total = 0
    for i in range(weights.shape[0]):
        w = weights[i]

        if torch.all(w > 0):
            continue

        if torch.all(w < 0):
            all_neg_total += (-1 * torch.sum(w * acts[i].cuda())).cpu().numpy()
        else:
            w = torch.clamp(w, min=None, max=0)
            mixed_neg_total += (-1 * torch.sum(w * acts[i].cuda())).cpu().numpy()

    return all_neg_total / (all_neg_total + mixed_neg_total)


def max_weight_stim_sim(extractor, weight_name, neuron, model_name, metric, mask=None, compare_layer="Conv2d8"):
    
    # intact_weights = extractor.model.state_dict()[weight_name][neuron]
    # channel_weights = torch.sum(weights, (1, 2))

    # weights = clamp_ablate_unit(extractor.model.state_dict(), weight_name, neuron, min=0)
    # weights = weights[weight_name][neuron]

    weights = extractor.model.state_dict()[weight_name][neuron]
    channel_weights = torch.sum(weights, (1, 2))

    all_ord_list = np.arange(channel_weights.shape[0]).tolist()
    _, all_ord_sorted = zip(*sorted(zip(channel_weights.tolist(), all_ord_list), reverse=True))

    max_channels = list(all_ord_sorted[:5])
    # max_channels = list(all_ord_sorted[-5:])

    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    norm_trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])

    all_most_compares = []
    all_least_compares = []
    for c in range(len(max_channels)):
        max_c = max_channels[c]

        model_id, neuron_coord, module_name, states, \
        curvedir, ranksdir, actsdir, \
        topdir, botdir, protodir, antiprotodir, poslucentdir, neglucentdir = get_max_stim_dirs(model_name, compare_layer, max_c, mask="")

        max_stims = [Image.open(os.path.join(topdir, file)) for file in os.listdir(topdir) if ".png" in file]

        max_compares = []
        for i in range(channel_weights.shape[0]):
            model_id, neuron_coord, module_name, states, \
            curvedir, ranksdir, actsdir, \
            topdir, botdir, protodir, antiprotodir, poslucentdir, neglucentdir = get_max_stim_dirs(model_name, compare_layer, i, mask="")

            compare_stims = [Image.open(os.path.join(topdir, file)) for file in os.listdir(topdir) if ".png" in file]
            
            all_distances = []
            for j in range(len(max_stims)):
                #   Use inner for j in range(start=j, stop=len(min_protos)) to perform all comparison permutations.
                max_stim = None
                if mask is not None:
                    max_stim = norm_trans(max_stims[j] * mask).cuda()
                else:
                    max_stim = norm_trans(max_stims[j]).cuda()

                for k in range(j, len(max_stims)):
                    compare_stim = None
                    if mask is not None:
                        compare_stim = norm_trans(compare_stims[k] * mask).cuda()
                    else:
                        compare_stim = norm_trans(compare_stims[k]).cuda()

                    #     d = metric((max_stim * mask).cuda(), (compare_stim * mask).cuda())
                    # else:
                    d = metric(max_stim, compare_stim)

                    all_distances.append(d)

            max_compares.append(torch.mean(torch.tensor(all_distances)))

        max_compares = torch.tensor(max_compares)
        max_compares[max_c] = 1000
        most_sim_channel = torch.argsort(max_compares)[:5]

        max_compares[max_c] = -1000
        least_sim_channel = torch.argsort(max_compares)[-5:]

        # print(f"Compare for max pos-weighted channel {max_c}:  most similar channels: {min_dist_c}  mean sim value:  {torch.mean(max_compares[min_dist_c])}  mean channel weight:  {torch.mean(channel_weights[min_dist_c])}")
        
        all_most_compares.append(torch.mean(channel_weights[most_sim_channel]))
        all_least_compares.append(torch.mean(channel_weights[least_sim_channel]))

    return torch.mean(torch.tensor(all_most_compares)), torch.mean(torch.tensor(all_least_compares))



def max_weight_curve_sim(extractor, weight_name, neuron, model_name, compare_layer="Conv2d8"):
    
    # intact_weights = extractor.model.state_dict()[weight_name][neuron]
    # channel_weights = torch.sum(weights, (1, 2))

    # weights = clamp_ablate_unit(extractor.model.state_dict(), weight_name, neuron, min=0)
    # weights = weights[weight_name][neuron]

    weights = extractor.model.state_dict()[weight_name][neuron]
    channel_weights = torch.sum(weights, (1, 2))

    # print(f"Percent positive channels in {weight_name}, neuron {neuron}: {torch.nonzero(channel_weights > 0).shape[0] / channel_weights.shape[0]}")

    all_ord_list = np.arange(channel_weights.shape[0]).tolist()
    _, all_ord_sorted = zip(*sorted(zip(channel_weights.tolist(), all_ord_list), reverse=True))

    max_channels = list(all_ord_sorted[:1])
    # max_channels = list(all_ord_sorted[-5:])

    all_most_compares = []
    all_least_compares = []

    all_corrs = []
    for c in range(len(max_channels)):
        max_c = max_channels[c]

        # model_id, neuron_coord, module_name, states, \
        # curvedir, ranksdir, actsdir, \
        # topdir, botdir, protodir, antiprotodir, poslucentdir, neglucentdir = get_max_stim_dirs(model_name, "layer3.0.downsample.1", max_c, mask="")

        # id_curve = np.load(os.path.join(curvedir, f"layer3.0.downsample.1_unit{max_c}_intact_unrolled_act.npy"))
        # id_curve = np.clip(id_curve, 0, None)

        model_id, neuron_coord, module_name, states, \
        curvedir, ranksdir, actsdir, \
        topdir, botdir, protodir, antiprotodir, poslucentdir, neglucentdir = get_max_stim_dirs(model_name, compare_layer, max_c, mask="")

        max_curve = np.load(os.path.join(curvedir, f"{compare_layer}_unit{max_c}_intact_unrolled_act.npy"))
        # max_curve = np.clip(max_curve + id_curve, 0, None)

        max_curve = np.clip(max_curve, 0, None)
        max_curve = normalize(max_curve)

        max_compares = []
        for i in range(channel_weights.shape[0]):
            # model_id, neuron_coord, module_name, states, \
            # curvedir, ranksdir, actsdir, \
            # topdir, botdir, protodir, antiprotodir, poslucentdir, neglucentdir = get_max_stim_dirs(model_name, "layer3.0.downsample.1", i, mask="")

            # id_curve = np.load(os.path.join(curvedir, f"layer3.0.downsample.1_unit{i}_intact_unrolled_act.npy"))
            # id_curve = np.clip(id_curve, 0, None)

            model_id, neuron_coord, module_name, states, \
            curvedir, ranksdir, actsdir, \
            topdir, botdir, protodir, antiprotodir, poslucentdir, neglucentdir = get_max_stim_dirs(model_name, compare_layer, i, mask="")

            compare_curve = np.load(os.path.join(curvedir, f"{compare_layer}_unit{i}_intact_unrolled_act.npy"))
            # compare_curve = np.clip(compare_curve + id_curve, 0, None)

            compare_curve = np.clip(compare_curve, 0, None)
            compare_curve = normalize(compare_curve)

            max_compares.append(pearsonr(max_curve, compare_curve)[0])

        max_compares = torch.tensor(max_compares)
        max_compares[max_c] = -1000
        most_sim_channel = torch.argsort(max_compares)[-1:]

        max_compares[max_c] = 1000
        least_sim_channel = torch.argsort(max_compares)[:1]

        if neuron == 0:
            print(f"Compare for max pos-weighted channel {max_c}:  most similar channels: {most_sim_channel}  mean corr value:  {torch.mean(max_compares[most_sim_channel])}  mean channel weight:  {torch.mean(channel_weights[most_sim_channel])}")
    
        all_most_compares.append(torch.mean(channel_weights[most_sim_channel]))
        all_least_compares.append(torch.mean(channel_weights[least_sim_channel]))

    return torch.mean(torch.tensor(all_most_compares)), torch.mean(torch.tensor(all_least_compares))

        #   Do not include last as it is the corr with itself.
    #     order = torch.argsort(max_compares)[:-1]
    #     all_corrs.append(max_compares[order].cpu().numpy())
    #     all_most_compares.append(channel_weights[order].cpu().numpy())

    # all_corrs = np.mean(np.array(all_corrs), 0)
    # all_most_compares = np.mean(np.array(all_most_compares), 0)

    # fig = plt.figure(figsize=(8,5))
    # ax = fig.subplots()

    # ax.scatter(all_corrs, all_most_compares)
    # ax.set_xlabel("Correlation")
    # ax.set_ylabel("Mean Weight")
    # ax.set_title(f"Weights of Ordered Corrs of Top 5 Positive Weight\n{args.network} Layer {args.layer} Neuron {neuron}")

    # plt.savefig(f'/home/andrelongon/Documents/inhibition_code/plots/corr_weights/all_rank_corrs/{args.network}_{args.layer}_5sim_neuron{neuron}.png', dpi=128)
    # plt.close()


    # return None, None


def top_bot_weight_corr(extractor, weight_name, neuron, model_name, compare_layer):
    weights = extractor.model.state_dict()[weight_name][neuron]
    channel_weights = torch.sum(weights, (1, 2))

    all_ord_list = np.arange(channel_weights.shape[0]).tolist()
    _, all_ord_sorted = zip(*sorted(zip(channel_weights.tolist(), all_ord_list), reverse=True))

    pos_max_c = all_ord_sorted[0]
    neg_max_c = all_ord_sorted[-1]

    all_corrs = []
    neg_corr = None

    model_id, neuron_coord, module_name, states, \
    curvedir, ranksdir, actsdir, \
    topdir, botdir, protodir, antiprotodir, poslucentdir, neglucentdir = get_max_stim_dirs(model_name, compare_layer, pos_max_c, mask="")

    max_curve = np.load(os.path.join(curvedir, f"{compare_layer}_unit{pos_max_c}_intact_unrolled_act.npy"))
    # max_curve = np.clip(max_curve, 0, None)
    max_curve = normalize(max_curve)

    for i in range(channel_weights.shape[0]):
        if i == pos_max_c:
            continue

        model_id, neuron_coord, module_name, states, \
        curvedir, ranksdir, actsdir, \
        topdir, botdir, protodir, antiprotodir, poslucentdir, neglucentdir = get_max_stim_dirs(model_name, compare_layer, i, mask="")

        compare_curve = np.load(os.path.join(curvedir, f"{compare_layer}_unit{i}_intact_unrolled_act.npy"))
        # compare_curve = np.clip(compare_curve, 0, None)
        compare_curve = normalize(compare_curve)

        if i == neg_max_c:
            neg_corr = pearsonr(max_curve, compare_curve)[0]
        else:
            all_corrs.append(pearsonr(max_curve, compare_curve)[0])

    return neg_corr, np.mean(np.array(all_corrs))


def top_pos_neg_input_corr(extractor, weight_name, neuron, model_name, target_layer, input_layer):
    weights = extractor.model.state_dict()[weight_name][neuron]
    channel_weights = torch.sum(weights, (1, 2))

    all_ord_list = np.arange(channel_weights.shape[0]).tolist()
    _, all_ord_sorted = zip(*sorted(zip(channel_weights.tolist(), all_ord_list), reverse=True))

    pos_max_c = all_ord_sorted[0]
    neg_max_c = all_ord_sorted[-1]

    model_id, neuron_coord, module_name, states, \
    curvedir, ranksdir, actsdir, \
    topdir, botdir, protodir, antiprotodir, poslucentdir, neglucentdir = get_max_stim_dirs(model_name, target_layer, neuron, mask="")

    target_curve = np.load(os.path.join(curvedir, f"{target_layer}_unit{neuron}_intact_unrolled_act.npy"))
    # target_curve = np.clip(target_curve, 0, None)
    # target_curve = normalize(target_curve)

    model_id, neuron_coord, module_name, states, \
    curvedir, ranksdir, actsdir, \
    topdir, botdir, protodir, antiprotodir, poslucentdir, neglucentdir = get_max_stim_dirs(model_name, input_layer, pos_max_c, mask="")

    pos_curve = np.load(os.path.join(curvedir, f"{input_layer}_unit{pos_max_c}_intact_unrolled_act.npy"))
    # pos_curve = np.clip(pos_curve, 0, None)
    # pos_curve = normalize(pos_curve)

    model_id, neuron_coord, module_name, states, \
    curvedir, ranksdir, actsdir, \
    topdir, botdir, protodir, antiprotodir, poslucentdir, neglucentdir = get_max_stim_dirs(model_name, input_layer, neg_max_c, mask="")

    neg_curve = np.load(os.path.join(curvedir, f"{input_layer}_unit{neg_max_c}_intact_unrolled_act.npy"))
    # neg_curve = np.clip(neg_curve, 0, None)
    # neg_curve = normalize(neg_curve)

    return pearsonr(target_curve, pos_curve)[0], pearsonr(target_curve, neg_curve)[0]


def top_pos_neg_rand_corr(extractor, weight_name, neuron, model_name, target_layer, input_layer):
    weights = extractor.model.state_dict()[weight_name][neuron]
    channel_weights = torch.sum(weights, (1, 2))

    all_ord_list = np.arange(channel_weights.shape[0]).tolist()
    _, all_ord_sorted = zip(*sorted(zip(channel_weights.tolist(), all_ord_list), reverse=True))

    pos_max_c = all_ord_sorted[0]
    neg_max_c = all_ord_sorted[-1]

    # neuron_exclude_list = [pos_max_c, neg_max_c]
    # rand_c = np.random.randint(0, channel_weights.shape[0])

    # while rand_c in neuron_exclude_list:
    #     rand_c = np.random.randint(0, channel_weights.shape[0])

    # model_id, neuron_coord, module_name, states, \
    # curvedir, ranksdir, actsdir, \
    # topdir, botdir, protodir, antiprotodir, poslucentdir, neglucentdir = get_max_stim_dirs(model_name, input_layer, rand_c, mask="")

    # rand_curve = np.load(os.path.join(curvedir, f"{input_layer}_unit{rand_c}_intact_unrolled_act.npy"))
    # rand_curve = np.clip(rand_curve, 0, None)
    # rand_curve = normalize(rand_curve)

    model_id, neuron_coord, module_name, states, \
    curvedir, ranksdir, actsdir, \
    topdir, botdir, protodir, antiprotodir, poslucentdir, neglucentdir = get_max_stim_dirs(model_name, input_layer, pos_max_c, mask="")

    pos_curve = np.load(os.path.join(curvedir, f"{input_layer}_unit{pos_max_c}_intact_unrolled_act.npy"))
    pos_curve = np.clip(pos_curve, 0, None)

    if np.nonzero(pos_curve > 0)[0].shape[0] == 0:
        return None, None

    pos_curve = normalize(pos_curve)

    # model_id, neuron_coord, module_name, states, \
    # curvedir, ranksdir, actsdir, \
    # topdir, botdir, protodir, antiprotodir, poslucentdir, neglucentdir = get_max_stim_dirs(model_name, input_layer, neg_max_c, mask="")

    # neg_curve = np.load(os.path.join(curvedir, f"{input_layer}_unit{neg_max_c}_intact_unrolled_act.npy"))
    # neg_curve = np.clip(neg_curve, 0, None)
    # neg_curve = normalize(neg_curve)

    all_corrs = []
    neg_corr = None
    for i in range(channel_weights.shape[0]):
        if i == pos_max_c:
            continue

        model_id, neuron_coord, module_name, states, \
        curvedir, ranksdir, actsdir, \
        topdir, botdir, protodir, antiprotodir, poslucentdir, neglucentdir = get_max_stim_dirs(model_name, input_layer, i, mask="")

        compare_curve = np.load(os.path.join(curvedir, f"{input_layer}_unit{i}_intact_unrolled_act.npy"))
        compare_curve = np.clip(compare_curve, 0, None)

        if np.nonzero(compare_curve > 0)[0].shape[0] == 0:
            if i == neg_max_c:
                return None, None
                # print(f"WTF {i} -> {neuron}")
            continue

        compare_curve = normalize(compare_curve)

        if i == neg_max_c:
            neg_corr = cosine(pos_curve, compare_curve)
            # neg_corr = pearsonr(pos_curve, compare_curve)[0]

        all_corrs.append(cosine(pos_curve, compare_curve))
        # all_corrs.append(pearsonr(pos_curve, compare_curve)[0])

    return neg_corr, np.mean(np.array(all_corrs))

    # return cosine(pos_curve, neg_curve), cosine(pos_curve, rand_curve), cosine(neg_curve, rand_curve)
    # return pearsonr(pos_curve, neg_curve)[0], pearsonr(pos_curve, rand_curve)[0], pearsonr(neg_curve, rand_curve)[0]


#   TODO:  try random batch of imnet val rather than other neuron's top stims.
def top_bot_stim_sim(extractor, weight_name, model_name, layer, neuron, metric, mask):
    model_id, neuron_coord, module_name, states, \
    curvedir, ranksdir, actsdir, \
    topdir, botdir, protodir, antiprotodir, poslucentdir, neglucentdir = get_max_stim_dirs(model_name, layer, neuron, mask="")

    top_stims = [Image.open(os.path.join(topdir, file)) for file in os.listdir(topdir) if ".png" in file]
    bot_stims = [Image.open(os.path.join(botdir, file)) for file in os.listdir(botdir) if ".png" in file]

    top_bot_dists = stim_compare((top_stims, bot_stims), metric, mask)

    # num_channels = extractor.model.state_dict()[weight_name].shape[0]
    # neuron_exclude_list = [neuron]
    # rand_c = np.random.randint(0, 384)

    # while rand_c in neuron_exclude_list:
    #     rand_c = np.random.randint(0, num_channels)
    
    # model_id, neuron_coord, module_name, states, \
    # curvedir, ranksdir, actsdir, \
    # topdir, botdir, protodir, antiprotodir, poslucentdir, neglucentdir = get_max_stim_dirs('alexnet', 'features.Conv2d6', rand_c, mask="")

    # rand_stims = [Image.open(os.path.join(topdir, file)) for file in os.listdir(topdir) if ".png" in file]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ])

    imnet_folder = r"/home/andrelongon/Documents/data/imagenet/val"
    imagenet_data = torchvision.datasets.ImageFolder(imnet_folder, transform=transform)
    dataloader = torch.utils.data.DataLoader(imagenet_data, batch_size=9, shuffle=True, drop_last=False)

    rand_stims, _ = next(iter(dataloader))
    rand_stims = [transforms.ToPILImage()(rand_stim) for rand_stim in rand_stims]

    # mask = np.transpose(mask, (2, 0, 1))
    top_rand_dists = stim_compare((top_stims, rand_stims), metric, mask)

    return np.mean(top_bot_dists), np.mean(top_rand_dists)


#   TODO:  Extract entire post-relu input tensor for a given layer.
#          For instance, features.2 for second conv layer of AlexNet.
#          Compute histogram of this tensor with zero as a bucket (b/t -1e-16 to 1e-16 or something).
#          Other buckets are all positive:  1e-16 to 0.01, 0.01 to 0.1, 0.1 to 1, 1 to 10, 10 to 100?
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str)
    parser.add_argument('--layer', type=str)
    parser.add_argument('--neuron', type=int, default=0, required=False)
    parser.add_argument('--ablate_type', type=str, default="")
    parser.add_argument('--input_layer_name', type=str, required=False)
    parser.add_argument('--mask_layer_name', type=str, default=None, required=False)
    parser.add_argument('--weight_name', type=str, required=False)
    # parser.add_argument('--layer_number', type=int, required=False)
    # parser.add_argument('--neuron_coord', type=int)
    # parser.add_argument('--type', type=str, required=False)
    args = parser.parse_args()

    torch.manual_seed(0)

    #   Work with intact stimuli for now.  Perhaps refactor to consolidate max directories into single dictionary.  Then can loop through
    #   max_dir_dict in get_all_stim_acts and create new dict with key + "_acts" to handle being different dirs.
    model_name, neuron_coord, module_name, states, \
    curvedir, ranksdir, actsdir, \
    topdir, botdir, protodir, antiprotodir, poslucentdir, neglucentdir = get_max_stim_dirs(args.network, args.layer, args.neuron, ablate_type=args.ablate_type, load_states=True)

    source_name = 'custom' if model_name == 'cornet-s' else 'torchvision'
    # extractor = get_extractor(
    #     model_name=model_name,
    #     source=source_name,
    #     device='cuda',
    #     pretrained=True
    # )

    extractor = get_extractor_from_model(model=AlexNet().cuda(), device='cuda', backend='pt')

    states = torch.load(f"/media/andrelongon/DATA/imnet_weights/overlap_finetune/gap_alexnet_features.16_baseline_try2_3ep.pth")
    if states is not None:
        extractor.model.load_state_dict(states)

    mask = None
    if args.mask_layer_name is not None:
        cent_pos, corner, imgsize, Xlim, Ylim, mask = get_center_pos_and_rf(
            extractor.model, args.mask_layer_name, input_size=(3, 224, 224), device="cuda"
        )
        mask = np.expand_dims(normalize(mask), -1)
        # mask = np.transpose(mask, (2, 0, 1))

    # get_pos_neg_input_mag(extractor, topdir, botdir, protodir, antiprotodir, poslucentdir, neglucentdir, args.input_layer_name, args.weight_name, args.neuron)

    #   3 LPIPS architectures:  alex, vgg, squeeze
    metric = lpips.LPIPS(net='alex').cuda()

    all_mean_most_weights = []
    all_mean_least_weights = []

    all_pos_corrs = []
    all_neg_corrs = []

    all_neg_percs = []
    
    all_pos_neg_corr = []
    all_pos_rand_corr = []
    all_pos_avg_corr = []

    all_top_bot_avgs = []
    all_top_rand_avgs = []
    for i in range(args.neuron):
    # for i in range(10):
        # model_name, neuron_coord, module_name, states, \
        # curvedir, ranksdir, actsdir, \
        # topdir, botdir, protodir, antiprotodir, poslucentdir, neglucentdir = get_max_stim_dirs(args.network, args.layer, i, ablate_type=args.ablate_type)

        # print(f"Neuron {i}")
        # mean_most_weight, mean_least_weight = max_weight_stim_sim(extractor, args.weight_name, i, args.network, metric, mask=mask, compare_layer=args.input_layer_name)
        # mean_most_weight, mean_least_weight = max_weight_curve_sim(extractor, args.weight_name, i, args.network, compare_layer=args.input_layer_name)
        # all_mean_most_weights.append(mean_most_weight)
        # all_mean_least_weights.append(mean_least_weight)

        # neg_corr, avg_corr = top_bot_weight_corr(extractor, args.weight_name, i, args.network, compare_layer=args.input_layer_name)
        # print(f"neg corr: {neg_corr}, avg_corr: {avg_corr}")

        ##   EXPERIMENT 1
        # pos_neg_corr, pos_avg_corr = top_pos_neg_rand_corr(extractor, args.weight_name, i, args.network, args.layer, args.input_layer_name)
        # if pos_neg_corr is not None and pos_avg_corr is not None:
        #     all_pos_neg_corr.append(pos_neg_corr)
        #     all_pos_avg_corr.append(pos_avg_corr)
        ##

        # all_neg_percs.append(intrachannel_top_stim(extractor, args.weight_name, i, args.network, args.layer, args.input_layer_name))

        ##   EXPERIMENT 2
        top_bot_dists, top_rand_dists = top_bot_stim_sim(extractor, args.weight_name, args.network, args.layer, i, metric, mask)
        all_top_bot_avgs.append(top_bot_dists)
        all_top_rand_avgs.append(top_rand_dists)
        ##


    # print(f"{args.network}_{args.layer}:  {torch.argsort(torch.tensor(all_mean_most_weights))[:5]}")
    # exit()

    fig = plt.figure(figsize=(8,5))
    ax = fig.subplots()

##   EXPERIMENT 2
    f_stat, p_val = f_oneway(np.array(all_top_bot_avgs), np.array(all_top_rand_avgs))
    print(f"{args.network} {args.layer} ANOVA results: F Score={f_stat}, p={p_val}")
    print(f"Top Bot Mean={np.mean(np.array(all_top_bot_avgs))}, Top Random Mean={np.mean(np.array(all_top_rand_avgs))}, \
Delta={np.mean(np.array(all_top_bot_avgs)) - np.mean(np.array(all_top_rand_avgs))}")

    ax.scatter(np.arange(len(all_top_bot_avgs)), np.array(all_top_bot_avgs), color='b', label="Top Bot")
    ax.scatter(np.arange(len(all_top_rand_avgs)), np.array(all_top_rand_avgs), color='y', label="Top Rand")
    ax.legend(loc="upper right")
    ax.set_xlabel("Neuron")
    ax.set_ylabel("LPIPS")
    ax.set_title(f"LPIPS of Neuron's Top Bot and Random Imgs\n{args.network} Layer {args.layer}")

    plt.savefig(f'/home/andrelongon/Documents/inhibition_code/plots/top_bot_sim/{args.network}_{args.layer}.png', dpi=128)
    plt.close()
##

##  EXPERIMENT 1
#     f_stat, p_val = f_oneway(np.array(all_pos_neg_corr), np.array(all_pos_avg_corr))
#     print(f"{args.network} {args.layer} ANOVA results: F Score={f_stat}, p={p_val}")
#     print(f"Pos Avg Mean={np.mean(np.array(all_pos_avg_corr))}, Pos Neg Mean={np.mean(np.array(all_pos_neg_corr))}, \
# Delta={np.mean(np.array(all_pos_avg_corr)) - np.mean(np.array(all_pos_neg_corr))}")

#     ax.scatter(np.arange(len(all_pos_neg_corr)), np.array(all_pos_neg_corr), color='y', label='Pos Neg')
#     ax.scatter(np.arange(len(all_pos_avg_corr)), np.array(all_pos_avg_corr), color='b', label='Pos Avg')
#     # ax.scatter(np.arange(len(all_neg_rand_corr)), np.array(all_neg_rand_corr), color='r', label='Neg Rand')
#     ax.set_xlabel("Neuron")
#     ax.set_ylabel("Cosine")
#     ax.legend(loc="lower right")
#     ax.set_title(f"Top Pos Neg Input Neuron Curve Cosine\n{args.network} Layer {args.layer}")

#     # f_stat, p_val = f_oneway(np.array(all_pos_neg_corr), np.array(all_neg_rand_corr))
#     # print(f"{args.network} {args.layer} ANOVA results: F Score={f_stat}, p={p_val}")

#     plt.savefig(f'/home/andrelongon/Documents/inhibition_code/plots/max_channel_corr/cosine/avg/{args.network}_{args.layer}.png', dpi=128)
#     plt.close()
##