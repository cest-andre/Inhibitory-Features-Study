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
from scipy.stats import spearmanr, f_oneway
from thingsvision import get_extractor
from PIL import Image

from insilico_RF_save import get_center_pos_and_rf
from utils import normalize, saveTopN, Sobel
from selectivity import batch_selectivity
from scatterplot import simple_scattplot
from imnet_val import validate_tuning_curve, validate_tuning_curve_thingsvision

from modify_weights import clamp_ablate_unit, random_ablate_unit, channel_random_ablate_unit, binarize_unit
from transplant import get_activations


def get_ranks(extractor, imgs, val_acts):
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

    # all_ords = np.arange(val_acts.shape[0])
    # val_acts_sorted, ords_sorted = zip(*sorted(zip(val_acts, all_ords), reverse=True))

    #   Obtain ranks for all img activation.
    ranks = [
        torch.nonzero(torch.tensor(val_acts) > torch.tensor(act)).shape[0]
        for act in img_acts
    ]

    return ranks

def extract_ranks(extractor, topdir, botdir, protodir, antiprotodir,
                  layer, selected_layer, selected_neuron, module_name,
                  inh_abl=False, actsdir=None):

    # original_states = copy.deepcopy(extractor.model.state_dict())

    top_imgs = [Image.open(os.path.join(topdir, file)) for file in os.listdir(topdir)]
    bot_imgs = [Image.open(os.path.join(botdir, file)) for file in os.listdir(botdir)]
    proto_imgs = [Image.open(os.path.join(protodir, file)) for file in os.listdir(protodir)]
    antiproto_imgs = [Image.open(os.path.join(antiprotodir, file)) for file in os.listdir(antiprotodir)]
    top_ranks = []
    bot_ranks = []
    proto_ranks = []
    antiproto_ranks = []

    percentages = np.arange(10, 100, 10)
    percentages = np.append(percentages, 99)
    percentages = np.append(percentages, 100)

    iters = 1
    # if inh_abl:
    #     iters = 1
    # else:
    #     iters = percentages.shape[0]

    for i in range(iters):
        if inh_abl:
            # extractor.model.load_state_dict(
            #     clamp_ablate_unit(extractor.model.state_dict(), "layer4.1.conv2.weight", selected_neuron, min=0, max=None)
            # )
            extractor.model.load_state_dict(
                clamp_ablate_unit(extractor.model.state_dict(), module_name + '.weight', selected_neuron, min=0, max=None)
            )

            # extractor.model.load_state_dict(
            #     torch.load(f'/home/andre/tuning_curves/untrained_alexnet/rand_abl/unit{selected_neuron}_rand_abl_weights.pth')
            # )

            # extractor.model.load_state_dict(
            #     binarize_unit(extractor.model.state_dict(), "features.10.weight", selected_neuron)
            # )
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

        top_ranks.append(get_ranks(extractor, top_imgs, val_acts))
        bot_ranks.append(get_ranks(extractor, bot_imgs, val_acts))
        proto_ranks.append(get_ranks(extractor, proto_imgs, val_acts))
        antiproto_ranks.append(get_ranks(extractor, antiproto_imgs, val_acts))

        # ranks.append(torch.nonzero(torch.tensor(abl_acts_sorted) > torch.mean(torch.tensor(proto_acts))).shape[0])

        #   Calculate width.

        #   Integrate antiprotos into tuning curve for plot.
        # all_ords = np.arange(len(rand_abl_acts) + len(imgs))
        # rand_abl_acts = np.append(rand_abl_acts, np.array(proto_acts))
        # rand_abl_acts_sorted, ords_sorted = zip(*sorted(zip(rand_abl_acts, all_ords), reverse=True))
        # proto_indices = [ords_sorted.index(i) for i in range(50000, 50000+9)]

        # abl_all_act_list = np.delete(np.array(rand_abl_acts_sorted), proto_indices).tolist()
        # all_ord_list = np.delete(np.array(all_ords), proto_indices).tolist()

        # plt.scatter(all_ord_list, abl_all_act_list, color='b', label="Imnet Val")
        # plt.scatter(proto_indices, proto_acts, color='r', label="Antiprotos")
        # plt.legend(loc="upper right")
        # plt.title(f"Inh Ablated Tuning Curve")
        # plt.savefig(os.path.join(savedir, f"unit{selected_neuron}_inh_abl_tuning_curve_antiprotos.png"))
        # plt.close()

        # extractor.model.load_state_dict(original_states)

    return top_ranks, bot_ranks, proto_ranks, antiproto_ranks


def compare_tuning_curves(model, savedir, layer, selected_layer, selected_neuron, module_name,
                          inh_abl=True, extractor=None, neuron_coord=None):
    #   Intact tuning curve.
    if inh_abl:
        model.load_state_dict(clamp_ablate_unit(model.state_dict(), module_name + '.weight', selected_neuron, min=0, max=None))

    all_images = act_list = unrolled_act = all_act_list = all_ord_sorted = None
    if extractor is None:
        all_images, act_list, unrolled_act, all_act_list, all_ord_sorted, _ = \
            validate_tuning_curve(model, layer, selected_layer, selected_neuron)
    else:
        all_images, act_list, unrolled_act, all_act_list, all_ord_sorted, _ = \
            validate_tuning_curve_thingsvision(extractor, module_name, selected_neuron, neuron_coord)

    # ord_list = np.arange(len(act_list)).tolist()
    # _, ord_list = zip(*sorted(zip(act_list, ord_list), reverse=True))
    # all_ord_list = np.arange(len(all_ord_sorted)).tolist()
    #
    # plt.scatter(all_ord_list, all_act_list, color='b')
    # plt.savefig(os.path.join(savedir, 'tuning_curve_l%d_n%d.png'
    #              %(selected_layer,selected_neuron)))
    # plt.close()
    #
    if inh_abl:
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

    # exit()

    # #   Inh ablated tuning curve.
    # # original_states = copy.deepcopy(model.state_dict())
    # if inh_abl:
    #     model.load_state_dict(clamp_ablate_unit(model.state_dict(), "features.10.weight", selected_neuron, min=0, max=None))
    #
    #     _, abl_act_list, abl_unrolled_act, abl_all_act_list, abl_all_ord_sorted, abl_gradAmpmap = \
    #         validate_tuning_curve(model, layer, selected_layer, selected_neuron)
    #
    #     np.save(os.path.join(savedir, f"unit{selected_neuron}_inh_abl_unrolled_act.npy"), np.array(abl_unrolled_act))
    #     np.save(os.path.join(savedir, f"unit{selected_neuron}_inh_abl_all_act_list.npy"), np.array(list(abl_all_act_list)))
    #     np.save(os.path.join(savedir, f"unit{selected_neuron}_inh_abl_all_ord_sorted.npy"), np.array(list(abl_all_ord_sorted)))
    # else:
    #     model.load_state_dict(random_ablate_unit(model.state_dict(), "features.10.weight", selected_neuron))
    #     torch.save(model.state_dict(), os.path.join(savedir, f"unit{selected_neuron}_rand_abl_weights.pth"))
    #
    #     _, abl_act_list, abl_unrolled_act, abl_all_act_list, abl_all_ord_sorted, abl_gradAmpmap = \
    #         validate_tuning_curve(model, layer, selected_layer, selected_neuron)
    #
    #     np.save(os.path.join(savedir, f"unit{selected_neuron}_rand_abl_unrolled_act.npy"), np.array(abl_unrolled_act))
    #     np.save(os.path.join(savedir, f"unit{selected_neuron}_rand_abl_all_act_list.npy"), np.array(list(abl_all_act_list)))
    #     np.save(os.path.join(savedir, f"unit{selected_neuron}_rand_abl_all_ord_sorted.npy"), np.array(list(abl_all_ord_sorted)))
    #
    # model.load_state_dict(original_states)

      # Random ablated tuning curve.
    # for p in range(10, 110, 10):
    # p = 99
    # model.load_state_dict(random_ablate_unit(model.state_dict(), "features.10.weight", selected_neuron, perc=p / 100))
    #
    # _, abl_act_list, abl_unrolled_act, abl_all_act_list, abl_all_ord_sorted, abl_gradAmpmap = \
    #     validate_tuning_curve(model, layer, selected_layer, selected_neuron)
    #
    # np.save(os.path.join(savedir, f"unit{selected_neuron}_{p}_abl_unrolled_act.npy"), np.array(abl_unrolled_act))
    # np.save(os.path.join(savedir, f"unit{selected_neuron}_{p}_abl_all_act_list.npy"), np.array(list(abl_all_act_list)))
    # np.save(os.path.join(savedir, f"unit{selected_neuron}_{p}_abl_all_ord_sorted.npy"), np.array(list(abl_all_ord_sorted)))
    #
    # model.load_state_dict(original_states)
    #
    # exit()

    # model.load_state_dict(binarize_unit(model.state_dict(), "features.10.weight", selected_neuron))
    #
    # _, abl_act_list, abl_unrolled_act, abl_all_act_list, abl_all_ord_sorted, abl_gradAmpmap = \
    #     validate_tuning_curve(model, layer, selected_layer, selected_neuron)
    #
    # np.save(os.path.join(savedir, f"unit{selected_neuron}_binarized_unrolled_act.npy"), np.array(abl_unrolled_act))
    # np.save(os.path.join(savedir, f"unit{selected_neuron}_binarized_all_act_list.npy"), np.array(list(abl_all_act_list)))
    # np.save(os.path.join(savedir, f"unit{selected_neuron}_binarized_all_ord_sorted.npy"),
    #         np.array(list(abl_all_ord_sorted)))

    #   Find indices in all_ord_list where all_ord_sorted[0:9] occurs in abl_all_ord_sorted.
    #   Index of abl_all_ord_sorted where it equals all_ord_sorted[0:9].  These indices
    #   can then be deleted from abl_all_act_list.
    # all_ord_sorted = np.load(
    #     f"/home/andre/tuning_curves/intact/layer{selected_layer}_unit{selected_neuron}_intact_all_ord_sorted.npy"
    # )
    # all_ord_list = np.arange(len(all_ord_sorted)).tolist()
    # abl_unrolled_act = np.load(
    #     f"/home/andre/tuning_curves/inh_ablate/layer{selected_layer}_unit{selected_neuron}_inh_abl_unrolled_act.npy"
    # )
    #
    # intact_top_abl_acts = np.array(abl_unrolled_act)[list(all_ord_sorted[:9])].tolist()
    # intact_bot_abl_acts = np.array(abl_unrolled_act)[list(all_ord_sorted[-9:])].tolist()
    #
    # abl_all_act_list, abl_all_ord_sorted = zip(*sorted(zip(abl_unrolled_act, all_ord_list), reverse=True))
    #
    # #   Perform Spearman rank corr for intact and ablate sorted orders.
    # # print(spearmanr(np.array(all_ord_list), np.array(abl_unrolled_act)[list(all_ord_sorted)]))
    # # exit()
    #
    # top_indices = [abl_all_ord_sorted.index(i) for i in all_ord_sorted[0:9]]
    # bot_indices = [abl_all_ord_sorted.index(i) for i in all_ord_sorted[-9:]]
    #
    # abl_all_act_list = np.delete(np.array(abl_all_act_list), top_indices + bot_indices).tolist()
    # all_ord_list = np.delete(np.array(all_ord_list), top_indices + bot_indices).tolist()
    #
    # plt.scatter(all_ord_list, abl_all_act_list, color='b')
    # plt.scatter(top_indices, intact_top_abl_acts, color='g', label="Intact Top 9")
    # plt.scatter(bot_indices, intact_bot_abl_acts, color='r', label="Intact Bot 9")
    # plt.xlabel("Image")
    # plt.ylabel("Activation")
    # plt.title(f"Inh Ablated Tuning Curve - AlexNet Conv2d10 Unit {selected_neuron}")
    # plt.legend(loc="upper right")
    # plt.savefig(os.path.join('/home/andre/tuning_curves/inh_ablate', 'inh_abl_tuning_curve_l%d_n%d.png'
    #                          % (selected_layer, selected_neuron)))
    # plt.close()

    # ord_list = np.arange(len(act_list)).tolist()
    # all_ord_list = np.arange(len(all_ord_sorted)).tolist()


def pair_plot_ranks(selected_layer, network):
    intact_top9_ranks = []
    intact_top9_errors = []
    abl_top9_ranks = []
    abl_top9_errors = []

    intact_bot9_ranks = []
    intact_bot9_errors = []
    abl_bot9_ranks = []
    abl_bot9_errors = []

    intact_proto_ranks = []
    intact_proto_errors = []
    abl_proto_ranks = []
    abl_proto_errors = []

    intact_antiproto_ranks = []
    intact_antiproto_errors = []
    abl_antiproto_ranks = []
    abl_antiproto_errors = []

    # intact_ticks = []
    # abl_ticks = []
    # tick_idx = []
    # tick_labels = []

    f, axarr = plt.subplots(1, 4, layout='tight', figsize=(8, 5))
    f.suptitle("Intact vs. Positive-Weight Ablated Rank\nTrained ResNet18 Layer 4.1 Conv2, First 21 Units")
    for i in range(4):
        axarr[i].set_xticks([0, 1], ["Intact", "Ablated"])
        axarr[i].set_ylabel("Rank")
        axarr[i].set_ylim(bottom=-1000, top=51000)
        axarr[i].invert_yaxis()

    num_units = 42
    jitter = np.random.normal(scale=0.1, size=num_units)
    alpha = 0.3
    for i in range(num_units):
        # intact_ticks.append(2 * i)
        # abl_ticks.append((2 * i) + 1)
        # tick_idx.append((2 * i + (2 * i) + 1) / 2)
        # tick_labels.append(f'{i}')

        #   Top 9
        intact_ranks = np.load(
            os.path.join(f"/home/andre/rank_data/intact", network, f"layer{selected_layer}_unit{i}_top9_ranks.npy")
        )
        intact_ranks = np.squeeze(intact_ranks, 0)
        intact_rank = np.mean(intact_ranks)
        intact_error = np.std(intact_ranks) / np.sqrt(intact_ranks.shape[0])
        intact_top9_ranks.append(intact_rank)
        intact_top9_errors.append(intact_error)

        abl_ranks = np.load(
            os.path.join(f"/home/andre/rank_data/exc_abl", network, f"layer{selected_layer}_unit{i}_top9_ranks.npy")
        )
        abl_ranks = np.squeeze(abl_ranks, 0)
        abl_rank = np.mean(abl_ranks)
        abl_error = np.std(abl_ranks) / np.sqrt(abl_ranks.shape[0])
        abl_top9_ranks.append(abl_rank)
        abl_top9_errors.append(abl_error)

        axarr[0].plot([jitter[i], 1+jitter[i]], np.array([intact_rank, abl_rank]), c='g', linewidth=0.5, alpha=alpha)

        #   Prototypes.
        intact_ranks = np.load(
            os.path.join(f"/home/andre/rank_data/intact", network, f"layer{selected_layer}_unit{i}_proto_ranks.npy")
        )
        intact_ranks = np.squeeze(intact_ranks, 0)
        intact_rank = np.mean(intact_ranks)
        intact_error = np.std(intact_ranks) / np.sqrt(intact_ranks.shape[0])
        intact_proto_ranks.append(intact_rank)
        intact_proto_errors.append(intact_error)

        abl_ranks = np.load(
            os.path.join(f"/home/andre/rank_data/exc_abl", network, f"layer{selected_layer}_unit{i}_proto_ranks.npy")
        )
        abl_ranks = np.squeeze(abl_ranks, 0)
        abl_rank = np.mean(abl_ranks)
        abl_error = np.std(abl_ranks) / np.sqrt(abl_ranks.shape[0])
        abl_proto_ranks.append(abl_rank)
        abl_proto_errors.append(abl_error)

        axarr[1].plot([jitter[i], 1+jitter[i]], np.array([intact_rank, abl_rank]), c='g', linewidth=0.5, alpha=alpha)

        #   Bot 9
        intact_ranks = np.load(
            os.path.join(f"/home/andre/rank_data/intact", network, f"layer{selected_layer}_unit{i}_bot9_ranks.npy")
        )
        intact_ranks = np.squeeze(intact_ranks, 0)
        intact_rank = np.mean(intact_ranks)
        intact_error = np.std(intact_ranks) / np.sqrt(intact_ranks.shape[0])
        intact_bot9_ranks.append(intact_rank)
        intact_bot9_errors.append(intact_error)

        abl_ranks = np.load(
            os.path.join(f"/home/andre/rank_data/exc_abl", network, f"layer{selected_layer}_unit{i}_bot9_ranks.npy")
        )
        abl_ranks = np.squeeze(abl_ranks, 0)
        abl_rank = np.mean(abl_ranks)
        abl_error = np.std(abl_ranks) / np.sqrt(abl_ranks.shape[0])
        abl_bot9_ranks.append(abl_rank)
        abl_bot9_errors.append(abl_error)

        axarr[2].plot([jitter[i], 1+jitter[i]], np.array([intact_rank, abl_rank]), c='r', linewidth=0.5, alpha=alpha)

        #   Anti-prototypes.
        intact_ranks = np.load(
            os.path.join(f"/home/andre/rank_data/intact", network, f"layer{selected_layer}_unit{i}_antiproto_ranks.npy")
        )
        intact_ranks = np.squeeze(intact_ranks, 0)
        intact_rank = np.mean(intact_ranks)
        intact_error = np.std(intact_ranks) / np.sqrt(intact_ranks.shape[0])
        intact_antiproto_ranks.append(intact_rank)
        intact_antiproto_errors.append(intact_error)

        abl_ranks = np.load(
            os.path.join(f"/home/andre/rank_data/exc_abl", network, f"layer{selected_layer}_unit{i}_antiproto_ranks.npy")
        )
        abl_ranks = np.squeeze(abl_ranks, 0)
        abl_rank = np.mean(abl_ranks)
        abl_error = np.std(abl_ranks) / np.sqrt(abl_ranks.shape[0])
        abl_antiproto_ranks.append(abl_rank)
        abl_antiproto_errors.append(abl_error)

        axarr[3].plot([jitter[i], 1+jitter[i]], np.array([intact_rank, abl_rank]), c='r', linewidth=0.5, alpha=alpha)

    axarr[0].set_title("Top 9")
    axarr[0].errorbar(
        jitter, np.array(intact_top9_ranks), np.array(intact_top9_errors),
        fmt='go', alpha=alpha
    )
    axarr[0].errorbar(
        np.ones(len(abl_top9_ranks)) + jitter, np.array(abl_top9_ranks), np.array(abl_top9_errors),
        fmt='go', alpha=alpha

    )
    axarr[1].set_title("Prototypes")
    axarr[1].errorbar(
        jitter, np.array(intact_proto_ranks), np.array(intact_proto_errors),
        fmt='go', alpha=alpha
    )
    axarr[1].errorbar(
        np.ones(len(abl_proto_ranks)) + jitter, np.array(abl_proto_ranks), np.array(abl_proto_errors),
        fmt='go', alpha=alpha
    )

    axarr[2].set_title("Bot 9")
    axarr[2].errorbar(
        jitter, np.array(intact_bot9_ranks), np.array(intact_bot9_errors),
        fmt='ro', alpha=alpha
    )
    axarr[2].errorbar(
        np.ones(len(abl_bot9_ranks)) + jitter, np.array(abl_bot9_ranks), np.array(abl_bot9_errors),
        fmt='ro', alpha=alpha
    )

    axarr[3].set_title("Anti-prototypes")
    axarr[3].errorbar(
        jitter, np.array(intact_antiproto_ranks), np.array(intact_antiproto_errors),
        fmt='ro', alpha=alpha
    )
    axarr[3].errorbar(
        np.ones(len(abl_antiproto_ranks)) + jitter, np.array(abl_antiproto_ranks), np.array(abl_antiproto_errors),
        fmt='ro', alpha=alpha
    )

    print(f"Mean proto:  {np.mean(np.array(abl_proto_ranks))},  Mean top9:  {np.mean(np.array(abl_top9_ranks))}")
    print(f"Proto - Top 9 Rank Diff:      {np.mean(np.array(abl_proto_ranks)) - np.mean(np.array(abl_top9_ranks))}")
    print(f_oneway(np.array(abl_proto_ranks), np.array(abl_top9_ranks)))

    print(f"Mean antiproto:  {np.mean(np.array(abl_antiproto_ranks))},  Mean bot9:  {np.mean(np.array(abl_bot9_ranks))}")
    print(f"Antiproto - Bot 9 Rank Diff:  {np.mean(np.array(abl_antiproto_ranks)) - np.mean(np.array(abl_bot9_ranks))}")
    print(f_oneway(np.array(abl_antiproto_ranks), np.array(abl_bot9_ranks)))

    plt.savefig("/home/andre/rank_data/imnet_val_rank_intact_v_exc_abl_trained_resnet18_layer4.1.Conv2dconv2_100.png", dpi=256)
    plt.close()


def plot_ranks(selected_layer):
    trained_top9_ranks = []
    untrained_top9_ranks = []
    trained_top9_errors = []
    untrained_top9_errors = []

    trained_bot9_ranks = []
    untrained_bot9_ranks = []
    trained_bot9_errors = []
    untrained_bot9_errors = []

    trained_proto_ranks = []
    untrained_proto_ranks = []
    trained_proto_errors = []
    untrained_proto_errors = []

    trained_antiproto_ranks = []
    untrained_antiproto_ranks = []
    trained_antiproto_errors = []
    untrained_antiproto_errors = []

    for i in range(10):
        #   Trained
        rank = np.load(f"/home/andre/rank_data/inh_abl/alexnet_trained/layer{selected_layer}_unit{i}_inh_abl_top9_ranks.npy")
        rank = np.squeeze(rank, 0)
        trained_top9_ranks.append(np.mean(rank))
        trained_top9_errors.append(np.std(rank) / np.sqrt(rank.shape[0]))

        rank = np.load(f"/home/andre/rank_data/inh_abl/alexnet_trained/layer{selected_layer}_unit{i}_inh_abl_bot9_ranks.npy")
        rank = np.squeeze(rank, 0)
        trained_bot9_ranks.append(np.mean(rank))
        trained_bot9_errors.append(np.std(rank) / np.sqrt(rank.shape[0]))

        rank = np.load(f"/home/andre/rank_data/inh_abl/alexnet_trained/layer{selected_layer}_unit{i}_inh_abl_proto_ranks.npy")
        rank = np.squeeze(rank, 0)
        trained_proto_ranks.append(np.mean(rank))
        trained_proto_errors.append(np.std(rank) / np.sqrt(rank.shape[0]))

        rank = np.load(f"/home/andre/rank_data/inh_abl/alexnet_trained/layer{selected_layer}_unit{i}_inh_abl_antiproto_ranks.npy")
        rank = np.squeeze(rank, 0)
        trained_antiproto_ranks.append(np.mean(rank))
        trained_antiproto_errors.append(np.std(rank) / np.sqrt(rank.shape[0]))

        #   Untrained
        rank = np.load(f"/home/andre/rank_data/inh_abl/alexnet_untrained/layer{selected_layer}_unit{i}_inh_abl_top9_ranks.npy")
        rank = np.squeeze(rank, 0)
        untrained_top9_ranks.append(np.mean(rank))
        untrained_top9_errors.append(np.std(rank) / np.sqrt(rank.shape[0]))

        rank = np.load(f"/home/andre/rank_data/inh_abl/alexnet_untrained/layer{selected_layer}_unit{i}_inh_abl_bot9_ranks.npy")
        rank = np.squeeze(rank, 0)
        untrained_bot9_ranks.append(np.mean(rank))
        untrained_bot9_errors.append(np.std(rank) / np.sqrt(rank.shape[0]))

        rank = np.load(f"/home/andre/rank_data/inh_abl/alexnet_untrained/layer{selected_layer}_unit{i}_inh_abl_proto_ranks.npy")
        rank = np.squeeze(rank, 0)
        untrained_proto_ranks.append(np.mean(rank))
        untrained_proto_errors.append(np.std(rank) / np.sqrt(rank.shape[0]))

        rank = np.load(f"/home/andre/rank_data/inh_abl/alexnet_untrained/layer{selected_layer}_unit{i}_inh_abl_antiproto_ranks.npy")
        rank = np.squeeze(rank, 0)
        untrained_antiproto_ranks.append(np.mean(rank))
        untrained_antiproto_errors.append(np.std(rank) / np.sqrt(rank.shape[0]))

    plt.errorbar(np.arange(10), np.array(trained_top9_ranks), np.array(trained_top9_errors), label='Tr Top 9', fmt='g+')
    plt.errorbar(np.arange(10), np.array(trained_bot9_ranks), np.array(trained_bot9_errors), label='Tr Bot 9', fmt='gx')
    plt.errorbar(np.arange(10), np.array(trained_proto_ranks), np.array(trained_proto_errors), label='Tr Proto', fmt='go')
    plt.errorbar(np.arange(10), np.array(trained_antiproto_ranks), np.array(trained_antiproto_errors), label='Tr Antiproto', fmt='gs')

    plt.errorbar(np.arange(10), np.array(untrained_top9_ranks), np.array(untrained_top9_errors), label='UTr Top 9', fmt='r+')
    plt.errorbar(np.arange(10), np.array(untrained_bot9_ranks), np.array(untrained_bot9_errors), label='UTr Bot 9', fmt='rx')
    plt.errorbar(np.arange(10), np.array(untrained_proto_ranks), np.array(untrained_proto_errors), label='UTr Proto', fmt='ro')
    plt.errorbar(np.arange(10), np.array(untrained_antiproto_ranks), np.array(untrained_antiproto_errors), label='UTr Antiproto', fmt='rs')

    plt.legend(loc="upper right")
    plt.xlabel("Unit")
    plt.ylabel("Rank")
    plt.title("Imnet Val Rank - Inh Ablated AlexNet Conv2d3")
    plt.savefig("/home/andre/rank_data/imnet_val_rank_inh_abl_alexnet_conv2d3.png")
    plt.close()


def make_stimuli_grid():
    f, axarr = plt.subplots(4, 10)

    for i in range(10):
        axarr[0, i].set_axis_off()
        # axarr[0, i].set_title(f"Unit {i} Top 9")
        axarr[0, i].imshow(Image.open(f'/home/andre/tuning_curves/untrained_alexnet/layer11_neuron{i}/exc_grid.png'))

        axarr[1, i].set_axis_off()
        # axarr[1, i].set_title(f"Unit {i} Bot 9")
        axarr[1, i].imshow(Image.open(f'/home/andre/tuning_curves/untrained_alexnet/layer11_neuron{i}/inh_grid.png'))

        axarr[2, i].set_axis_off()
        # axarr[2, i].set_title(f"Unit {i} Prototypes")
        axarr[2, i].imshow(Image.open(f'/home/andre/evolve_grid/alexnet_untrained/unit{i}_all_Exc_Prototype_alexnet_untrained_.features.Conv2d10.png'))

        axarr[3, i].set_axis_off()
        # axarr[3, i].set_title(f"Unit {i} Antiprototypes")
        axarr[3, i].imshow(Image.open(f'/home/andre/evolve_grid/alexnet_untrained/unit{i}_all_Inh_Prototype_alexnet_untrained_.features.Conv2d10.png'))

    plt.subplots_adjust(wspace=0.75, hspace=-0.85)
    plt.savefig("/home/andre/rank_data/alexnet_untrained_stimuli.png", dpi=750, bbox_inches='tight')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str, required=False)
    parser.add_argument('--neuron', type=int)
    parser.add_argument('--layer', type=str, required=False)
    parser.add_argument('--selected_layer', type=int, required=False)
    parser.add_argument('--savedir', type=str)
    parser.add_argument('--protodir', type=str)
    parser.add_argument('--inh_abl', action='store_true', default=False)
    args = parser.parse_args()

    layer = args.layer
    selected_layer = args.selected_layer
    selected_neuron = args.neuron

    model_name = None
    neuron_coord = None
    module_name = None
    states = None

    curvedir = None
    ranksdir = None
    topdir = None
    botdir = None
    protodir = None
    antiprotodir = None
    actsdir = None
    if args.network == 'alexnet':
        model_name = 'alexnet'
        neuron_coord = 6
        module_name = 'features.10'
        if args.inh_abl:
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

    elif args.network == 'alexnet-untrained':
        model_name = 'alexnet'
        neuron_coord = 6
        module_name = 'features.10'
        states = torch.load("/home/andre/tuning_curves/untrained_alexnet/random_weights.pth")

        if args.inh_abl:
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

    elif args.network == 'resnet18':
        model_name = 'resnet18'
        neuron_coord = 3
        module_name = 'layer4.1.conv2'

        if args.inh_abl:
            curvedir = '/home/andre/tuning_curves/resnet18/inh_abl'
            ranksdir = '/home/andre/rank_data/inh_abl/resnet18_trained'

            actsdir = os.path.join(curvedir, f'layer{selected_layer}_unit{selected_neuron}_inh_abl_unrolled_act.npy')
        else:
            curvedir = '/home/andre/tuning_curves/resnet18/intact'
            ranksdir = '/home/andre/rank_data/intact/resnet18_trained'

            actsdir = os.path.join(curvedir, f'layer{selected_layer}_unit{selected_neuron}_intact_unrolled_act.npy')

        topdir = f'/home/andre/tuning_curves/resnet18/intact/layer{selected_layer}_neuron{selected_neuron}/max/exc'
        botdir = f'/home/andre/tuning_curves/resnet18/intact/layer{selected_layer}_neuron{selected_neuron}/max/inh'
        protodir = f'/home/andre/evolved_data/resnet18_.layer4.1.Conv2dconv2_unit{selected_neuron}/exc/no_mask'
        antiprotodir = f'/home/andre/evolved_data/resnet18_.layer4.1.Conv2dconv2_unit{selected_neuron}/inh/no_mask'

    elif args.network == 'resnet18-untrained':
        model_name = 'resnet18'
        neuron_coord = 3
        module_name = 'layer4.1.conv2'
        states = torch.load("/home/andre/tuning_curves/untrained_resnet18/random_weights.pth")

        if args.inh_abl:
            curvedir = '/home/andre/tuning_curves/untrained_resnet18/inh_abl'
            ranksdir = '/home/andre/rank_data/inh_abl/resnet18_untrained'

            actsdir = os.path.join(curvedir, f'layer{selected_layer}_unit{selected_neuron}_inh_abl_unrolled_act.npy')
        else:
            curvedir = '/home/andre/tuning_curves/untrained_resnet18/intact'
            ranksdir = '/home/andre/rank_data/intact/resnet18_untrained'

            actsdir = os.path.join(curvedir, f'layer{selected_layer}_unit{selected_neuron}_intact_unrolled_act.npy')

        topdir = f'/home/andre/tuning_curves/untrained_resnet18/intact/layer{selected_layer}_neuron{selected_neuron}/max/exc'
        botdir = f'/home/andre/tuning_curves/untrained_resnet18/intact/layer{selected_layer}_neuron{selected_neuron}/max/inh'
        protodir = f'/home/andre/evolved_data/resnet18_untrained_.layer4.1.Conv2dconv2_unit{selected_neuron}/exc/no_mask'
        antiprotodir = f'/home/andre/evolved_data/resnet18_untrained_.layer4.1.Conv2dconv2_unit{selected_neuron}/inh/no_mask'

    elif args.network == 'resnet18-robust':
        model_name = 'resnet18'
        neuron_coord = 3
        module_name = 'layer4.1.conv2'
        states = torch.load("/home/andre/model_weights/resnet-18-l2-eps3.pt")

        if args.inh_abl:
            curvedir = '/home/andre/tuning_curves/resnet18_robust/inh_abl'
            ranksdir = '/home/andre/rank_data/inh_abl/resnet18_robust'

            actsdir = os.path.join(curvedir, f'layer{selected_layer}_unit{selected_neuron}_inh_abl_unrolled_act.npy')
        else:
            curvedir = '/home/andre/tuning_curves/resnet18_robust/intact'
            ranksdir = '/home/andre/rank_data/intact/resnet18_robust'

            actsdir = os.path.join(curvedir, f'layer{selected_layer}_unit{selected_neuron}_intact_unrolled_act.npy')

        topdir = f'/home/andre/tuning_curves/resnet18_robust/intact/layer{selected_layer}_neuron{selected_neuron}/max/exc'
        botdir = f'/home/andre/tuning_curves/resnet18_robust/intact/layer{selected_layer}_neuron{selected_neuron}/max/inh'
        protodir = f'/home/andre/evolved_data/resnet18_robust_.layer4.1.Conv2dconv2_unit{selected_neuron}/exc/no_mask'
        antiprotodir = f'/home/andre/evolved_data/resnet18_robust_.layer4.1.Conv2dconv2_unit{selected_neuron}/inh/no_mask'

    extractor = get_extractor(
        model_name=model_name,
        source='torchvision',
        device='cuda',
        pretrained=True
    )

    if states is not None:
        extractor.model.load_state_dict(states)

    compare_tuning_curves(extractor.model, curvedir, layer, selected_layer, selected_neuron, module_name,
                          inh_abl=args.inh_abl, extractor=extractor, neuron_coord=neuron_coord)

    top_ranks, bot_ranks, proto_ranks, antiproto_ranks = extract_ranks(extractor, topdir, botdir, protodir,
                                                                       antiprotodir,
                                                                       layer, selected_layer, selected_neuron,
                                                                       module_name,
                                                                       inh_abl=args.inh_abl, actsdir=actsdir)

    np.save(os.path.join(ranksdir, f"layer{selected_layer}_unit{selected_neuron}_top9_ranks.npy"), np.array(top_ranks))
    np.save(os.path.join(ranksdir, f"layer{selected_layer}_unit{selected_neuron}_bot9_ranks.npy"), np.array(bot_ranks))
    np.save(os.path.join(ranksdir, f"layer{selected_layer}_unit{selected_neuron}_proto_ranks.npy"),
            np.array(proto_ranks))
    np.save(os.path.join(ranksdir, f"layer{selected_layer}_unit{selected_neuron}_antiproto_ranks.npy"),
            np.array(antiproto_ranks))


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