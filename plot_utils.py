import numpy as np
import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models
from scipy.stats import spearmanr, pearsonr, f_oneway, ttest_ind

import sys
sys.path.append('/home/andrelongon/Documents/inhibition_code/jsputils')

from train_nonneg_weights import AlexNet


def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def pair_plot_intact_abl_norms(layers, network):
    savedir = "/home/andre/max_stim_inputs/intact_v_ablate"

    f, axarr = plt.subplots(1, len(layers), layout='tight', figsize=(12, 5))
    #   Put axarr in list if len(layers) == 1?
    f.suptitle("Intact vs. Ablated Prototype Intact Activations\nTrained AlexNet First 32 Units")
    # f.suptitle("Intact vs. Ablated Mean L1 Norm of Positive-Position Input Tensor\nTrained AlexNet Layer Conv3, First 32 Units")

    # axarr[1].set_xticks([0, 1], ["Bot 9", "Anti-prototypes"])
    # axarr[2].set_xticks([0, 1], ["Top 9", "Positive Lucents"])
    # axarr[3].set_xticks([0, 1], ["Bot 9", "Negative Lucents"])

    num_units = 32
    jitter = np.random.normal(scale=0.1, size=num_units)
    alpha = 0.3

    for l in range(len(layers)):
        intact_proto_all_acts = []
        intact_proto_all_errors = []

        ablate_proto_all_acts = []
        ablate_proto_all_errors = []

        layer = layers[l]
        axarr[l].set_title(f"Layer {layer}")
        axarr[l].set_xticks([0, 1], ["Intact", "Ablated"])

        for i in range(num_units):
            intact_proto_acts = np.load(
                os.path.join(savedir, network, f"layer{layer}_unit{i}_intact_proto_intact_act.npy")
            )
            intact_proto_mean = np.mean(intact_proto_acts)
            # print(f"Unit {i} proto act: {intact_proto_mean}")
            intact_proto_error = np.std(intact_proto_acts) / np.sqrt(intact_proto_acts.shape[0])
            intact_proto_all_acts.append(intact_proto_mean)
            intact_proto_all_errors.append(intact_proto_error)

            ablate_proto_acts = np.load(
                os.path.join(savedir, network, f"layer{layer}_unit{i}_ablated_proto_intact_act.npy")
            )
            ablate_proto_mean = np.mean(ablate_proto_acts)
            ablate_proto_error = np.std(ablate_proto_acts) / np.sqrt(ablate_proto_acts.shape[0])
            ablate_proto_all_acts.append(ablate_proto_mean)
            ablate_proto_all_errors.append(ablate_proto_error)

            #   Plot line which connects the two.
            axarr[l].plot([jitter[i], 1+jitter[i]], np.array([intact_proto_mean, ablate_proto_mean]), c='g', linewidth=0.5, alpha=alpha)

        axarr[l].errorbar(
            jitter, np.array(intact_proto_all_acts), np.array(intact_proto_all_errors),
            fmt='go', alpha=alpha
        )
        axarr[l].errorbar(
            np.ones(len(ablate_proto_all_acts)) + jitter, np.array(ablate_proto_all_acts), np.array(ablate_proto_all_errors),
            fmt='go', alpha=alpha
        )

        all_acts = np.concatenate(
            (intact_proto_all_acts, ablate_proto_all_acts)
        )
        act_range = np.max(all_acts) - np.min(all_acts)
        axarr[l].set_ylabel("Activation")
        axarr[l].set_ylim(bottom=np.min(all_acts) - (0.1 * act_range), top=np.max(all_acts) + (0.1 * act_range))

    plt.savefig(os.path.join(savedir, "intact_v_ablate_proto_intact_act_trained_alexnet_32.png"), dpi=256)
    plt.close()


def pair_plot_acts(selected_layer, network):
    savedir = "/home/andre/max_stim_inputs/split/pos"

    top9_all_acts = []
    top9_all_errors = []

    bot9_all_acts = []
    bot9_all_errors = []

    proto_all_acts = []
    proto_all_errors = []

    antiproto_all_acts = []
    antiproto_all_errors = []

    poslucent_all_acts = []
    poslucent_all_errors = []

    neglucent_all_acts = []
    neglucent_all_errors = []

    f, axarr = plt.subplots(1, 4, layout='tight', figsize=(12, 5))
    f.suptitle("Mean L1 Norm of Positive-Position Input Tensor\nUntrained ResNet18 Layer 4.1 Conv2, First 32 Units")
    axarr[0].set_xticks([0, 1], ["Top 9", "Prototypes"])
    axarr[1].set_xticks([0, 1], ["Bot 9", "Anti-prototypes"])
    axarr[2].set_xticks([0, 1], ["Top 9", "Positive Lucents"])
    axarr[3].set_xticks([0, 1], ["Bot 9", "Negative Lucents"])

    num_units = 32
    jitter = np.random.normal(scale=0.1, size=num_units)
    alpha = 0.3

    for i in range(num_units):
        #   Top 9
        top9_acts = np.load(
            os.path.join(savedir, network, f"layer{selected_layer}_unit{i}_top_norms.npy")
        )
        top9_mean = np.mean(top9_acts)
        top9_error = np.std(top9_acts) / np.sqrt(top9_acts.shape[0])
        top9_all_acts.append(top9_mean)
        top9_all_errors.append(top9_error)

        #   Prototype
        proto_acts = np.load(
            os.path.join(savedir, network, f"layer{selected_layer}_unit{i}_proto_norms.npy")
        )
        proto_mean = np.mean(proto_acts)
        proto_error = np.std(proto_acts) / np.sqrt(proto_acts.shape[0])
        proto_all_acts.append(proto_mean)
        proto_all_errors.append(proto_error)

        #   Plot line which connects the two.
        axarr[0].plot([jitter[i], 1+jitter[i]], np.array([top9_mean, proto_mean]), c='g', linewidth=0.5, alpha=alpha)

        #   Bot 9
        bot9_acts = np.load(
            os.path.join(savedir, network, f"layer{selected_layer}_unit{i}_bot_norms.npy")
        )
        bot9_mean = np.mean(bot9_acts)
        bot9_error = np.std(bot9_acts) / np.sqrt(bot9_acts.shape[0])
        bot9_all_acts.append(bot9_mean)
        bot9_all_errors.append(bot9_error)

        #   Anti-Prototype
        antiproto_acts = np.load(
            os.path.join(savedir, network, f"layer{selected_layer}_unit{i}_antiproto_norms.npy")
        )
        antiproto_mean = np.mean(antiproto_acts)
        antiproto_error = np.std(antiproto_acts) / np.sqrt(antiproto_acts.shape[0])
        antiproto_all_acts.append(antiproto_mean)
        antiproto_all_errors.append(antiproto_error)

        #   Plot line which connects the two.
        axarr[1].plot([jitter[i], 1 + jitter[i]], np.array([bot9_mean, antiproto_mean]), c='r', linewidth=0.5, alpha=alpha)

        #   Positive Lucents
        poslucent_acts = np.load(
            os.path.join(savedir, network, f"layer{selected_layer}_unit{i}_poslucent_norms.npy")
        )
        poslucent_mean = np.mean(poslucent_acts)
        poslucent_error = np.std(poslucent_acts) / np.sqrt(poslucent_acts.shape[0])
        poslucent_all_acts.append(poslucent_mean)
        poslucent_all_errors.append(poslucent_error)

        #   Plot line which connects the two.
        axarr[2].plot([jitter[i], 1 + jitter[i]], np.array([top9_mean, poslucent_mean]), c='g', linewidth=0.5, alpha=alpha)

        #   Negative Lucents
        neglucent_acts = np.load(
            os.path.join(savedir, network, f"layer{selected_layer}_unit{i}_neglucent_norms.npy")
        )
        neglucent_mean = np.mean(neglucent_acts)
        neglucent_error = np.std(neglucent_acts) / np.sqrt(neglucent_acts.shape[0])
        neglucent_all_acts.append(neglucent_mean)
        neglucent_all_errors.append(neglucent_error)

        #   Plot line which connects the two.
        axarr[3].plot([jitter[i], 1 + jitter[i]], np.array([bot9_mean, neglucent_mean]), c='r', linewidth=0.5, alpha=alpha)

    axarr[0].errorbar(
        jitter, np.array(top9_all_acts), np.array(top9_all_errors),
        fmt='go', alpha=alpha
    )
    axarr[0].errorbar(
        np.ones(len(proto_all_acts)) + jitter, np.array(proto_all_acts), np.array(proto_all_errors),
        fmt='go', alpha=alpha
    )

    axarr[1].errorbar(
        jitter, np.array(bot9_all_acts), np.array(bot9_all_errors),
        fmt='ro', alpha=alpha
    )
    axarr[1].errorbar(
        np.ones(len(antiproto_all_acts)) + jitter, np.array(antiproto_all_acts), np.array(antiproto_all_errors),
        fmt='ro', alpha=alpha
    )

    axarr[2].errorbar(
        jitter, np.array(top9_all_acts), np.array(top9_all_errors),
        fmt='go', alpha=alpha
    )
    axarr[2].errorbar(
        np.ones(len(poslucent_all_acts)) + jitter, np.array(poslucent_all_acts), np.array(poslucent_all_errors),
        fmt='go', alpha=alpha
    )

    axarr[3].errorbar(
        jitter, np.array(bot9_all_acts), np.array(bot9_all_errors),
        fmt='ro', alpha=alpha
    )
    axarr[3].errorbar(
        np.ones(len(neglucent_all_acts)) + jitter, np.array(neglucent_all_acts), np.array(neglucent_all_errors),
        fmt='ro', alpha=alpha
    )

    all_acts = np.concatenate(
        (top9_all_acts, bot9_all_acts, proto_all_acts, antiproto_all_acts, poslucent_all_acts, neglucent_all_acts)
    )
    act_range = np.max(all_acts) - np.min(all_acts)
    for ax in axarr:
        ax.set_ylabel("Norm Mean")
        ax.set_ylim(bottom=np.min(all_acts) - (0.1 * act_range), top=np.max(all_acts) + (0.1 * act_range))

    plt.savefig(os.path.join(savedir, "l1_norms_untrained_resnet18_layers4.1.conv2_32.png"), dpi=256)
    plt.close()


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
    f.suptitle("Intact vs. Negative-Weight Ablated Rank\nTrained AlexNet Layer Conv2d2, First 10 Units")
    # f.suptitle("Intact vs. Negative-Weight Ablated Rank\nRobust ResNet18 Layer 4.1 Conv2, First 100 Units")
    for i in range(4):
        axarr[i].set_xticks([0, 1], ["Intact", "Ablated"])
        axarr[i].set_ylabel("Rank")
        axarr[i].set_ylim(bottom=-1000, top=51000)
        axarr[i].invert_yaxis()

    num_units = 10
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
            os.path.join(f"/home/andre/rank_data/inh_abl", network, f"layer{selected_layer}_unit{i}_top9_ranks.npy")
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
            os.path.join(f"/home/andre/rank_data/inh_abl", network, f"layer{selected_layer}_unit{i}_proto_ranks.npy")
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
            os.path.join(f"/home/andre/rank_data/inh_abl", network, f"layer{selected_layer}_unit{i}_bot9_ranks.npy")
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
            os.path.join(f"/home/andre/rank_data/inh_abl", network, f"layer{selected_layer}_unit{i}_antiproto_ranks.npy")
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
    print(ttest_ind(np.array(abl_proto_ranks), np.array(abl_top9_ranks), equal_var=False))

    print(f"Mean antiproto:  {np.mean(np.array(abl_antiproto_ranks))},  Mean bot9:  {np.mean(np.array(abl_bot9_ranks))}")
    print(f"Antiproto - Bot 9 Rank Diff:  {np.mean(np.array(abl_antiproto_ranks)) - np.mean(np.array(abl_bot9_ranks))}")
    print(ttest_ind(np.array(abl_antiproto_ranks), np.array(abl_bot9_ranks), equal_var=False))

    plt.savefig("/home/andre/rank_data/imnet_val_rank_intact_v_inh_abl_trained_alexnet_conv2d2_10.png", dpi=256)
    # plt.savefig("/home/andre/rank_data/imnet_val_rank_intact_v_inh_abl_robust_resnet18_layer4.1.Conv2dconv2_100.png", dpi=256)
    plt.close()


def plot_weight_histos():
    # model = AlexNet()
    # model.load_state_dict(torch.load("/home/andrelongon/Documents/inhibition_code/weights/cifar_alexnet_nonneg_3_batchnorm.pth"))

    # model = models.vgg16(True)
    # model, transform = torch.hub.load("harvard-visionlab/open_ipcl", "alexnetgn_ipcl_ref06_diet_imagenet")
    #  keys for barlow:  'backbone.1.0.weight', 'backbone.2.0.weight', 'backbone.3.0.weight', 'backbone.4.0.weight'
    # model = torch.load('/home/andrelongon/Documents/inhibition_code/jsputils/alexnet-barlow-twins.pth')
    # model = models.alexnet(True)

    from thingsvision import get_extractor

    extractor = get_extractor(
        model_name='cornet-s',
        source='custom',
        device='cpu',
        pretrained=True
    )
    model = extractor.model

    states = model.state_dict()

    # model.load_state_dict(torch.load("/home/andre/model_weights/resnet-18-l2-eps3.pt"))

    #   Keys: 'conv2_2.weight', 'conv3_2.weight', 'conv4_2.weight', 'conv5_2.weight'
    # states = torch.load('/home/andrelongon/Documents/inhibition_code/vgg_face_dag.pth')

    # state_keys = [
    #     'layer1.1.conv2.weight', 'layer2.1.conv2.weight', 'layer3.1.conv2.weight', 'layer4.1.conv2.weight'
    # ]

    # state_keys = ['features.3.weight', 'features.7.weight', 'features.9.weight', 'features.11.weight']

    state_keys = ['V1.conv2.weight', 'V2.conv3.weight', 'V4.conv3.weight', 'IT.conv3.weight']

    f, axarr = plt.subplots(1, len(state_keys), layout='tight', figsize=(12, 5))
    f.suptitle("Weight Histogram Across Layers\nCORNet-S")
    # f.suptitle("Weight Histogram Across Layers\nCIFAR AlexNet with BatchNorm Conv 3 Constrained\nTop 1: 74%")
    # max = 0

    for i in range(len(state_keys)):
        weights = torch.flatten(states[state_keys[i]])
        bins = [-0.1  , -0.075, -0.05 , -0.025,  0.   ,  0.025,  0.05 ,  0.075, 0.1]

        hist = axarr[i].hist(weights, bins=bins)[0]
        axarr[i].set_title(state_keys[i])
        print(hist / np.sum(hist))

    #     if np.max(hist) > max:
    #         max = np.max(hist)
    #
    # for i in range(4):
    #     axarr[i].set_ylim(bottom=0, top=max)

    plt.savefig("/home/andrelongon/Documents/inhibition_code/plots/weight_histos/cornet-s.png")


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


def max_projected_curve_plot(layer, neuron):
    # Find indices in all_ord_list where all_ord_sorted[0:9] occurs in abl_all_ord_sorted.
    #   Index of abl_all_ord_sorted where it equals all_ord_sorted[0:9].  These indices
    #   can then be deleted from abl_all_act_list.
    all_ord_sorted = np.load(
        f"/home/andre/tuning_curves/resnet18/intact/layer{layer}_unit{neuron}_intact_all_ord_sorted.npy"
    )
    all_ord_list = np.arange(len(all_ord_sorted)).tolist()
    abl_unrolled_act = np.load(
        f"/home/andre/tuning_curves/resnet18/exc_abl/layer{layer}_unit{neuron}_exc_abl_unrolled_act.npy"
    )

    intact_top_abl_acts = np.array(abl_unrolled_act)[list(all_ord_sorted[:9])].tolist()
    intact_bot_abl_acts = np.array(abl_unrolled_act)[list(all_ord_sorted[-9:])].tolist()

    abl_all_act_list, abl_all_ord_sorted = zip(*sorted(zip(abl_unrolled_act, all_ord_list), reverse=True))

    #   Perform Spearman rank corr for intact and ablate sorted orders.
    # print(spearmanr(np.array(all_ord_list), np.array(abl_unrolled_act)[list(all_ord_sorted)]))
    # exit()

    top_indices = [abl_all_ord_sorted.index(i) for i in all_ord_sorted[0:9]]
    bot_indices = [abl_all_ord_sorted.index(i) for i in all_ord_sorted[-9:]]

    abl_all_act_list = np.delete(np.array(abl_all_act_list), top_indices + bot_indices).tolist()
    all_ord_list = np.delete(np.array(all_ord_list), top_indices + bot_indices).tolist()

    plt.scatter(all_ord_list, abl_all_act_list, color='b')
    plt.scatter(top_indices, intact_top_abl_acts, color='g', label="Intact Top 9")
    plt.scatter(bot_indices, intact_bot_abl_acts, color='r', label="Intact Bot 9")
    plt.xlabel("Image")
    plt.ylabel("Activation")
    plt.title(f"Positive-Weight Ablated Tuning Curve - Trained ResNet18 Layer 4.1 Conv2 Unit {neuron}")
    plt.legend(loc="upper right")
    plt.savefig(os.path.join('/home/andre/tuning_curves/alexnet/inh_abl', 'neg_abl_tuning_curve_l%d_n%d_new.png'
                             % (layer, neuron)))
    plt.close()


def curve_corr_plot(same_corrs, all_exc_mean, all_inh_mean, all_exc_err, all_inh_err):
    plt.scatter(np.arange(same_corrs.shape[0]), same_corrs, color='b')#, label="Same Neuron Opposite")
    # plt.errorbar(
    #     np.arange(all_exc_mean.shape[0]), all_exc_mean, all_exc_err, fmt='go',
    #     label="Other Intact Neuron Mean"
    # )
    # plt.errorbar(
    #     np.arange(all_inh_mean.shape[0]), all_inh_mean, all_inh_err, fmt='ro',
    #     label="Other Neuron Inh Mean"
    # )
    plt.xlabel("Neuron")
    plt.ylabel("Pearson R")
    plt.title(f"Intact vs Neg-Only Tuning Curve Correlation\nRobust ResNet18 layer4.1.Conv2dconv2 - First 64 Neurons")
    # plt.legend(loc="lower right")
    plt.savefig("/home/andrelongon/Documents/inhibition_code/plots/intact_v_negative-only/intact_v_no_curve_corr_robust_resnet18_layer4.1.Conv2dconv2_first_64.png")
    plt.close()


def tuning_curve_plot(layer, neuron):
    # curve = np.load(
    #     f"/home/andrelongon/Documents/inhibition_code/tuning_curves/alexnet/{layer}/exc_abl/{layer}_unit{neuron}_exc_abl_all_act_list.npy"
    # )

    # all_ord_list = np.arange(curve.shape[0]).tolist()

    # plt.scatter(all_ord_list, curve, color='r')
    # plt.xlabel("Image")
    # plt.ylabel("Activation")
    # plt.title(f"Neg-Only Tuning Curve\nTrained AlexNet Layer {layer} Neuron {neuron}")
    # plt.savefig(f'/home/andrelongon/Documents/inhibition_code/plots/negative-only_curves/no_curve_trained_alexnet_{layer}_unit{neuron}.png')
    # plt.close()

    intact_curve = np.load(
        f"/home/andrelongon/Documents/inhibition_code/tuning_curves/resnet18/{layer}/intact/{layer}_unit{neuron}_intact_all_act_list.npy"
    )
    intact_order = np.load(f"/home/andrelongon/Documents/inhibition_code/tuning_curves/resnet18/{layer}/intact/{layer}_unit{neuron}_intact_all_ord_sorted.npy")

    po_curve = np.load(
        f"/home/andrelongon/Documents/inhibition_code/tuning_curves/resnet18/{layer}/inh_abl/{layer}_unit{neuron}_inh_abl_unrolled_act.npy"
    )

    #   Obtain intact pos-only subset (every 100 ranks)
    # intact_curve = intact_curve[np.nonzero(intact_curve > 0)]
    # intact_order = intact_order[:intact_curve.shape[0]]
    # intact_order = intact_order[::100]
    # intact_curve = intact_curve[::100]

    po_curve = po_curve[intact_order]

    f_stat, p_val = f_oneway(intact_curve, po_curve)
    print(f"ANOVA results on subset: F Score={f_stat}, p={p_val}")

    all_ord_list = np.arange(po_curve.shape[0]).tolist()

    fig = plt.figure(figsize=(12,5))
    ax = fig.subplots()

    ax.scatter(all_ord_list, po_curve, color='g', label="Positive Only")
    ax.scatter(all_ord_list, intact_curve, color='b', label="Intact")
    ax.set_xlabel("Image")
    ax.set_ylabel("Activation")
    ax.legend(loc="lower left")
    ax.set_title(f"Intact and Pos-Only Tuning Curve\nTrained ResNet18 Layer {layer} Neuron {neuron}")
    plt.savefig(f'/home/andrelongon/Documents/inhibition_code/plots/positive-only_curves/intact_and_po_curve_trained_resnet18_{layer}_unit{neuron}.png', dpi=256)
    plt.close()


def tuning_cross_plot(layer, first_neuron, second_neuron):
    first_curve = np.load(
        f"/home/andrelongon/Documents/inhibition_code/tuning_curves/alexnet/{layer}/inh_abl/{layer}_unit{first_neuron}_inh_abl_unrolled_act.npy"
    )
    # first_curve = -1 * first_curve
    first_curve = normalize(first_curve)

    second_curve = np.load(
        f"/home/andrelongon/Documents/inhibition_code/tuning_curves/alexnet/{layer}/inh_abl/{layer}_unit{second_neuron}_inh_abl_unrolled_act.npy"
    )
    # second_curve = -1 * second_curve
    second_curve = normalize(second_curve)

    r, p_val = pearsonr(first_curve, second_curve)
    print(f"Correlation results: r={r}, p={p_val}")

    fig = plt.figure(figsize=(8,5))
    ax = fig.subplots()

    ax.scatter(first_curve, second_curve, color='g')
    ax.set_xlabel(f"Neuron {first_neuron} Activation")
    ax.set_ylabel(f"Neuron {second_neuron} Activation")
    ax.set_title(f"Pos-Only Activations\nTrained AlexNet Layer {layer} Neurons {first_neuron} and {second_neuron}")
    plt.savefig(f'/home/andrelongon/Documents/inhibition_code/plots/curve_cross_plot/po_curves_trained_alexnet_{layer}_units{first_neuron}x{second_neuron}.png')
    plt.close()


def norm_curve_plot(layer, neuron):
    po_curve = np.load(
        f"/home/andrelongon/Documents/inhibition_code/tuning_curves/alexnet/{layer}/inh_abl/{layer}_unit{neuron}_inh_abl_all_act_list.npy"
    )
    
    no_curve = np.load(
        f"/home/andrelongon/Documents/inhibition_code/tuning_curves/alexnet/{layer}/exc_abl/{layer}_unit{neuron}_exc_abl_unrolled_act.npy"
    )

    norm_curve = po_curve + (-1 * no_curve)

    all_ord_list = np.arange(po_curve.shape[0]).tolist()
    norm_curve, all_ord_sorted = zip(*sorted(zip(norm_curve, all_ord_list), reverse=True))

    plt.scatter(all_ord_list, po_curve, color='b')
    plt.xlabel("Image")
    plt.ylabel("Activation")
    plt.title(f"Norm Tuning Curve - Trained AlexNet Layer {layer} Neuron {neuron}")
    plt.savefig(f'/home/andrelongon/Documents/inhibition_code/plots/norm_curves/norm_curve_trained_alexnet_{layer}_unit{neuron}.png')
    plt.close()


def input_proto_sim_plot(min_max_d, rand_d, layer):
    f_stat, p_val = f_oneway(np.array(min_max_d), np.array(rand_d))
    print(f"ANOVA results: F Score={f_stat}, p={p_val}")

    fig = plt.figure(figsize=(8,5))
    ax = fig.subplots()

    ax.scatter(np.arange(len(min_max_d)), np.array(min_max_d), color='b', label="Min Max")
    ax.scatter(np.arange(len(rand_d)), np.array(rand_d), color='y', label="Random")
    ax.set_xlabel("Neuron")
    ax.set_ylabel("LPIPS (AlexNet Backbone)")
    ax.legend(loc="lower right")
    ax.set_title(f"Max Min Channel Weight Prototype Similarity\nTrained ResNet18 Layer {layer}")

    plt.savefig(f'/home/andrelongon/Documents/inhibition_code/plots/max_channel_proto_sim/trained_resnet18_{layer}.png')
    plt.close()


def proto_anti_sim_plot(same_d, rand_d, layer):
    f_stat, p_val = f_oneway(np.array(same_d), np.array(rand_d))
    print(f"ANOVA results: F Score={f_stat}, p={p_val}")

    fig = plt.figure(figsize=(8,5))
    ax = fig.subplots()

    ax.scatter(np.arange(len(same_d)), np.array(same_d), color='b', label="Same")
    ax.scatter(np.arange(len(rand_d)), np.array(rand_d), color='y', label="Random")
    ax.set_xlabel("Neuron")
    ax.set_ylabel("LPIPS (AlexNet Backbone)")
    ax.legend(loc="lower right")
    ax.set_title(f"Ablated Prototype Antiproto Similarity\nTrained AlexNet Layer {layer}")

    plt.savefig(f'/home/andrelongon/Documents/inhibition_code/plots/proto_antiproto_sim/ablated_trained_alexnet_{layer}.png', dpi=128)
    plt.close()


def top_bot_layer_ranks(avg_top_ranks, avg_top_error, avg_bot_ranks, avg_bot_error, layer_name):
    fig = plt.figure(figsize=(8,5))
    ax = fig.subplots()

    ax.errorbar(np.arange(avg_top_ranks.shape[0]), avg_top_ranks, avg_top_error, fmt='go', label="Top Image")
    ax.errorbar(np.arange(avg_bot_ranks.shape[0]), avg_bot_ranks, avg_bot_error, fmt='ro', label="Bot Image")
    ax.set_xlabel("Neuron")
    ax.set_ylabel("Rank")
    ax.invert_yaxis()
    ax.legend(loc="upper right")
    ax.set_title(f"Trained AlexNet Layer {layer_name} Ranks")
    plt.savefig(f'/home/andrelongon/Documents/inhibition_code/plots/layer_ranks/trained_alexnet_{layer_name}.png')
    plt.close()