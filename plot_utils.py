import numpy as np
import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models
from scipy.stats import spearmanr, f_oneway


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
    f.suptitle("Intact vs. Negative-Weight Ablated Rank\nUntrained ResNet18 Layer 4.1 Conv2, First 24 Units")
    for i in range(4):
        axarr[i].set_xticks([0, 1], ["Intact", "Ablated"])
        axarr[i].set_ylabel("Rank")
        axarr[i].set_ylim(bottom=-1000, top=51000)
        axarr[i].invert_yaxis()

    num_units = 24
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
    print(f_oneway(np.array(abl_proto_ranks), np.array(abl_top9_ranks)))

    print(f"Mean antiproto:  {np.mean(np.array(abl_antiproto_ranks))},  Mean bot9:  {np.mean(np.array(abl_bot9_ranks))}")
    print(f"Antiproto - Bot 9 Rank Diff:  {np.mean(np.array(abl_antiproto_ranks)) - np.mean(np.array(abl_bot9_ranks))}")
    print(f_oneway(np.array(abl_antiproto_ranks), np.array(abl_bot9_ranks)))

    plt.savefig("/home/andre/rank_data/imnet_val_rank_intact_v_inh_abl_untrained_resnet18_layer4.1.Conv2dconv2_100.png", dpi=256)
    plt.close()


def plot_weight_histos():
    bins = [-0.1  , -0.075, -0.05 , -0.025,  0.   ,  0.025,  0.05 ,  0.075, 0.1]
    model = models.resnet18(False)
    state_keys = [
        'layer1.1.conv2.weight', 'layer2.1.conv2.weight', 'layer3.1.conv2.weight', 'layer4.1.conv2.weight'
    ]

    model.load_state_dict(torch.load("/home/andre/model_weights/resnet-18-l2-eps3.pt"))

    f, axarr = plt.subplots(1, 4, layout='tight', figsize=(8, 5))
    f.suptitle("Weight Histogram Across Layers\nRobust ResNet18 Conv2 for 1.1, 2.1, 3.1, 4.1")
    # max = 0

    for i in range(len(state_keys)):
        weights = torch.flatten(model.state_dict()[state_keys[i]])
        hist = axarr[i].hist(weights, bins=bins)[0]

    #     if np.max(hist) > max:
    #         max = np.max(hist)
    #
    # for i in range(4):
    #     axarr[i].set_ylim(bottom=0, top=max)

    plt.savefig("/home/andre/weight_histos/robust_resnet_x.1.conv2.png")


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