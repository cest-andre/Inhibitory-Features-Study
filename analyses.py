import sys
import os
import copy

from thingsvision import get_extractor
from scipy.stats import spearmanr, pearsonr
from scipy.spatial.distance import cosine
import numpy as np
import torch
import matplotlib.pyplot as plt
from circuit_toolkit.selectivity_utils import saveTopN

from imnet_val import validate_tuning_curve_thingsvision
from modify_weights import clamp_ablate_unit, invert_weights
from plot_utils import curve_corr_plot, normalize, top_bot_layer_ranks


#   TODO - how to measure similarity?  Cosine between the two 50k tuning vectors?
#          Spearman corr of ascending order of intact tuning curve (order inverted curve according to this)?
#          Compute both for now.  What about Pearson?  Is the monotonicity of Spearman preferred?
#          Perform on ablated units (for invert, ablate positives then invert).
#          If inverted is closer than a random unit in the trained case, will this hold for untrained?
def tuning_curve_similarity(unit, model_name="resnet18_robust", layer_name="layer4.1.Conv2dconv2", curve_dir="/home/andrelongon/Documents/inhibition_code/tuning_curves"):
    exc_curve = np.load(os.path.join(curve_dir, f"{model_name}/{layer_name}/intact", f"{layer_name}_unit{unit}_intact_unrolled_act.npy"))
    exc_order = np.load(os.path.join(curve_dir, f"{model_name}/{layer_name}/intact", f"{layer_name}_unit{unit}_intact_all_ord_sorted.npy"))
    # exc_order = np.flip(exc_order)
    # exc_curve = normalize(exc_curve[exc_order])
    exc_curve = normalize(exc_curve)
    # exc_curve = exc_curve[exc_order[:1000]]

    #   Get exc_abl curve of same unit, multiply by -1 to obtain inverted curve.
    #   Order inverted curve to match.  Calculate cosine and correlation.
    inverted_curve = -1 * np.load(os.path.join(curve_dir, f"{model_name}/{layer_name}/exc_abl", f"{layer_name}_unit{unit}_exc_abl_unrolled_act.npy"))
    # inverted_curve = np.load(os.path.join(curve_dir, f"{model_name}/{layer_name}/inh_abl", f"{layer_name}_unit{unit}_inh_abl_unrolled_act.npy"))
    # inverted_order = np.load(os.path.join(curve_dir, f"{model_name}/{layer_name}/exc_abl", f"{layer_name}_unit{unit}_exc_abl_all_ord_sorted.npy"))
    # inverted_order = np.flip(inverted_order)
    inverted_curve = normalize(inverted_curve)
    # inverted_curve = inverted_curve[exc_order[:1000]]

    # inverted_cos = cosine(exc_curve, inverted_curve[exc_order])
    # print(f"Cosine sim of same unit invert:  {inverted_cos}")

    # inverted_corr = pearsonr(exc_curve, inverted_curve[exc_order])[0]
    inverted_corr = pearsonr(exc_curve, inverted_curve)[0]
    # print(f"Corr of same unit invert:  {inverted_corr}")

    # inverted_curve = inverted_curve[inverted_order]

    all_exc_cos = []
    all_inh_cos = []
    all_exc_corr = []
    all_inh_corr = []
    #   Loop through all neurons i =/= unit and repeat wrt exc curve.
    for i in range(64):
        if i == unit:
            continue

        alt_exc_curve = np.load(os.path.join(curve_dir, f"{model_name}/{layer_name}/intact", f"{layer_name}_unit{i}_intact_unrolled_act.npy"))
        alt_exc_curve = normalize(alt_exc_curve)
        # all_exc_cos.append(cosine(exc_curve, alt_exc_curve[exc_order]))

        # all_exc_corr.append(pearsonr(exc_curve, alt_exc_curve[exc_order])[0])
        # all_exc_corr.append(pearsonr(exc_curve, alt_exc_curve)[0])
        # all_exc_corr.append(pearsonr(inverted_curve, alt_exc_curve[inverted_order])[0])


        # alt_inverted_curve = -1 * np.load(os.path.join(curve_dir, f"{model_name}/{layer_name}/exc_abl", f"{layer_name}_unit{i}_exc_abl_unrolled_act.npy"))
        # alt_inverted_curve = normalize(alt_inverted_curve)
        # all_inh_cos.append(cosine(exc_curve, alt_inverted_curve[exc_order]))

        # # all_inh_corr.append(pearsonr(exc_curve, alt_inverted_curve[exc_order])[0])
        # all_inh_corr.append(pearsonr(inverted_curve, alt_inverted_curve[inverted_order])[0])

    # print(f"Cosine sim average over all exc units:  {np.mean(np.array(all_exc_cos))}")
    # print(f"Corr average over all exc units:  {np.mean(np.array(all_exc_corr))}")

    # print(f"Cosine sim average over all inh units:  {np.mean(np.array(all_inh_cos))}")
    # print(f"Corr average over all inh units:  {np.mean(np.array(all_inh_corr))}")

    return inverted_corr, np.array(all_exc_corr), np.array(all_inh_corr)


def ablated_curve_broadness(unit, layer_name, frac_greater=0.25, model_name="alexnet", curve_dir="/home/andrelongon/Documents/inhibition_code/tuning_curves"):
    pos_curve = np.load(os.path.join(curve_dir, f"{model_name}/{layer_name}/inh_abl", f"{layer_name}_unit{unit}_inh_abl_all_act_list.npy"))
    neg_curve = np.load(os.path.join(curve_dir, f"{model_name}/{layer_name}/exc_abl", f"{layer_name}_unit{unit}_exc_abl_all_act_list.npy"))

    pos_frac = np.where(pos_curve > (pos_curve[0] * frac_greater))[0].shape[0] / pos_curve.shape[0]
    neg_frac = np.where(neg_curve < (neg_curve[-1] * frac_greater))[0].shape[0] / neg_curve.shape[0]

    # pos_max = np.mean(pos_curve[:100])
    # neg_max = np.mean(neg_curve[-100:])
    # pos_frac = []
    # neg_frac = []

    # # for i in range(1, 9, 1):
    # for i in range(9999, 50000, 10000):
    #     pos_frac.append(pos_curve[i] / pos_max)
    #     neg_frac.append(neg_curve[-i] / neg_max)

    #     # pos_frac.append(np.nonzero(pos_curve > ((i / 10) * pos_max))[0].shape[0] / pos_curve.shape[0])
    #     # neg_frac.append(np.nonzero(neg_curve < ((i / 10) * neg_max))[0].shape[0] / neg_curve.shape[0])

    return pos_frac, neg_frac


def intact_curve_broadness(unit, layer_name, frac_greater=0.25, model_name="resnet18_robust", curve_dir="/home/andrelongon/Documents/inhibition_code/tuning_curves"):
    curve = np.load(os.path.join(curve_dir, f"{model_name}/{layer_name}/intact", f"{layer_name}_unit{unit}_intact_all_act_list.npy"))

    pos_frac = np.where(curve > (curve[0] * frac_greater))[0].shape[0] / curve.shape[0]
    neg_frac = np.where(curve < (curve[-1] * frac_greater))[0].shape[0] / curve.shape[0]

    return pos_frac, neg_frac


#   Larger error implies image is top rank in some and low in others, so sparser.  Curtain kid is not this case.  Higher avg imgs
#   will have lower error is my prediction.
def average_rank(layer_name, num_neurons, model_name="alexnet", curve_dir="/home/andrelongon/Documents/inhibition_code/tuning_curves"):
    avg_ranks = []

    for n in range(num_neurons):
        order = np.load(os.path.join(curve_dir, f"{model_name}/{layer_name}/intact", f"{layer_name}_unit{n}_intact_all_ord_sorted.npy"))
        ranks = [np.nonzero(order == i)[0][0] for i in range(order.shape[0])]

        avg_ranks.append(ranks)

    avg_ranks = np.array(avg_ranks)
    errors = np.std(avg_ranks, axis=0) / np.sqrt(avg_ranks.shape[0])

    avg_ranks, _ = zip(*sorted(zip(np.mean(avg_ranks, axis=0).tolist(), np.arange(avg_ranks.shape[1]).tolist()), reverse=False))
    avg_ranks = np.array(avg_ranks)

    fig = plt.figure(figsize=(8,5))
    ax = fig.subplots()

    ax.errorbar(
        np.arange(avg_ranks.shape[0]), avg_ranks, errors, fmt='bo'
    )
    ax.set_xlabel("Image")
    ax.set_ylabel("Average Rank")
    ax.set_title(f"Imnet Val Average Rank - Trained AlexNet Layer {layer_name}")
    plt.savefig(f'/home/andrelongon/Documents/inhibition_code/plots/avg_rank/trained_alexnet_{layer_name}.png')
    plt.close()


def image_rank(layer_name, ref_neuron, ref_rank, num_neurons, model_name="alexnet", curve_dir="/home/andrelongon/Documents/inhibition_code/tuning_curves"):
    ref_order = np.load(os.path.join(curve_dir, f"{model_name}/{layer_name}/intact", f"{layer_name}_unit{ref_neuron}_intact_all_ord_sorted.npy"))
    #   Get the img idx of most inh img of ref_neuron (for alexnet conv10 neuron 0: parrot).
    target_idx = ref_order[ref_rank]
    ranks = []
    for n in range(num_neurons):
        if n == ref_neuron:
            continue

        order = np.load(os.path.join(curve_dir, f"{model_name}/{layer_name}/intact", f"{layer_name}_unit{n}_intact_all_ord_sorted.npy"))
        rank = np.nonzero(order == target_idx)[0][0]
        ranks.append(rank)

    # fig = plt.figure(figsize=(8,5))
    # ax = fig.subplots()

    ranks = np.array(ranks)

    return np.mean(ranks), np.std(ranks, axis=0) / np.sqrt(ranks.shape[0])

    # ax.scatter(np.arange(ranks.shape[0]), ranks, color='b')
    # ax.set_xlabel("Neuron")
    # ax.set_ylabel("Rank")
    # ax.invert_yaxis()
    # ax.set_title(f"Trained AlexNet Layer {layer_name} Ranks\nNeuron {ref_neuron} Rank {ref_rank+1}")
    # plt.savefig(f'/home/andrelongon/Documents/inhibition_code/plots/layer_ranks/individual/trained_alexnet_{layer_name}_neuron{ref_neuron}_rank{ref_rank+1}.png')
    # plt.close()



#   TODO:   Create a thingsvision extractor and pass to imnet val thingsvision function to get all
#           post-relu activations (first_layer) along with images.  Now, we can extract subset of acts
#           according to second_layer's center neuron and kernel size (use input_tools code).
#           With all this info, we can rank images by sparsity (proporition of zeros in input) and input mag.
#
#           *Next, we wish to see if increased mag (less sparsity) is correlated with decreased activation
#           in second_layer (pre-relu to get negative acts).  Can also save top 9 mags and see how often they
#           appear in neurons' intact bot 9 in second_layer.
#
#           If this is all the case, the next thing would be to check which features in curtain kid images
#           so broadly activate the first_layer.  Perhaps circuit analysis will reveal how curtain kid produces
#           such broad, non-sparse activity.
#
#     ***   As curtain kid images produce large input magnitude, it is easier to include these images in both bot
#           AND top activating responses by the downstream neuron.  Perhaps this is what guides the top selectivity
#           of neurons: thus, top images of neurons will be tuned for a small subset of curtain kids which will necessarily
#           have high exc and inh info.  So during optimization, it is constructing the neuron's preferred curtain kid
#           that it decided to select for rather than suppress.
#
#           What is the avg rank in second_layer for images with highest mag/lowest sparsity?  If low, suggests there is a
#           population wide effort to suppress.  Could also plot avg rank of just these images for each neuron.
#
#   **      Results from input mag are interesting.  There is a small but significant neg corr between conv10 neuron act and
#           conv8 post-relu magnitude.  Corr varies strongly across neurons, ranging from -0.7 to 0.05.
#           This kinda makes sense in this layer as a majority of weights are negative, thus the higher mags are more likely
#           to encounter a negative weight.  Maybe sparsity provides additional info about this relationship.
#
#    ***    Next, decompose mag into pos and neg weighted inputs.  Rather than just summing these, take the average to control
#           for the imbalance of pos and neg weights.  If there is higher neg-positioned input mag mean than pos-positioned for
#           these large mag images, suggests that neurons learn to selectivity inhibit features in these images.  These features
#           may in turn wind up in top activating images, hence why top 9 have high inhibition.  Maybe this is where entangling lies.
def imnet_layer_outputs(model_name, first_layer, second_layer, weight_name, statesdir=None, curve_dir="/home/andrelongon/Documents/inhibition_code/tuning_curves"):
    extractor = get_extractor(
        model_name=model_name,
        source='torchvision',
        device='cuda',
        pretrained=True
    )

    if statesdir is not None:
        extractor.model.load_state_dict(torch.load(statesdir))

    weights = extractor.model.state_dict()[weight_name]
    num_neurons = weights.shape[0]
    idx_offset = weights.shape[-1] // 2
    # weights = torch.flatten(weights)

    all_images, _, unrolled_act, _, _, _ = validate_tuning_curve_thingsvision(extractor, first_layer, sort_acts=False)
    unrolled_act = np.clip(np.array(unrolled_act), 0, None)

    mid_idx = unrolled_act.shape[-1] // 2
    # print(f"IDX: {mid_idx-idx_offset}:{mid_idx+idx_offset+1}")
    unrolled_act = unrolled_act[:, :, mid_idx-idx_offset:mid_idx+idx_offset+1, mid_idx-idx_offset:mid_idx+idx_offset+1]

    #   For each image, obtain its non-zero percentage (sparsity) and magnitude summation.  Plot a scatter of these and get r and p_val.
    # sparsities = []
    # mags = []
    # for act in unrolled_act:
    #     flattened_act = np.ndarray.flatten(act)
    #     sparsity = np.nonzero(flattened_act <= 0)[0].shape[0] / flattened_act.shape[0]
    #     mag = np.sum(flattened_act)

    #     sparsities.append(sparsity)
    #     mags.append(mag)

    # sparsities = np.array(sparsities)
    # mags = np.array(mags)
    # r, p_val = pearsonr(sparsities, mags)
    # print(f"Correlation results: r={r}, p={p_val}")
    
    # fig = plt.figure(figsize=(8,5))
    # ax = fig.subplots()

    # ax.scatter(mags, sparsities, color='b')
    # ax.set_xlabel(f"Magnitude")
    # ax.set_ylabel(f"Sparsity (Fraction of Zeros)")
    # ax.set_title(f"\nTrained {model_name} Layer {first_layer}")
    # plt.savefig(f'/home/andrelongon/Documents/inhibition_code/plots/mag_sparsity_corr/trained_{model_name}_{first_layer}.png')
    # plt.close()

    input_mag = np.sum(unrolled_act, axis=(1, 2, 3))
    mag_mean = np.mean(unrolled_act, axis=(1, 2, 3))
    ordered_mag_sum, all_ord_sorted = zip(*sorted(zip(input_mag.tolist(), np.arange(input_mag.shape[0]).tolist()), reverse=True))
    ordered_mag_mean, all_ord_sorted = zip(*sorted(zip(mag_mean.tolist(), np.arange(mag_mean.shape[0]).tolist()), reverse=True))
    

    # rs = []
    # all_ranks = []
    # all_errors = []
    # for n in range(num_neurons):
    #     acts = np.load(os.path.join(curve_dir, f"{model_name}/{second_layer}/intact", f"{second_layer}_unit{n}_intact_unrolled_act.npy"))
    #     r, p_val = pearsonr(normalize(input_mag), normalize(acts))
    #     rs.append(r)

    #     ranks = []
    #     for i in range(10):
    #         rank = np.nonzero(acts > acts[all_ord_sorted[i]])[0].shape[0]
    #         ranks.append(rank)

    #     ranks = np.array(ranks)
    #     all_ranks.append(np.mean(ranks))
    #     all_errors.append(np.std(ranks, axis=0) / np.sqrt(ranks.shape[0]))

        # if n == 1:
        #     fig = plt.figure(figsize=(8,5))
        #     ax = fig.subplots()
        #     ax.scatter(input_mag, acts, color='b')
        #     ax.set_xlabel(f"Input Magnitude")
        #     ax.set_ylabel(f"Neuron {n} Activation")
        #     ax.set_title(f"Input Magnitude and Activation\nTrained AlexNet Layer {second_layer} Neuron {n}")
        #     plt.savefig(f'/home/andrelongon/Documents/inhibition_code/plots/input_mag_acts_corr/trained_alexnet_{second_layer}_unit{n}.png')
        #     plt.close()
        #     exit()


        # print(f"Correlation results (normalized): r={r}, p={p_val}")

    # fig = plt.figure(figsize=(8,5))
    # ax = fig.subplots()

    # ax.scatter(input_mag, acts, color='b')
    # ax.set_xlabel(f"Input Magnitude")
    # ax.set_ylabel(f"Neuron {0} Activation")
    # ax.set_title(f"Input Magnitude and Activation\nTrained AlexNet Layer {second_layer} Neuron {0}")
    # plt.savefig(f'/home/andrelongon/Documents/inhibition_code/plots/input_mag_acts_corr/features.8.test/trained_alexnet_{second_layer}_unit{0}.png')
    # plt.close()

    # saveTopN(all_images, all_ord_sorted, f"{model_name}_{first_layer}", path="/home/andrelongon/Documents/inhibition_code/input_mags")

    fig = plt.figure(figsize=(8,5))
    ax = fig.subplots()

    # ax.scatter(np.arange(len(rs)), np.array(rs), color='b')
    # ax.set_xlabel(f"Neuron")
    # ax.set_ylabel(f"(Input Magnitude, Activation) Correlation")
    # ax.set_title(f"Input Mag Act Correlation for All Neurons\nTrained {model_name} Layer {second_layer}")
    # plt.savefig(f'/home/andrelongon/Documents/inhibition_code/plots/input_mag_acts_corr/trained_{model_name}_{second_layer}_all_neurons.png')
    # plt.close()

    ax.scatter(np.arange(len(ordered_mag_sum)), np.array(ordered_mag_sum), color='b')
    ax.set_xlabel(f"Image")
    ax.set_ylabel(f"Magnitude")
    ax.set_title(f"Magnitude for ImageNet Val\nTrained {model_name} Layer {first_layer}")
    plt.savefig(f'/home/andrelongon/Documents/inhibition_code/plots/input_mag_curves/trained_{model_name}_{first_layer}.png')
    plt.close()

    fig = plt.figure(figsize=(8,5))
    ax = fig.subplots()

    ax.scatter(np.arange(len(ordered_mag_mean)), np.array(ordered_mag_mean), color='b')
    ax.set_xlabel(f"Image")
    ax.set_ylabel(f"Avg Magnitude")
    ax.set_title(f"Avg Mag for ImageNet Val\nTrained {model_name} Layer {first_layer}")
    plt.savefig(f'/home/andrelongon/Documents/inhibition_code/plots/input_mag_curves/avg_trained_{model_name}_{first_layer}.png')
    plt.close()

    # fig = plt.figure(figsize=(8,5))
    # ax = fig.subplots()

    # ax.errorbar(np.arange(np.array(all_ranks).shape[0]), np.array(all_ranks), np.array(all_errors), fmt='bo')
    # ax.set_xlabel(f"Neuron")
    # ax.set_ylabel(f"Average Rank")
    # ax.invert_yaxis()
    # ax.set_title(f"Top 9 Input Mag Avg Rank\nTrained {model_name} Layer {second_layer}")
    # plt.savefig(f'/home/andrelongon/Documents/inhibition_code/plots/input_mag_ranks/trained_{model_name}_{second_layer}.png')
    # plt.close()


#   TODO:  one would expect negative weights to overlap more by virtue of there being more negative weights in all the neurons.
#          might need to compensate by random sampling the same number.  Take 10% of negative and pos base weights and see how
#          much they overlap in other neurons.
#          *Maybe I also control by randomly samping weights from base neuron and seeing how big the overlap is in that case.
#          Method:  get proportion of neg/pos weights in baseline.  Get pos and neg weight indices.  Then randomly sample
#          neg weight % and pos weight %.  Perform 4 index comparisons with other neurons:  pos, neg, rand pos%, rand neg%.
#          
#          Maybe that control is unnecessary.  If we know the neg/pos proportion in target neuron, we just want to see if
#          overlap with neg and pos is higher.
#
#          Maybe randomly shuffle target weights instead?
def weight_overlap(model_name, weight_name, base_neuron, statesdir=None):
    extractor = get_extractor(
        model_name=model_name,
        source='torchvision',
        device='cpu',
        pretrained=True
    )

    if statesdir is not None:
        extractor.model.load_state_dict(torch.load(statesdir))

    weights = extractor.model.state_dict()[weight_name]

    base_weights = torch.flatten(weights[base_neuron])
    pos_idx = torch.nonzero(base_weights > 0)
    neg_idx = torch.nonzero(base_weights < 0)

    pos_laps = []
    neg_laps = []

    for n in range(weights.shape[0]):
        if n == base_neuron:
            continue

        target_weights = torch.flatten(weights[n])
        pos_overlap = torch.nonzero(target_weights[pos_idx] > 0).shape[0] / pos_idx.shape[0]
        neg_overlap = torch.nonzero(target_weights[neg_idx] < 0).shape[0] / neg_idx.shape[0]

        # print(f"Positive overlap neuron {n}: {pos_overlap}")
        # print(f"Negative overlap neuron {n}: {neg_overlap}\n\n")

        pos_laps.append(pos_overlap)
        neg_laps.append(neg_overlap)

    print(f"Avg pos overlap: {torch.mean(torch.Tensor(pos_laps))}")
    print(f"Avg neg overlap: {torch.mean(torch.Tensor(neg_laps))}")




if __name__ == "__main__":
    extractor = get_extractor(
        model_name='alexnet',
        source='torchvision',
        device='cuda',
        pretrained=True
    )

    pos_fracs = []
    neg_fracs = []
    for i in range(512):
        pos_frac, neg_frac = intact_curve_broadness(i, "layer4.1.Conv2dconv2")

        pos_fracs.append(pos_frac)
        neg_fracs.append(neg_frac)

    print(np.mean(np.array(pos_fracs)))
    print(np.mean(np.array(neg_fracs)))

    exit()

    # inverted_corrs = []
    # all_exc_mean = []
    # all_inh_mean = []
    # all_exc_err = []
    # all_inh_err = []
    # for i in range(64):
    #     inverted_corr, all_exc_corrs, all_inh_corrs = tuning_curve_similarity(i)

    #     inverted_corrs.append(inverted_corr)

    #     all_exc_mean.append(np.mean(all_exc_corrs))
    #     all_inh_mean.append(np.mean(all_inh_corrs))
    #     all_exc_err.append(np.std(all_exc_corrs) / np.sqrt(all_exc_corrs.shape[0]))
    #     all_inh_err.append(np.std(all_exc_corrs) / np.sqrt(all_exc_corrs.shape[0]))

    # curve_corr_plot(
    #     np.array(inverted_corrs), np.array(all_exc_mean), np.array(all_inh_mean),
    #     np.array(all_exc_err), np.array(all_inh_err)
    # )

    # exit()

    # average_rank("features.Conv2d3", 192)
    # average_rank("features.Conv2d6", 384)
    # average_rank("features.Conv2d10", 256)

    # avg_bot_ranks = []
    # avg_top_ranks = []
    # avg_bot_error = []
    # avg_top_error = []
    # for i in range(384):
    #     avg_rank, error = image_rank("features.Conv2d6", i, 0, 384, model_name="alexnet")
    #     avg_top_ranks.append(avg_rank)
    #     avg_top_error.append(error)

    #     avg_rank, avg_err = image_rank("features.Conv2d6", i, 49999, 384, model_name="alexnet")
    #     avg_bot_ranks.append(avg_rank)
    #     avg_bot_error.append(error)

    # top_bot_layer_ranks(np.array(avg_top_ranks), np.array(avg_top_error), np.array(avg_bot_ranks), np.array(avg_bot_error), "features.Conv2d6")

    # imnet_layer_outputs("alexnet", "features.2", "features.Conv2d3", "features.3.weight")
    # imnet_layer_outputs("alexnet", "features.5", "features.Conv2d6", "features.6.weight")
    # imnet_layer_outputs("alexnet", "features.8", "features.Conv2d10", "features.10.weight")

    # imnet_layer_outputs("resnet18", "layer2.1.conv1", "layer2.1.Conv2dconv2", "layer2.1.conv2.weight")
    # imnet_layer_outputs("resnet18", "layer4.1.conv1", "layer4.1.Conv2dconv2", "layer4.1.conv2.weight")
    # imnet_layer_outputs("resnet18", "layer4.1.bn1", "layer4.1.Conv2dconv2", "layer4.1.conv2.weight")

    # for i in range(16):
    #     weight_overlap("alexnet", "features.10.weight", i)