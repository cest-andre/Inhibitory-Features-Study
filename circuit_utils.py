import torch
from torchvision import transforms
from thingsvision import get_extractor, get_extractor_from_model
from PIL import Image
import numpy as np
from scipy.stats import pearsonr
import plotly.express as px

from cornet_s import CORnet_S
from transplant import get_activations


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
norm_trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])


def measure_specialization(extractor, weight_name, input_module=None):
    weights = extractor.model.state_dict()[weight_name]
    idx_offset = weights.shape[-1] // 2

    weights = torch.flatten(weights, start_dim=2)
    abs_weights = torch.abs(weights)
    # print(torch.std(torch.sum(abs_weights, -1)))

    #   TODO:  rather than std and any over w,h (idv weights), use weight magnitudes (sum across h,w)
    thresh_count = torch.zeros(abs_weights.shape[1])
    for n in abs_weights:
        # thresh_count += torch.any((n > 2*torch.std(abs_weights)), -1)

        thresh_count += torch.sum(n, -1) > 2*torch.std(torch.sum(abs_weights, -1))

    genericity = thresh_count / abs_weights.shape[0]

    #   NOTE:  run lucent opt image and obtain input acts, then see which inputs
    #          contributed most to exc and inh.  Then see if these inputs are
    #          special or generic.

    mean_pos_ubiq = 0
    mean_neg_ubiq = 0

    mean_pos_viz_pos_ubiq = 0
    mean_pos_viz_neg_ubiq = 0
    # mean_neg_viz_pos_ubiq = 0
    # mean_neg_viz_neg_ubiq = 0
    for i in range(weights.shape[0]):
        n = weights[i]
        max_weights, _ = torch.max(n, -1)
        min_weights, _ = torch.max(-n, -1)

        _, pos_idx = torch.topk(max_weights, 9)
        _, neg_idx = torch.topk(min_weights, 9)

        mean_pos_ubiq += torch.mean(genericity[pos_idx])
        mean_neg_ubiq += torch.mean(genericity[neg_idx])

        pos_viz = Image.open(f"/media/andrelongon/DATA/feature_viz/intact/googlenet/{weight_name[:-7]}/unit{i}/pos/0.png")
        pos_viz = norm_trans(pos_viz)

        acts = get_activations(extractor, pos_viz, input_module)
        mid_idx = torch.Tensor(acts).shape[-1] // 2

        acts = torch.Tensor(acts)[0, :, mid_idx-idx_offset:mid_idx+idx_offset+1, mid_idx-idx_offset:mid_idx+idx_offset+1]
        acts = torch.clamp(acts, min=0, max=None)
        acts = torch.flatten(acts, start_dim=1)

        weighted_acts = torch.sum(weights[i] * acts, -1)

        _, pos_idx = torch.topk(weighted_acts, 9)
        _, neg_idx = torch.topk(weighted_acts, 9, largest=False)

        mean_pos_viz_pos_ubiq += torch.mean(genericity[pos_idx])
        mean_pos_viz_neg_ubiq += torch.mean(genericity[neg_idx])

        # neg_viz = Image.open(f"/media/andrelongon/DATA/feature_viz/intact/resnet18/{weight_name[:-7]}/unit{i}/neg/0.png")
        # neg_viz = norm_trans(neg_viz)

        # acts = get_activations(extractor, neg_viz, input_module)
        # mid_idx = torch.Tensor(acts).shape[-1] // 2

        # acts = torch.Tensor(acts)[0, :, mid_idx-idx_offset:mid_idx+idx_offset+1, mid_idx-idx_offset:mid_idx+idx_offset+1]
        # acts = torch.clamp(acts, min=0, max=None)
        # acts = torch.flatten(acts, start_dim=1)

        # weighted_acts = torch.sum(weights[i] * acts, -1)

        # _, pos_idx = torch.topk(weighted_acts, 9)
        # _, neg_idx = torch.topk(weighted_acts, 9, largest=False)

        # mean_neg_viz_pos_ubiq += torch.mean(genericity[pos_idx])
        # mean_neg_viz_neg_ubiq += torch.mean(genericity[neg_idx])

    print(f"\nLayer: {weight_name[:-7]}")
    print(weights.shape)

    print("Mean ubiquity")
    print(torch.mean(genericity))

    print("Mean top pos and neg ubiquity")
    print(mean_pos_ubiq / weights.shape[0])
    print(mean_neg_ubiq / weights.shape[0])

    print("Mean top pos and neg ubiquity for pos feature viz")
    print(mean_pos_viz_pos_ubiq / weights.shape[0])
    print(mean_pos_viz_neg_ubiq / weights.shape[0])

    # print("Mean pos and neg ubiquity for neg feature viz")
    # print(mean_neg_viz_pos_ubiq / abs_weights.shape[0])
    # print(mean_neg_viz_neg_ubiq / abs_weights.shape[0])


def viz_inputs(extractor, weights, model_name, target_module, neuron, input_module, pos_idx, neg_idx):
    # top_img = Image.open(f"/media/andrelongon/DATA/feature_viz/intact/{model_name}/{target_module}/unit{neuron}/pos/0_distill_center.png")
    top_img = Image.open(f"/media/andrelongon/DATA/tuning_curves/{model_name}/{target_module}/intact/{target_module}_neuron{neuron}/max/exc/0.png")
    top_img = norm_trans(top_img)

    idx_offset = weights.shape[-1] // 2
    weights = torch.flatten(weights, start_dim=-2)

    acts = get_activations(extractor, top_img, input_module)
    mid_idx = torch.Tensor(acts).shape[-1] // 2
    acts = torch.Tensor(acts)[0, :, mid_idx-idx_offset:mid_idx+idx_offset+1, mid_idx-idx_offset:mid_idx+idx_offset+1]
    acts = torch.flatten(acts, start_dim=-2)
    acts = torch.clamp(acts, min=0, max=None)
    acts = torch.squeeze(acts)

    pos_mags = torch.clamp(weights, min=0, max=None)
    pos_mags = torch.squeeze(pos_mags)
    pos_acts = acts * pos_mags

    neg_mags = -1 * torch.clamp(weights, min=None, max=0)
    neg_mags = torch.squeeze(neg_mags)
    neg_acts = acts * neg_mags

    if weights.shape[-1] > 1:
        acts = torch.sum(acts, -1)
        pos_acts = torch.sum(pos_acts, -1)
        neg_acts = torch.sum(neg_acts, -1)

    _, top_pos_idx = torch.topk(pos_acts, 9)
    _, top_neg_idx = torch.topk(neg_acts, 9)

    #   TODO:  Only print neg results if a top neg-weighted input is in top neg acts list. 
    #   if any neg_idx is in top_neg_idx
    intersect = np.intersect1d(neg_idx[:3].numpy(), top_neg_idx.numpy())
    if intersect.shape[0] > 0:
    # if torch.nonzero(acts[neg_idx]).shape[0] > 0 or torch.nonzero(pos_acts[top_pos_idx] < neg_acts[top_neg_idx[0]]).shape[0] > 0:
        # print(f"Top pos weight chans: {np.asarray(pos_idx.numpy())}\nTop neg weight chans: {np.asarray(neg_idx.numpy())}")
        # print(f"Top pos weight inputs: {np.asarray(acts[pos_idx].numpy())}\nTop neg weight inputs: {np.asarray(acts[neg_idx].numpy())}")
        # # print(f"Top neg weight inputs nonzero:  {torch.nonzero(acts[neg_idx]).shape[0]}")

        # print(f"\nTotal pos act: {torch.sum(pos_acts)}\nTotal neg act: {torch.sum(neg_acts)}")

        # print(f"Top pos acts: {pos_acts[top_pos_idx]}\nTop pos acts idx: {top_pos_idx}")
        # print(f"Top neg acts: {neg_acts[top_neg_idx]}\nTop neg acts idx: {top_neg_idx}")

        # print(intersect)

        # print(f"Top pos acts less than top neg act:  {torch.nonzero(pos_acts[top_pos_idx] < neg_acts[top_neg_idx[0]]).shape[0]}")

        return intersect


def top_weighted_inputs(weights):
    weights = torch.flatten(weights, start_dim=1)

    total_mags = torch.sum(torch.abs(weights), -1)
    pos_mags = torch.sum(torch.clamp(weights, min=0, max=None), -1)
    neg_mags = -1 * torch.sum(torch.clamp(weights, min=None, max=0), -1)

    _, mag_idx = torch.topk(total_mags, 9)
    _, pos_idx = torch.topk(pos_mags, 9)
    _, neg_idx = torch.topk(neg_mags, 9)

    # print(pos_mags[pos_idx])
    # print(neg_mags[neg_idx])

    # print(f"Top total mags: {np.asarray(mag_idx.numpy())}\nTop pos mags: {np.asarray(pos_idx.numpy())}\nTop neg mags: {np.asarray(neg_idx.numpy())}")

    return mag_idx, pos_idx, neg_idx


def top_weights(weights):
    all_tops = torch.zeros(weights.shape[1])
    for c in range(weights.shape[0]):
        # print(f"CHANNEL {c}")
        _, w_idx = torch.topk(torch.flatten(weights[c]), 9)
        # print(f"TOP WEIGHTS: {w_idx}")

        all_tops[w_idx] += 1

        # if torch.nonzero(w_idx == c).shape[0]:
        #     in_top9 += 1

    _, tops_idx = torch.topk(all_tops, 9)
    print(f"Top weighted input channels:  {tops_idx}")
    print(f"Number in out channel's top 9: {all_tops[tops_idx]}")


def measure_invariance(extractor, model_name, output_module, input_module, middle_module, neuron, variant="scale", imnet_val=False, show_results=False):
    top_img = None
    if imnet_val:
        top_img = Image.open(
            f"/media/andrelongon/DATA/tuning_curves/{model_name}/{input_module.split('.prerelu_out')[0]}/intact/{input_module.split('.prerelu_out')[0]}_neuron{neuron}/max/exc/0.png"
        )
        top_img = norm_trans(top_img)
    else:
        top_img = Image.open(f"/media/andrelongon/DATA/feature_viz/intact/{model_name}/{input_module}/unit{neuron}/pos/0_distill_channel.png")
        top_img = norm_trans(top_img)

    variant_trans = None
    if variant == "scale":
        variant_trans = transforms.Compose([
            transforms.CenterCrop(112),
            transforms.Resize(224)
        ])
    elif variant == "flip":
        variant_trans = transforms.RandomHorizontalFlip(p=1)

    mid_base_act = get_activations(extractor, top_img, middle_module, channel_id=neuron, use_center=True)
    mid_variant_act = get_activations(extractor, variant_trans(top_img), middle_module, channel_id=neuron, use_center=True)

    mid_act_delta = np.clip(mid_variant_act, 0, None)[0] - np.clip(mid_base_act, 0, None)[0]

    if imnet_val:
        top_img = Image.open(
            f"/media/andrelongon/DATA/tuning_curves/{model_name}/{middle_module}/intact/{middle_module}_neuron{neuron}/max/exc/0.png"
        )
        top_img = norm_trans(top_img)
    else:
        top_img = Image.open(f"/media/andrelongon/DATA/feature_viz/intact/{model_name}/{middle_module}/unit{neuron}/pos/0_distill_channel.png")
        top_img = norm_trans(top_img)

    if variant == "scale":
        #   TODO:  perhaps use padding_mode='edge' or 'reflect'.
        variant_trans = transforms.Compose([
            transforms.Resize(112),
            transforms.Pad(56, padding_mode='reflect')
        ])

    in_base_act = get_activations(extractor, top_img, input_module, channel_id=neuron, use_center=True)
    in_variant_act = get_activations(extractor, variant_trans(top_img), input_module, channel_id=neuron, use_center=True)

    in_act_delta = np.clip(in_variant_act, 0, None)[0] - np.clip(in_base_act, 0, None)[0]

    if show_results:
        print(f"\nMid base Act:  {mid_base_act[0]},  Mid {variant} act:  {mid_variant_act[0]}")
        print(f"\nIn base Act:  {in_base_act[0]},  In {variant} act:  {in_variant_act[0]}")

    return mid_act_delta, in_act_delta



def stream_inspect(extractor, model_name, output_module, input_module, middle_module, neuron, imnet_val=False, show_results=False):
    top_img = None
    if imnet_val:
        # top_img = [
        #     norm_trans(Image.open(f"/media/andrelongon/DATA/tuning_curves/{model_name}/{output_module}/intact/{output_module}_neuron{neuron}/max/exc/{i}.png"))
        #     for i in range(9)
        # ]
        # top_img = torch.stack(top_img)

        top_img = Image.open(
            f"/media/andrelongon/DATA/tuning_curves/{model_name}/{output_module.split('.prerelu_out')[0]}/intact/{output_module.split('.prerelu_out')[0]}_neuron{neuron}/max/exc/0.png"
        )
        top_img = norm_trans(top_img)
    else:
        top_img = Image.open(f"/media/andrelongon/DATA/feature_viz/intact/{model_name}/{output_module}/unit{neuron}/pos/0_center.png")
        top_img = norm_trans(top_img)

    #   NOTE:  retrieve center act from channel as this is how top imnet imgs are sorted.
    out_out_acts = get_activations(extractor, top_img, output_module, channel_id=neuron, use_center=True)
    out_in_acts = get_activations(extractor, top_img, input_module, channel_id=neuron, use_center=True)
    out_mid_acts = get_activations(extractor, top_img, middle_module, channel_id=neuron, use_center=True)

    if imnet_val:
        # top_img = [
        #     norm_trans(Image.open(f"/media/andrelongon/DATA/tuning_curves/{model_name}/{input_module}/intact/{input_module}_neuron{neuron}/max/exc/{i}.png"))
        #     for i in range(9)
        # ]
        # top_img = torch.stack(top_img)

        top_img = Image.open(
            f"/media/andrelongon/DATA/tuning_curves/{model_name}/{input_module.split('.prerelu_out')[0]}/intact/{input_module.split('.prerelu_out')[0]}_neuron{neuron}/max/exc/0.png"
        )
        top_img = norm_trans(top_img)
    else:
        top_img = Image.open(f"/media/andrelongon/DATA/feature_viz/intact/{model_name}/{input_module}/unit{neuron}/pos/0_center.png")
        top_img = norm_trans(top_img)

    in_out_acts = get_activations(extractor, top_img, output_module, channel_id=neuron, use_center=True)
    in_in_acts = get_activations(extractor, top_img, input_module, channel_id=neuron, use_center=True)
    in_mid_acts = get_activations(extractor, top_img, middle_module, channel_id=neuron, use_center=True)

    if imnet_val:
        # top_img = [
        #     norm_trans(Image.open(f"/media/andrelongon/DATA/tuning_curves/{model_name}/{middle_module}/intact/{middle_module}_neuron{neuron}/max/exc/{i}.png"))
        #     for i in range(9)
        # ]
        # top_img = torch.stack(top_img)

        top_img = Image.open(f"/media/andrelongon/DATA/tuning_curves/{model_name}/{middle_module}/intact/{middle_module}_neuron{neuron}/max/exc/0.png")
        top_img = norm_trans(top_img)
    else:
        top_img = Image.open(f"/media/andrelongon/DATA/feature_viz/intact/{model_name}/{middle_module}/unit{neuron}/pos/0_center.png")
        top_img = norm_trans(top_img)

    mid_out_acts = get_activations(extractor, top_img, output_module, channel_id=neuron, use_center=True)
    mid_in_acts = get_activations(extractor, top_img, input_module, channel_id=neuron, use_center=True)
    mid_mid_acts = get_activations(extractor, top_img, middle_module, channel_id=neuron, use_center=True)

    if show_results:
        print("---OUTPUT MODULE TOP IMGS---")
        print("OUTPUT ACT")
        print(f"Total act: {np.mean(out_out_acts)}, Pos act: {np.mean(np.clip(out_out_acts, 0, None))}, Neg act: {np.mean(np.clip(out_out_acts, None, 0))}")

        print("INPUT ACT")
        print(f"Total act: {np.mean(out_in_acts)}, Pos act: {np.mean(np.clip(out_in_acts, 0, None))}, Neg act: {np.mean(np.clip(out_in_acts, None, 0))}")

        print("BN ACT")
        print(f"Total act: {np.mean(out_mid_acts)}, Pos act: {np.mean(np.clip(out_mid_acts, 0, None))}, Neg act: {np.mean(np.clip(out_mid_acts, None, 0))}")
        print("-------------------------------")

        print("---INPUT MODULE TOP IMGS---")
        print("OUTPUT ACT")
        print(f"Total act: {np.mean(in_out_acts)}, Pos act: {np.mean(np.clip(in_out_acts, 0, None))}, Neg act: {np.mean(np.clip(in_out_acts, None, 0))}")

        print("INPUT ACT")
        print(f"Total act: {np.mean(in_in_acts)}, Pos act: {np.mean(np.clip(in_in_acts, 0, None))}, Neg act: {np.mean(np.clip(in_in_acts, None, 0))}")

        print("BN ACT")
        print(f"Total act: {np.mean(in_mid_acts)}, Pos act: {np.mean(np.clip(in_mid_acts, 0, None))}, Neg act: {np.mean(np.clip(in_mid_acts, None, 0))}")
        print("-------------------------------")

        print("---MIDDLE MODULE TOP IMGS---")
        print("OUTPUT ACT")
        print(f"Total act: {np.mean(mid_out_acts)}, Pos act: {np.mean(np.clip(mid_out_acts, 0, None))}, Neg act: {np.mean(np.clip(mid_out_acts, None, 0))}")

        print("INPUT ACT")
        print(f"Total act: {np.mean(mid_in_acts)}, Pos act: {np.mean(np.clip(mid_in_acts, 0, None))}, Neg act: {np.mean(np.clip(mid_in_acts, None, 0))}")

        print("BN ACT")
        print(f"Total act: {np.mean(mid_mid_acts)}, Pos act: {np.mean(np.clip(mid_mid_acts, 0, None))}, Neg act: {np.mean(np.clip(mid_mid_acts, None, 0))}")
        print("-------------------------------")

    return {
               "out_out_acts": np.mean(out_out_acts), "out_in_acts": np.mean(out_in_acts), "out_mid_acts": np.mean(out_mid_acts),
               "in_out_acts": np.mean(in_out_acts), "in_in_acts": np.mean(in_in_acts), "in_mid_acts": np.mean(in_mid_acts),
               "mid_out_acts": np.mean(mid_out_acts), "mid_in_acts": np.mean(mid_in_acts), "mid_mid_acts": np.mean(mid_mid_acts)
           }


def mix_metric(out_out_acts, out_in_acts, out_mid_acts, in_out_acts, in_in_acts, in_mid_acts, mid_out_acts, mid_in_acts, mid_mid_acts, inverse=False, mix_only=False):
    mix_ratio = None
    if out_in_acts >= out_mid_acts:
        mix_ratio = out_mid_acts / (out_in_acts + 1e-10)
    else:
        mix_ratio = out_in_acts / (out_mid_acts + 1e-10)

    if mix_only:
        return mix_ratio

    if inverse:
        return mix_ratio * ((mid_in_acts / (in_in_acts + 1e-10)) + (in_mid_acts / (mid_mid_acts + 1e-10)))
    else:
        return mix_ratio * ((out_in_acts - mid_in_acts) + (out_mid_acts - in_mid_acts))


def xor_metric(out_out_acts, out_in_acts, out_mid_acts, in_out_acts, in_in_acts, in_mid_acts, mid_out_acts, mid_in_acts, mid_mid_acts, thresh=5):
    mix_ratio = None
    if out_in_acts >= out_mid_acts:
        mix_ratio = out_in_acts / (out_mid_acts + 1e-10)
    else:
        mix_ratio = out_mid_acts / (out_in_acts + 1e-10)

    if mix_ratio < thresh or in_out_acts < 0.5 * out_out_acts or mid_out_acts < 0.5 * out_out_acts or \
    (in_in_acts / (in_mid_acts + 1e-10)) < thresh or (mid_mid_acts / (mid_in_acts + 1e-10)) < thresh:
        return 0
    else:
        return (in_out_acts / (out_out_acts + 1e-10)) + (mid_out_acts / (out_out_acts + 1e-10))


def and_metric(out_out_acts, out_in_acts, out_mid_acts, in_out_acts, in_in_acts, in_mid_acts, mid_out_acts, mid_in_acts, mid_mid_acts, mix_thresh=0.75, cross_thresh=3):
    mix_ratio = None
    if out_in_acts >= out_mid_acts:
        mix_ratio = out_mid_acts / (out_in_acts + 1e-10)
    else:
        mix_ratio = out_in_acts / (out_mid_acts + 1e-10)

    if mix_ratio < mix_thresh or in_out_acts > 0.5 * out_out_acts or mid_out_acts > 0.5 * out_out_acts:# or \
    #(in_in_acts / (in_mid_acts + 1e-10)) < cross_thresh or (mid_mid_acts / (mid_in_acts + 1e-10)) < cross_thresh:
        return 0
    else:
        return mix_ratio + (in_out_acts / out_out_acts) + (mid_out_acts / out_out_acts)


if __name__ == "__main__":
    network = "resnet18"
    run_mode = 'stream_logic'

    extractor = None
    if network == "cornet-s":
        model = CORnet_S()
        model.load_state_dict(torch.load("/home/andrelongon/Documents/inhibition_code/weights/cornet-s.pth"))
        extractor = get_extractor_from_model(model=model, device='cuda:0', backend='pt')
    elif network == "resnet50_barlow":
        model = torch.hub.load('facebookresearch/barlowtwins:main', 'resnet50')
        extractor = get_extractor_from_model(model=model, device='cuda:0', backend='pt')
    else:
        model_params = {'weights': 'IMAGENET1K_V1'} if network == 'resnet152' or network == 'resnet50' else None

        extractor = get_extractor(
            model_name=network,
            source='torchvision',
            device='cuda:0',
            pretrained=True,
            model_parameters=model_params
        )

    # print(extractor.model)
    # exit()

    if run_mode == 'overlap':
        target_module = "layer4.2.conv3"
        input_module = "layer4.2.bn2"
        num_neurons = 2048

        intersects = np.empty(0)
        overlaps = 0
        for neuron in range(num_neurons):
        # neuron = 147
            # print(f"NEURON {neuron}")
            weights = extractor.model.state_dict()[target_module + ".weight"][neuron]

            mag_idx, pos_idx, neg_idx = top_weighted_inputs(weights)
            intersect = viz_inputs(extractor, weights, network, target_module, neuron, input_module, pos_idx, neg_idx)

            if intersect is not None:
                intersects = np.concatenate((intersects, intersect))
                overlaps += 1

        print(f"Layer {target_module} percent neurons with overlap:  {overlaps / num_neurons}")
        # print(np.unique(intersects))
    elif run_mode == 'top_weights':
        module = 'layer4.1.conv1'
        weights = extractor.model.state_dict()[module + ".weight"]

        top_weights(weights)
    elif run_mode == 'stream_logic':
        output_module = 'layer4.1.prerelu_out'
        input_module = 'layer4.0.prerelu_out'
        middle_module = 'layer4.1.bn2'
        num_neurons = 512

        xors = []
        ands = []

        scale_mid_measures = []
        scale_in_measures = []
        flip_mid_measures = []
        flip_in_measures = []
        
        mixes = []
        in_mid_acts = []
        imnet_val = False
        show_results = False
        for n in range(num_neurons):
            # print(f"\n\nNEURON {neuron}")
            acts = stream_inspect(extractor, network, output_module, input_module, middle_module, n, imnet_val=imnet_val, show_results=show_results)

            xors.append(xor_metric(**acts))
            ands.append(and_metric(**acts))

            mid_act_delta, in_act_delta = measure_invariance(extractor, network, output_module, input_module, middle_module, n, imnet_val=imnet_val, show_results=show_results)
            #   NOTE:  store deltas as a perc of max activation to control for different act ranges in the two modules.
            scale_mid_measures.append(mid_act_delta / acts['mid_mid_acts'])
            scale_in_measures.append(in_act_delta / acts['in_in_acts'])

            mid_act_delta, in_act_delta = measure_invariance(extractor, network, output_module, input_module, middle_module, n, variant="flip", imnet_val=imnet_val, show_results=show_results)
            flip_mid_measures.append(mid_act_delta / acts['mid_mid_acts'])
            flip_in_measures.append(in_act_delta / acts['in_in_acts'])
            
            converted_mix = None
            if acts['out_mid_acts'] <= 0:
                converted_mix = acts['out_in_acts'] / (1e-4)
            elif acts['out_in_acts'] <= 0:
                converted_mix = 0
            else:
                converted_mix = acts['out_in_acts'] / acts['out_mid_acts']

            #   TODO:  Plot in_out_acts to see how many channels successfully erase the channel contents.
            #   NOTE:  filter by in_mid_acts < 0 to concentrate on overwrites with negative bns (alt filter by mix < 0.5)
            # if converted_mix < 0.5:
            mixes.append(converted_mix)
            in_mid_acts.append(acts['in_mid_acts'])

        # print("XORS")
        # print(torch.nonzero(torch.tensor(xors)).shape)
        # print("\nANDS")
        # print(torch.nonzero(torch.tensor(ands)).shape)

        scale_mid_measures = torch.tensor(scale_mid_measures)
        scale_in_measures = torch.tensor(scale_in_measures)
        flip_mid_measures = torch.tensor(flip_mid_measures)
        flip_in_measures = torch.tensor(flip_in_measures)

        print(f"\nMean mix:  {torch.mean(torch.tensor(mixes))}")

        print("\nSCALES (top mid, top in)")
        #   mean in vals of top mid
        #   \n{torch.mean(scale_in_measures[torch.topk(scale_mid_measures, 9)[1]])}
        print(f"{torch.topk(scale_mid_measures, 9)}\n{torch.topk(scale_in_measures, 9)}")
        scale_mid_copy = torch.clone(scale_mid_measures)
        scale_in_copy = torch.clone(scale_in_measures)
        scale_mid_copy[torch.nonzero(scale_mid_copy < 0)] = -1e5
        scale_in_copy[torch.nonzero(scale_in_copy < 0)] = -1e5
        print(f"Top cross variance act delta:  {torch.topk(scale_mid_copy + scale_in_copy, 9)}")
        print("\nMEAN MIX FOR TOP CROSS SCALES")
        print(f"{torch.mean(torch.tensor(mixes)[torch.topk(scale_mid_copy + scale_in_copy, 9)[1]])}")

        #   TODO:  does higher cross variance delta act correlation with mix=1?

        print("\nFLIPS (top mid, top in)")
        print(f"{torch.topk(flip_mid_measures, 9)}\n{torch.topk(flip_in_measures, 9)}")
        flip_mid_copy = torch.clone(flip_mid_measures)
        flip_in_copy = torch.clone(flip_in_measures)
        flip_mid_copy[torch.nonzero(flip_mid_copy < 0)] = -1e5
        flip_in_copy[torch.nonzero(flip_in_copy < 0)] = -1e5
        print(f"Top cross variance act delta:  {torch.topk(flip_mid_copy + flip_in_copy, 9)}")
        print("\nMEAN MIX FOR TOP CROSS FLIPS")
        print(f"{torch.mean(torch.tensor(mixes)[torch.topk(flip_mid_copy + flip_in_copy, 9)[1]])}")

        #   TODO:  somehow maintain their positive or negative delta sign.  Maybe plot separately?
        # scale_mid_measures = np.array(scale_mid_measures)
        # scale_mid_measures = (scale_mid_measures - np.min(scale_mid_measures)) / (np.max(scale_mid_measures) - np.min(scale_mid_measures))
        # scale_in_measures = np.array(scale_in_measures)
        # scale_in_measures = (scale_in_measures - np.min(scale_in_measures)) / (np.max(scale_in_measures) - np.min(scale_in_measures))

        # flip_mid_measures = np.array(flip_mid_measures)
        # flip_mid_measures = (flip_mid_measures - np.min(flip_mid_measures)) / (np.max(flip_mid_measures) - np.min(flip_mid_measures))
        # flip_in_measures = np.array(flip_in_measures)
        # flip_in_measures = (flip_in_measures - np.min(flip_in_measures)) / (np.max(flip_in_measures) - np.min(flip_in_measures))

        scale_fig = px.scatter(x=scale_mid_measures, y=scale_in_measures, title="Scale")
        flip_fig = px.scatter(x=flip_mid_measures, y=flip_in_measures, title="Flip")
        scale_fig.show()
        flip_fig.show()

        r, p_val = pearsonr(scale_mid_measures, scale_in_measures)
        print(f"\nScale correlation results: r={r}, p={p_val}") 

        r, p_val = pearsonr(flip_mid_measures, flip_in_measures)
        print(f"Flip correlation results: r={r}, p={p_val}")

        # print("\nTOP OVERWRITES")
        # print(torch.topk(torch.tensor(mixes), 9, largest=False))
        # print("\nTOP BN INPUT INHIBITION")
        # print(torch.topk(torch.tensor(in_mid_acts), 9, largest=False))

        # fig = px.scatter(x=mixes, y=in_mid_acts)
        # fig.show()

        # r, p_val = pearsonr(mixes, in_mid_acts)
        # print(f"\nMix-bn_act correlation results: r={r}, p={p_val}")

        #   TODO:  auto-generate 1x3 grid of in-block-out channel feature vizs of top cross delta channels for paper figures.
        #          put code in feature_grid.py and call here.