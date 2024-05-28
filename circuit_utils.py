import torch
from torchvision import transforms
from thingsvision import get_extractor, get_extractor_from_model
from PIL import Image
import numpy as np

from cornet_s import CORnet_S
from transplant import get_activations


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
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    norm_trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])

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
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    norm_trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])

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


def feature_overwrite(extractor, model_name, output_module, input_module, middle_module, neuron, imnet_val=False):
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    norm_trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])

    top_img = None
    if imnet_val:
        top_img = Image.open(f"/media/andrelongon/DATA/tuning_curves/{model_name}/{output_module}/intact/{output_module}_neuron{neuron}/max/exc/0.png")
    else:
        top_img = Image.open(f"/media/andrelongon/DATA/feature_viz/intact/{model_name}/{output_module}/unit{neuron}/pos/0_distill_channel.png")

    top_img = norm_trans(top_img)

    #   NOTE:  retrieve center act from channel as this is how top imnet imgs are sorted.
    output_acts = get_activations(extractor, top_img, output_module, channel_id=neuron, use_center=imnet_val)
    input_acts = get_activations(extractor, top_img, input_module, channel_id=neuron, use_center=imnet_val)
    middle_acts = get_activations(extractor, top_img, middle_module, channel_id=neuron, use_center=imnet_val)

    print("---OUTPUT MODULE FEATURE VIZ---")
    print("OUTPUT ACT")
    print(f"Total act: {np.sum(output_acts)}, Pos act: {np.sum(np.clip(output_acts, 0, None))}, Neg act: {np.sum(np.clip(output_acts, None, 0))}")

    print("INPUT ACT")
    print(f"Total act: {np.sum(input_acts)}, Pos act: {np.sum(np.clip(input_acts, 0, None))}, Neg act: {np.sum(np.clip(input_acts, None, 0))}")

    print("BN3 ACT")
    print(f"Total act: {np.sum(middle_acts)}, Pos act: {np.sum(np.clip(middle_acts, 0, None))}, Neg act: {np.sum(np.clip(middle_acts, None, 0))}")
    print("-------------------------------")

    if imnet_val:
        top_img = Image.open(f"/media/andrelongon/DATA/tuning_curves/{model_name}/{input_module}/intact/{input_module}_neuron{neuron}/max/exc/0.png")
    else:
        top_img = Image.open(f"/media/andrelongon/DATA/feature_viz/intact/{model_name}/{input_module}/unit{neuron}/pos/0_distill_channel.png")

    top_img = norm_trans(top_img)

    output_acts = get_activations(extractor, top_img, output_module, channel_id=neuron, use_center=imnet_val)
    input_acts = get_activations(extractor, top_img, input_module, channel_id=neuron, use_center=imnet_val)
    middle_acts = get_activations(extractor, top_img, middle_module, channel_id=neuron, use_center=imnet_val)

    print("---INPUT MODULE FEATURE VIZ---")
    print("OUTPUT ACT")
    print(f"Total act: {np.sum(output_acts)}, Pos act: {np.sum(np.clip(output_acts, 0, None))}, Neg act: {np.sum(np.clip(output_acts, None, 0))}")

    print("INPUT ACT")
    print(f"Total act: {np.sum(input_acts)}, Pos act: {np.sum(np.clip(input_acts, 0, None))}, Neg act: {np.sum(np.clip(input_acts, None, 0))}")

    print("BN3 ACT")
    print(f"Total act: {np.sum(middle_acts)}, Pos act: {np.sum(np.clip(middle_acts, 0, None))}, Neg act: {np.sum(np.clip(middle_acts, None, 0))}")
    print("-------------------------------")

    if imnet_val:
        top_img = Image.open(f"/media/andrelongon/DATA/tuning_curves/{model_name}/{middle_module}/intact/{middle_module}_neuron{neuron}/max/exc/0.png")
    else:
        top_img = Image.open(f"/media/andrelongon/DATA/feature_viz/intact/{model_name}/{middle_module}/unit{neuron}/pos/0_distill_channel.png")

    top_img = norm_trans(top_img)

    output_acts = get_activations(extractor, top_img, output_module, channel_id=neuron, use_center=imnet_val)
    input_acts = get_activations(extractor, top_img, input_module, channel_id=neuron, use_center=imnet_val)
    middle_acts = get_activations(extractor, top_img, middle_module, channel_id=neuron, use_center=imnet_val)

    print("---MIDDLE MODULE FEATURE VIZ---")
    print("OUTPUT ACT")
    print(f"Total act: {np.sum(output_acts)}, Pos act: {np.sum(np.clip(output_acts, 0, None))}, Neg act: {np.sum(np.clip(output_acts, None, 0))}")

    print("INPUT ACT")
    print(f"Total act: {np.sum(input_acts)}, Pos act: {np.sum(np.clip(input_acts, 0, None))}, Neg act: {np.sum(np.clip(input_acts, None, 0))}")

    print("BN3 ACT")
    print(f"Total act: {np.sum(middle_acts)}, Pos act: {np.sum(np.clip(middle_acts, 0, None))}, Neg act: {np.sum(np.clip(middle_acts, None, 0))}")
    print("-------------------------------")


def top_weights(weights):
    for c in range(weights.shape[0]):
        print(f"CHANNEL {c}")
        _, w_idx = torch.topk(torch.flatten(weights[c]), 9)
        print(f"TOP WEIGHTS: {w_idx}")

        if c == 100:
            break


if __name__ == "__main__":
    network = "resnet50"
    run_mode = 'stream_logic'

    extractor = None
    if network == "cornet-s":
        model = CORnet_S()
        model.load_state_dict(torch.load("/home/andrelongon/Documents/inhibition_code/weights/cornet-s.pth"))
        extractor = get_extractor_from_model(model=model, device='cpu', backend='pt')
    elif network == "resnet50_barlow":
        model = torch.hub.load('facebookresearch/barlowtwins:main', 'resnet50')
        extractor = get_extractor_from_model(model=model, device='cpu', backend='pt')
    else:
        model_params = {'weights': 'IMAGENET1K_V1'} if network == 'resnet152' or network == 'resnet50' else None

        extractor = get_extractor(
            model_name=network,
            source='torchvision',
            device='cpu',
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
    elif run_mode == 'stream_logic':
        output_module = 'layer4.1'
        input_module = 'layer4.0'
        middle_module = 'layer4.1.bn3'
        # middle_name = 'layer3.5.bn3'
        num_neurons = 9

        for neuron in range(num_neurons):
            print(f"\n\nNEURON {neuron}")
            feature_overwrite(extractor, network, output_module, input_module, middle_module, neuron, imnet_val=False)

    elif run_mode == 'top_weights':
        module = 'layer4.0.downsample.0'
        weights = extractor.model.state_dict()[module + ".weight"]

        top_weights(weights)