import numpy as np
import torch
import torchvision
from torchvision import models, transforms
from thingsvision import get_extractor, get_extractor_from_model
from PIL import Image

import torch.distributed as dist
from torch.nn.parallel import DataParallel

from train_imnet import AlexNet
from transplant import get_activations


def check_background_inh(extractor, input_layer, target_layer):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ])
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    norm_transform = transforms.Normalize(mean=MEAN, std=STD)

    imnet_folder = r"/home/andrelongon/Documents/data/imagenet/val"
    imagenet_data = torchvision.datasets.ImageFolder(imnet_folder, transform=transform)
    dataloader = torch.utils.data.DataLoader(imagenet_data, batch_size=1024, shuffle=True, drop_last=False)

    rand_stims, _ = next(iter(dataloader))

    norm_inputs = norm_transform(rand_stims)
    acts = get_activations(extractor, norm_inputs, input_layer)
    acts = torch.tensor(acts)
    print(f"Activation shape: {acts.shape}")

    #   TODO:  calculate mean activation for total and center vs corner.  maxpool will eliminate all negatives.
    print("\n\nProportion Negatives (zero after relu)")
    print(f"All: {torch.nonzero(torch.flatten(acts[:, :]) < 0).shape[0] / torch.flatten(acts[:, :]).shape[0]}")

    #   TODO:  separate center region from surround for separate activation comparisons.
    center_idx = acts.shape[-1] // 3

    center_acts = acts[:, :, center_idx:acts.shape[-1]-center_idx, center_idx:acts.shape[-1]-center_idx]
    print(f"Center: {torch.nonzero(torch.flatten(center_acts) < 0).shape[0] / torch.flatten(center_acts).shape[0]}")

    corner_acts = acts[:, :, 0:center_idx, 0:center_idx]
    print(f"Upper corner:  {torch.nonzero(torch.flatten(corner_acts) < 0).shape[0] / torch.flatten(corner_acts).shape[0]}")


    #   TODO:  Get acts from prev layer and weights from target layer.  Compute neg and pos input quantities in center vs surround.
    #          This will better demonstrate what regions pos and neg weights of a kernel prefer.
    # input_acts = get_activations(extractor, norm_inputs, input_layer)
    # input_acts = torch.tensor(input_acts)
    # print(input_acts.shape)
    #   clamp negatives of input_acts to emulate relu.
    input_acts = torch.clamp(acts, min=0)

    weights = extractor.model.state_dict()[target_layer + '.weight'].cpu()
    print(f"negative / total weights:  {torch.nonzero(torch.flatten(weights) < 0).shape[0] / torch.flatten(weights).shape[0]}")
    # print(weights.shape)

    center_idx = input_acts.shape[-1] // 2
    idx_offset = weights.shape[-1] // 2

    center_acts = input_acts[:, :, center_idx-idx_offset:center_idx+idx_offset+1, center_idx-idx_offset:center_idx+idx_offset+1]
    corner_acts = input_acts[:, :, 0:weights.shape[-1], 0:weights.shape[-1]]    #  upper left corner
    # corner_acts = input_acts[:, :, -weights.shape[-1]:, -weights.shape[-1]:]      #  bottom right corner

    # print(center_acts.shape)
    # print(corner_acts.shape)

    center_acts = torch.flatten(center_acts, start_dim=1, end_dim=-1)
    corner_acts = torch.flatten(corner_acts, start_dim=1, end_dim=-1)

    # channel_idx = 0
    center_np_ratio = 0
    corner_np_ratio = 0
    for c in range(weights.shape[0]):
        chan_weights = torch.flatten(weights[c])

        center_pos_in = torch.mean(center_acts[:, torch.nonzero(chan_weights > 0)])
        center_neg_in = torch.mean(center_acts[:, torch.nonzero(chan_weights < 0)])

        corner_pos_in = torch.mean(corner_acts[:, torch.nonzero(chan_weights > 0)])
        corner_neg_in = torch.mean(corner_acts[:, torch.nonzero(chan_weights < 0)])

        #   TODO:  Compute lesioned acts and take ratios (should I control for neg pos weight imbalance in this case?
        #          Or should that be a highlighted feature?)


        # print(f"\nChannel:  {c}")
        # print(f"Center Input --  pos:  {center_pos_in},  neg:  {center_neg_in},  neg / pos:  {center_neg_in / center_pos_in}")
        # print(f"Corner Input --  pos:  {corner_pos_in},  neg:  {corner_neg_in},  neg / pos:  {corner_neg_in / corner_pos_in}")

        center_np_ratio += center_neg_in / center_pos_in
        corner_np_ratio += corner_neg_in / corner_pos_in

    print(f"\n\nCenter Input --  neg / pos:  {center_np_ratio / weights.shape[0]}")
    print(f"Corner Input --  neg / pos:  {corner_np_ratio / weights.shape[0]}")

    #   TODO:  How to compensate for more zeros in one region versus other?  Does ratio comparison accomplish this?



def stim_compare(stim_sets, metric, mask=None):
    set1, set2 = stim_sets

    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    norm_trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])

    if mask is not None:
        set1 = [img * mask for img in set1]
        set2 = [img * mask for img in set2]

    all_distances = []
    for i in range(len(set1)):
        #   Use inner for j in range(start=j, stop=len(min_protos)) to perform all comparison permutations.
        stim = norm_trans(set1[i]).cuda()

        for j in range(len(set2)):
            compare_stim = norm_trans(set2[j]).cuda()

            d = metric(stim, compare_stim)
            all_distances.append(d.detach().cpu().item())

    return np.array(all_distances)


def weight_mixture_measure(weight_name):
    weights = models.resnet18(True).state_dict()[weight_name]
    print(weights.shape)

    all_pos = []
    for i in range(weights.shape[0]):
        w = torch.flatten(weights[i], start_dim=1, end_dim=2)
        all_pos.append(torch.nonzero(torch.logical_or(torch.all(w > 0, 1), torch.all(w < 0, 1))).shape[0] / w.shape[0])

    max_pos = torch.argmin(torch.tensor(all_pos))
    print(all_pos[max_pos])
    print(max_pos)

    print(torch.mean(torch.tensor(all_pos, dtype=torch.float)))


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
    

if __name__ == '__main__':
    extractor = get_extractor(model_name="googlenet", source='torchvision', device='cpu', pretrained=True)
#   resnet18
#     measure_specialization(extractor, "layer1.0.conv1.weight")
#     measure_specialization(extractor, "layer1.0.conv2.weight")
#     measure_specialization(extractor, "layer1.1.conv1.weight")
#     measure_specialization(extractor, "layer1.1.conv2.weight")
#     measure_specialization(extractor, "layer2.0.conv1.weight")
#     measure_specialization(extractor, "layer2.0.conv2.weight")
#     measure_specialization(extractor, "layer2.1.conv1.weight")
#     measure_specialization(extractor, "layer2.1.conv2.weight")
#     measure_specialization(extractor, "layer3.0.conv1.weight")
#     measure_specialization(extractor, "layer3.0.conv2.weight")

#     measure_specialization(extractor, "layer3.1.conv1.weight", input_module="layer3.0")

#     measure_specialization(extractor, "layer3.1.conv2.weight")
# #     # print("LAYER4.0")
#     measure_specialization(extractor, "layer4.0.conv1.weight")
#     measure_specialization(extractor, "layer4.0.conv2.weight")
#     measure_specialization(extractor, "layer4.0.conv3.weight")
# #     # # print("LAYER4.1")
#     measure_specialization(extractor, "layer4.1.conv1.weight")

#     measure_specialization(extractor, "layer4.1.conv2.weight", input_module="layer4.1.bn1")

#     measure_specialization(extractor, "layer4.1.conv3.weight")
# #   resnet50
#     # print("LAYER4.2")
#     measure_specialization(extractor, "layer4.2.conv1.weight")
#     measure_specialization(extractor, "layer4.2.conv2.weight")
#     measure_specialization(extractor, "layer4.2.conv3.weight")

#   alexnet
    # measure_specialization(extractor, "features.3.weight")
    # measure_specialization(extractor, "features.6.weight")
    # measure_specialization(extractor, "features.8.weight")
    # measure_specialization(extractor, "features.10.weight")

#   googlenet
    # print("4E")
    # measure_specialization(extractor, "inception4e.branch1.conv.weight")
    measure_specialization(extractor, "inception4e.branch2.1.conv.weight", input_module="inception4e.branch2.0.bn")
    # measure_specialization(extractor, "inception4e.branch3.1.conv.weight")
    # measure_specialization(extractor, "inception4e.branch4.1.conv.weight")
    # print("5A")
    # measure_specialization(extractor, "inception5a.branch1.conv.weight")
    # measure_specialization(extractor, "inception5a.branch2.1.conv.weight")
    # measure_specialization(extractor, "inception5a.branch3.1.conv.weight")
    # measure_specialization(extractor, "inception5a.branch4.1.conv.weight")
    # print("5B")
    # measure_specialization(extractor, "inception5b.branch1.conv.weight")
    # measure_specialization(extractor, "inception5b.branch2.1.conv.weight")
    # measure_specialization(extractor, "inception5b.branch3.1.conv.weight")
    # measure_specialization(extractor, "inception5b.branch4.1.conv.weight")

#   vgg16
    # measure_specialization(extractor, "features.24.weight")
    # measure_specialization(extractor, "features.26.weight")
    # measure_specialization(extractor, "features.28.weight")



# weight_mixture_measure("layer1.1.conv2.weight")


# model_name = 'cornet-s'
# source_name = 'custom' if model_name == 'cornet-s' else 'torchvision'
# extractor = get_extractor(
#     model_name=model_name,
#     source=source_name,
#     device='cuda',
#     pretrained=True
# )

# model = AlexNet().cpu()
# model = DataParallel(model, device_ids=None)
# states = torch.load("/media/andrelongon/DATA/imnet_weights/imnet_alexnet_less_pool_10ep.pth", map_location='cpu')
# model.load_state_dict(states)

# model = model.module

# extractor = get_extractor_from_model(model=model, device='cpu', backend='pt')

# #   Second conv
# # check_background_inh(extractor, 'features.3', 'features.4')

# #   last conv (features.14 and 15 for missing pool model)
# check_background_inh(extractor, 'features.14', 'features.15')