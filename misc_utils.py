import numpy as np
import torch
import torchvision
from torchvision import models, transforms
from thingsvision import get_extractor, get_extractor_from_model
from PIL import Image
import torch.distributed as dist
from torch.nn.parallel import DataParallel

from cornet_s import CORnet_S

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
    

# if __name__ == '__main__':
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