import argparse
import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from thingsvision import get_extractor
import matplotlib.pyplot as plt
import numpy as np

#   TODO
#   Scale up to handle folders of images (path is a directory).

def get_activations(extractor, x, module_name, neuron_coord=None, channel_id=None):
    if len(x.shape) == 3:
        x = torch.unsqueeze(x, 0)

    x = torch.unsqueeze(x, 0)

    features = extractor.extract_features(
        batches=x,
        module_name=module_name,
        flatten_acts=False
    )

    activations = features
    if channel_id is not None:
        activations = features[:, channel_id]

        if neuron_coord is not None:
            activations = features[:, channel_id, neuron_coord, neuron_coord]

    return activations


def center_transplant(map, donor_img, recipient_img, save_path):
    #   Exclude perimeter from max_val.  We will form a 3x3 grid surrounding
    #   the max_val block so this ensures the transplant region is all contained in the image.
    max_val = torch.max(transforms.CenterCrop(map.shape[1] - 1)(map))
    max_region = torch.nonzero(map[0] == max_val)[0]

    block_size = 224 // map.shape[1]
    donor_img = transforms.ToTensor()(donor_img)
    donor_crop = donor_img[:,
                    (max_region[0] * block_size) - block_size:(max_region[0] * block_size) + (2 * block_size),
                    (max_region[1] * block_size) - block_size:(max_region[1] * block_size) + (2 * block_size)
                 ]

    #   Now that the crop is obtained, set the centered 3x3 equiv-sized grid of target_img
    #   equal to this crop.
    new_img = transforms.ToTensor()(recipient_img)
    left_anchor = 111 - (block_size // 2)
    right_anchor = 111 + (block_size // 2)
    new_img[:,
        left_anchor - block_size:right_anchor + 1 + block_size,
        left_anchor - block_size:right_anchor + 1 + block_size] = donor_crop

    new_img = transforms.ToPILImage()(new_img)
    new_img.save(os.path.join(save_path, f"center_transplant_unit25.png"))

    return new_img


def percentile_transplant(map, donor_img, recipient_img, save_path):
    donor_img = transforms.ToTensor()(donor_img)
    recipient_img = transforms.ToTensor()(recipient_img)

    map = F.interpolate(map.reshape(1, 1, map.shape[1], map.shape[1]), size=(224, 224), mode='nearest')[0]
    min_val = torch.min(map)
    max_val = torch.max(map)
    map = torch.div(map - min_val, max_val - min_val)
    #   Find elements of map that are within the xth percentile.
    map = torch.broadcast_to(map, (3, map.shape[1], map.shape[2]))

    perc_imgs = []
    for p in range(25, 100, 5):
        perc = p * 0.01
        plant_img = torch.where(map > torch.quantile(map, perc), donor_img, recipient_img)
        plant_img = transforms.ToPILImage()(plant_img)
        plant_img.save(os.path.join(save_path, "percentile", f"{p}_transplant.png"))

        #   Reverse order so that plot will be from smallest to largest transplants.
        perc_imgs.append(plant_img)
        # perc_imgs.insert(0, plant_img)

    return perc_imgs


def heatmap_transplant(donor_img, recipient_img, save_path,
                       model_name, source, neuron_coord,
                       module_name, channel_id, map_type):

    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    norm_transform = transforms.Normalize(mean=MEAN, std=STD)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    extractor = get_extractor(
        model_name=model_name,
        source=source,
        device=device,
        pretrained=True
    )

    img = transforms.ToTensor()(donor_img)
    img = norm_transform(img)

    features = get_activations(extractor, img, module_name, None, channel_id)
    print(f"Donor activation:  {features[0, neuron_coord, neuron_coord]}")

    map = torch.from_numpy(features)
    if map_type == "cold":
        map = -1 * map
    # map = torch.clamp(map, min=0, max=None)

    # new_img = center_transplant(map, donor_img, recipient_img, save_path)
    plant_imgs = percentile_transplant(map, donor_img, recipient_img, save_path)

    perc_acts = []
    for img in plant_imgs:
        img = transforms.ToTensor()(img)
        img = norm_transform(img)

        perc_acts.append(get_activations(extractor, img, module_name, neuron_coord, channel_id))

    #   Activation for target img before and after transplant.
    recipient_img = transforms.ToTensor()(recipient_img)
    recipient_img = norm_transform(recipient_img)

    perc_acts.append(get_activations(extractor, recipient_img, module_name, neuron_coord, channel_id))

    plt.scatter(np.arange(25, 105, 5), np.array(perc_acts))
    plt.xlabel("Percentile")
    plt.ylabel("Activation")
    plt.savefig(os.path.join(save_path, "percentile", "percentile_act_plot.png"))


def center_grow_transplant(donor_img, recipient_img, save_path):
    for crop_size in range(2, 224, 2):
        recipient_tensor = transforms.ToTensor()(recipient_img)
        recipient_tensor = transforms.CenterCrop(crop_size)(recipient_tensor)
        pad_size = (224-crop_size) // 2
        recipient_tensor = F.pad(recipient_tensor, (pad_size, pad_size, pad_size, pad_size), value=-1)

        donor_tensor = transforms.ToTensor()(donor_img)
        donor_tensor = torch.where(recipient_tensor != -1, recipient_tensor, donor_tensor)

        donor_tensor = transforms.ToPILImage()(donor_tensor)
        donor_tensor.save(os.path.join(save_path, f"top0_unit250_{crop_size}.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--donor_path', type=str)
    parser.add_argument('--recipient_path', type=str)
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--map_type', choices=['hot', 'cold'], default=None)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--source', type=str, default='torchvision')
    parser.add_argument('--neuron_coord', type=int, default=None)
    parser.add_argument('--module_name', type=str)
    parser.add_argument('--channel_id', type=int)
    args = parser.parse_args()

    donor_img = Image.open(args.donor_path)
    recipient_img = Image.open(args.recipient_path)

    if args.map_type is not None:
        savedir = os.path.join(args.save_path, f"{args.model_name}_{args.module_name}_unit{args.channel_id}")
        if not os.path.isdir(savedir):
            os.mkdir(savedir)
            os.mkdir(os.path.join(savedir, "percentile"))
        else:
            exit()

        heatmap_transplant(donor_img, recipient_img, savedir,
                           args.model_name, args.source, args.neuron_coord,
                           args.module_name, args.channel_id, args.map_type)
    else:
        center_grow_transplant(donor_img, recipient_img, args.save_path)