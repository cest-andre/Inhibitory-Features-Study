import os
import sys
sys.path.append(r'/home/andre/evolve-code')
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from thingsvision import get_extractor

from modify_weights import clamp_ablate_unit

if __name__ == "__main__":
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    norm_transform = transforms.Normalize(mean=MEAN, std=STD)

    # new_transformation = transforms.GaussianBlur(kernel_size=(5,5), sigma=(5,5))
    # new_transformation = transforms.Lambda(lambda x : x + (0.1*torch.randn_like(x)))
    # new_transformation = Rotation(angles=180)
    # new_transformation = transforms.RandomHorizontalFlip(p=1)
    # new_tranform = transforms.Grayscale(num_output_channels=3)
    # new_tranform = transforms.Lambda(lambda x: torch.cat((torch.zeros(1, x.shape[1], x.shape[2]), x[1:])))
    new_transform = transforms.Lambda(lambda x: x)
    # new_tranform = transforms.Lambda(lambda x: torch.zeros(x.shape))

    batch_size = 9
    model_name = 'alexnet'
    source = 'torchvision'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    neuron_coord = None
    conv_layer = True
    if conv_layer:
        neuron_coord = 6
    module_name = 'features.10'
    unit_id = 31
    heatmap = False

    extractor = get_extractor(
        model_name=model_name,
        source=source,
        device=device,
        pretrained=True
    )

    # extractor.model.load_state_dict(
    #     clamp_ablate_unit(extractor.model.state_dict(), "features.10.weight", unit_id, min=0, max=None)
    # )

    scores = []
    top9_folder = "/home/andre/tuning_curves/inh_ablate/layer11_neuron31/max/inh"
    # top9_folder = ""
    # img_paths = [os.path.join(top9_folder, file) for file in os.listdir(top9_folder)]

    # img_paths.append(r"/home/andre/evolved_data/alexnet_.features.Conv2d10_unit25/Avg_Exc_Prototype_alexnet_.features.Conv2d10.png")
    # img_paths.append(r"/home/andre/evolved_data/alexnet_.features.Conv2d10_unit25/Avg_Inh_Prototype_alexnet_.features.Conv2d10.png")

    #   Proto load.
    # exc_proto = Image.open(r"/home/andre/evolved_data/alexnet_.features.Conv2d10_unit25/Avg_Exc_Prototype_alexnet_.features.Conv2d10.png")
    # inh_proto = Image.open(r'/home/andre/evolved_data/alexnet_.features.Conv2d10_unit25/Avg_Inh_Prototype_alexnet_.features.Conv2d10.png')

    for file in os.listdir(top9_folder):
    # for i in range(2, 224, 2):
    #     file = f"top0_unit250_{i}.png"
        print(file)
        path = os.path.join(top9_folder, file)
        img = Image.open(path)
        img = transforms.ToTensor()(img)
        #   Perform new transform here before norm!
        img = new_transform(img)
        transformed_img = transforms.ToPILImage()(img)
        # transformed_img.save(f"/home/andre/tuning_curves/intact/layer11_neuron25/transformed/inverted_{file}")
        img = norm_transform(img)
        img = torch.unsqueeze(img, 0)
        img = torch.unsqueeze(img, 0)

        features = extractor.extract_features(
            batches=img,
            module_name=module_name,
            flatten_acts=False
        )

        score = None
        if neuron_coord is not None:
            score = features[:, unit_id, neuron_coord, neuron_coord]
        else:
            score = features[:, unit_id]

        print(score)
        scores.append(score)

        if heatmap:
            map = -1 * torch.from_numpy(features[:, unit_id])
            resize_featmap = F.interpolate(map.reshape(1, 1, map.shape[1], map.shape[1]), size=(224, 224), mode='nearest')[
                0, 0]  # , align_corners=True
            min_val = torch.min(resize_featmap)
            max_val = torch.max(resize_featmap)
            # Normalize the tensor between 0 and 1
            resize_featmap = torch.div(resize_featmap - min_val, max_val - min_val)

            heatmap_img = resize_featmap * transforms.ToTensor()(transformed_img)
            heatmap_img = transforms.ToPILImage()(heatmap_img)
            heatmap_img.save(f"/home/andre/tuning_curves/intact/layer11_neuron31/max/heatmap_bot_{file}")

    score_mean = np.mean(np.array(scores))
    score_std = np.std(np.array(scores))
    print(f"Avg:  {score_mean}")
    print(f"std:  {score_std}")
    #
    # plt.scatter(np.array(range(2, 224, 2)), np.array(scores), color='b')
    # plt.title("Top->Random (Top 0 for Unit 250) Center Transplant")
    # plt.xlabel("Transplant Size")
    # plt.ylabel("Activation")
    # plt.savefig("/home/andre/tuning_curves/intact/layer11_neuron25/top0_unit250_transplant.png")