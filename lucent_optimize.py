import sys
import os
import argparse
sys.path.append(r'/home/andre/evolve-code/lucent')
from thingsvision import get_extractor
from PIL import Image
import numpy as np
import torch
from torchvision import models, transforms
from lucent.optvis import render, objectives, transform
import lucent.optvis.param as param

from transplant import get_activations


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str)
    parser.add_argument('--module', type=str)
    parser.add_argument('--neuron', type=int)
    parser.add_argument('--type', type=str)
    args = parser.parse_args()

    savedir = os.path.join("/home/andre/lucent_imgs", args.type, f"{args.neuron}")
    subdirs = ["pos", "neg"]
    if not os.path.isdir(savedir):
        os.mkdir(savedir)
        os.mkdir(os.path.join(savedir, "pos"))
        os.mkdir(os.path.join(savedir, "neg"))

    states = None
    if args.type == 'alexnet_untrained':
        states = torch.load("/home/andre/tuning_curves/untrained_alexnet/random_weights.pth")
    elif args.type == 'resnet18_untrained':
        states = torch.load("/home/andre/tuning_curves/untrained_resnet18/random_weights.pth")
    elif args.type == 'resnet18_robust':
        states = torch.load("/home/andre/model_weights/resnet-18-l2-eps3.pt")

    extractor = get_extractor(model_name=args.network, source='torchvision', device='cuda', pretrained=True)

    if states is not None:
        extractor.model.load_state_dict(states)

    for i in range(9):
        for j in range(2):
            obj = objectives.neuron("features->10", args.neuron, negative=j)

            # trans = [transforms.Lambda(lambda x: x + torch.normal(0, 32, size=x.shape, device="cuda"))]
            param_f = lambda: param.images.image(224)

            imgs = render.render_vis(
                extractor.model, obj, param_f, transforms=[], thresholds=(1024,), show_image=False
            )

            img = torch.tensor(imgs[0])
            img = torch.permute(img, (0, 3, 1, 2))
            img = transform.normalize()(img)
            act = get_activations(extractor, img, args.module, neuron_coord=6, channel_id=args.neuron)
            print(act)

            img = Image.fromarray((imgs[0][0]*255).astype(np.uint8))
            # img = transforms.ToPILImage()((imgs[0][0]*255).astype(np.uint8))
            img.save(os.path.join(savedir, subdirs[j], f"{i}.png"))