import sys
import os
import copy
import argparse
sys.path.append('/home/andrelongon/Documents/inhibition_code/lucent')
from thingsvision import get_extractor
from PIL import Image
import numpy as np
import torch
from torchvision import models, transforms
from lucent.optvis import render, objectives, transform
from lucent.modelzoo.util import get_model_layers
import lucent.optvis.param as param

from transplant import get_activations
from modify_weights import clamp_ablate_unit


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str)
    parser.add_argument('--module', type=str)
    # parser.add_argument('--weight_name', type=str)
    # parser.add_argument('--obj_layer', type=str)
    parser.add_argument('--neuron', type=int)
    parser.add_argument('--type', type=str)
    parser.add_argument('--ablate', action='store_true', default=False)
    args = parser.parse_args()

    basedir = None
    if args.ablate:
        basedir = "/media/andrelongon/DATA/feature_viz/ablated"
    else:
        basedir = "/media/andrelongon/DATA/feature_viz/intact"

    savedir = os.path.join(basedir, args.network, args.module, f"unit{args.neuron}")
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

    extractor = get_extractor(model_name=args.network, source='torchvision', device='cuda:0', pretrained=True)

    if states is not None:
        extractor.model.load_state_dict(states)
    else:
        states = extractor.model.state_dict()

    states = copy.deepcopy(states)

    # print(extractor.model)
    # print(get_model_layers(extractor.model))
    # exit()

    # JITTER = 1
    # ROTATE = 5
    # SCALE  = 1.1

    # transforms = [
    #     transform.pad(2*JITTER),
    #     transform.jitter(JITTER),
    #     transform.random_scale([SCALE ** (n/10.) for n in range(-10, 11)]),
    #     transform.random_rotate(range(-ROTATE, ROTATE+1))
    # ]
    param_f = lambda: param.images.image(224, decorrelate=True)

    extractor.model.eval()
    for i in range(1):
        # savedir = os.path.join(basedir, args.network, f"{args.module}_unit{i}")
        # if not os.path.isdir(savedir):
        #     os.mkdir(savedir)
        #     os.mkdir(os.path.join(savedir, "pos"))
        #     os.mkdir(os.path.join(savedir, "neg"))

        for j in range(2):
            obj = objectives.neuron(args.module.replace('.', '_'), args.neuron, negative=j)
            # obj = objectives.neuron("layer4_1_conv2", args.neuron, negative=j)

            # trans = [transforms.Lambda(lambda x: x + torch.normal(0, 32, size=x.shape, device="cuda"))]
            
            if args.ablate:
                extractor.model.load_state_dict(
                    clamp_ablate_unit(extractor.model.state_dict(), args.module + ".weight", args.neuron,
                                      min=(None if j else 0), max=(0 if j else None))
                )

            imgs = render.render_vis(extractor.model, obj, param_f=param_f, transforms=transform.standard_transforms, thresholds=(512,), show_image=False)

            # img = torch.tensor(imgs[0]).to("cuda:0")
            # img = torch.permute(img, (0, 3, 1, 2))
            # img = transform.normalize()(img)
            # act = get_activations(extractor, img, args.module, neuron_coord=3, channel_id=args.neuron)
            # print(act)

            img = Image.fromarray((imgs[0][0]*255).astype(np.uint8))
            img.save(os.path.join(savedir, subdirs[j], f"{i}.png"))
            # img.save(os.path.join(savedir, subdirs[j], f"0.png"))

            if args.ablate:
                extractor.model.load_state_dict(states)