import sys
import os
import copy
import argparse
sys.path.append('/home/andrelongon/Documents/inhibition_code/lucent')
from thingsvision import get_extractor, get_extractor_from_model
from PIL import Image
import numpy as np
import torch
from torchvision import models, transforms
from lucent.optvis import render, objectives, transform
from lucent.modelzoo.util import get_model_layers
import lucent.optvis.param as param

from cornet_s import CORnet_S

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
    parser.add_argument('--skip_neg', action='store_true', default=False)
    args = parser.parse_args()

    print(f"BEGIN MODULE {args.module} NEURON {args.neuron}")
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
    extractor = None
    if args.type == 'alexnet_untrained':
        states = torch.load("/home/andre/tuning_curves/untrained_alexnet/random_weights.pth")
    elif args.type == 'resnet18_untrained':
        states = torch.load("/home/andre/tuning_curves/untrained_resnet18/random_weights.pth")
    elif args.type == 'resnet18_robust':
        states = torch.load("/home/andre/model_weights/resnet-18-l2-eps3.pt")
    elif args.network == 'resnet50_barlow':
        model = torch.hub.load('facebookresearch/barlowtwins:main', 'resnet50')
        extractor = get_extractor_from_model(model=model, device='cuda:0', backend='pt')

    if args.network == 'cornet-s':
        cornet = CORnet_S()
        cornet.load_state_dict(torch.load("/home/andrelongon/Documents/inhibition_code/weights/cornet-s.pth"))
        extractor = get_extractor_from_model(model=cornet, device='cuda:0', backend='pt')
    elif extractor is None:
        model_params = {'weights': 'IMAGENET1K_V1'} if args.network == 'resnet152' or args.network == 'resnet50' else None
        extractor = get_extractor(model_name=args.network, source='torchvision', device='cuda:0', pretrained=True, model_parameters=model_params)

    if states is not None:
        extractor.model.load_state_dict(states)
    else:
        states = extractor.model.state_dict()

    states = copy.deepcopy(states)

    # print(extractor.model)
    # print(get_model_layers(extractor.model))
    # exit()

    transforms = [
        transform.pad(16),
        transform.jitter(16),
        transform.random_scale([1, 0.975, 1.025, 0.95, 1.05]),
        transform.random_rotate([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]),
        transform.jitter(8),
        transform.crop(224)
    ]
    param_f = lambda: param.images.image(224, decorrelate=True)

    extractor.model.eval()
    for i in range(3):
        # savedir = os.path.join(basedir, args.network, f"{args.module}_unit{i}")
        # if not os.path.isdir(savedir):
        #     os.mkdir(savedir)
        #     os.mkdir(os.path.join(savedir, "pos"))
        #     os.mkdir(os.path.join(savedir, "neg"))

        for j in range(2):
            obj = objectives.channel(args.module.replace('.', '_'), args.neuron, negative=j)
            # obj = objectives.neuron("layer4_1_conv2", args.neuron, negative=j)

            # trans = [transforms.Lambda(lambda x: x + torch.normal(0, 32, size=x.shape, device="cuda"))]
            
            if args.ablate:
                extractor.model.load_state_dict(
                    clamp_ablate_unit(extractor.model.state_dict(), args.module + ".weight", args.neuron,
                                      min=(None if j else 0), max=(0 if j else None))
                )

            imgs = render.render_vis(extractor.model, obj, param_f=param_f, transforms=transforms, thresholds=(2560,), show_image=False)

            # img = torch.tensor(imgs[0]).to("cuda:0")
            # img = torch.permute(img, (0, 3, 1, 2))
            # img = transform.normalize()(img)
            # act = get_activations(extractor, img, args.module, neuron_coord=3, channel_id=args.neuron)
            # print(act)

            img = Image.fromarray((imgs[0][0]*255).astype(np.uint8))
            img.save(os.path.join(savedir, subdirs[j], f"{i}_distill_channel.png"))
            # img.save(os.path.join(savedir, subdirs[j], f"0.png"))

            if args.ablate:
                extractor.model.load_state_dict(states)

            if args.skip_neg:
                break