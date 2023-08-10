import sys
sys.path.append(r'/home/andre/evolve-code/lucent')
from thingsvision import get_extractor
from PIL import Image
import numpy as np
from torchvision import models, transforms
from lucent.optvis import render, objectives
import lucent.optvis.param as param

from transplant import get_activations


if __name__ == '__main__':
    extractor = get_extractor(model_name='alexnet', source='torchvision', device='cuda', pretrained=True)
    obj = objectives.neuron("features->10", 0, x=6, y=6, batch=0)

    transforms = []

    imgs = render.render_vis(
        extractor.model, obj, param_f=lambda: param.images.image(224, sd=0.1), transforms=transforms,
        show_image=False, fixed_image_size=224
    )

    img = transforms.ToPILImage()((imgs[0][0]*255).astype(np.uint8))
    img.save("/home/andre/misc_data/lucent_imgs/alexnet_trained_conv10_unit0.png")