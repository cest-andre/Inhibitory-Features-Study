from os.path import join
from thingsvision import get_extractor
from evolve_utils import load_GAN, label2optimizer, resize_and_pad, visualize_trajectory
from torchvision import transforms
from torchvision.utils import make_grid
import numpy as np
import torch
from PIL import Image

from circuit_toolkit.Optimizers import CholeskyCMAES

savedir = r"/home/andre/evolved_data/thingsvision/inhib_mix_evolve"

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
norm_trans = transforms.Normalize(mean=MEAN, std=STD)
imnet_trans = transforms.Compose([
    transforms.CenterCrop(224),
    norm_trans
])

inv_trans = transforms.Compose([
    transforms.Normalize(mean = [ 0., 0., 0. ], std = [ 1/0.229, 1/0.224, 1/0.225 ]),
    transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ], std = [ 1., 1., 1. ])
])

exc_prototype = Image.open(r"/home/andre/evolved_data/alexnet_.features.Conv2d10_unit25/exc/Best_Img_alexnet_.features.Conv2d10_7.png")
exc_prototype = transforms.ToTensor()(exc_prototype)
exc_prototype = norm_trans(exc_prototype)

evolve_mode = 'inh'
steps = 250
batch_size = 32
g_name = "fc6"

G = load_GAN(g_name)

model_name = 'alexnet'
source = 'torchvision'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
neuron_coord = None
conv_layer = True
if conv_layer:
    neuron_coord = 6
module_name = 'features.11'
unit_id = 25

extractor = get_extractor(
    model_name=model_name,
    source=source,
    device=device,
    pretrained=True
)

init_code = np.random.randn(1, 4096)
new_codes = init_code

optimizer = CholeskyCMAES(
    4096, population_size=40, init_sigma=2.0,
    Aupdate_freq=10, init_code=np.zeros([1, 4096]),
    maximize=(evolve_mode == 'exc')
)

best_imgs = []
for i in range(steps):
    latent_code = torch.from_numpy(np.array(new_codes)).float()

    imgs = G.visualize(latent_code.cuda()).cpu()
    imgs = imnet_trans(imgs)

    mixed_img = (0.75 * exc_prototype) + (0.25 * imgs)
    # mixed_img = (exc_prototype + imgs) / 2
    mixed_img = torch.unsqueeze(mixed_img, 0)

    features = extractor.extract_features(
        batches=mixed_img,
        module_name=module_name,
        flatten_acts=False
    )

    scores = None
    if neuron_coord is not None:
        scores = features[:, unit_id, neuron_coord, neuron_coord]
    else:
        scores = features[:, unit_id]

    print(scores.min())
    new_codes = optimizer.step_simple(scores, new_codes)

    mixed_img = inv_trans(mixed_img[0])
    best_imgs.append(mixed_img[scores.argmin(),:,:,:])

#   Save best imgs
mtg_exp = transforms.ToPILImage()(make_grid(best_imgs, nrow=10))
mtg_exp.save(join(savedir, f"mixed_250_75-25_{evolve_mode}_besteachgen_thingsvision_module_{module_name}_unit_{unit_id}.jpg"))