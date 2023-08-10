from os.path import join
from thingsvision import get_extractor
from evolve_utils import load_GAN
from torchvision import transforms
from torchvision.utils import make_grid
import numpy as np
import torch

from circuit_toolkit.Optimizers import CholeskyCMAES


savedir = r"/home/andre/evolved_data/thingsvision"

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

evolve_mode = 'exc'
steps = 100
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
module_name = 'features.10'
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
    imgs = torch.unsqueeze(imgs, 0)

    features = extractor.extract_features(
        batches=imgs,
        module_name=module_name,
        flatten_acts=False
    )

    scores = None
    if neuron_coord is not None:
        scores = features[:, unit_id, neuron_coord, neuron_coord]
    else:
        scores = features[:, unit_id]

    new_codes = optimizer.step_simple(scores, new_codes)

    imgs = inv_trans(imgs[0])
    idx = scores.argmax() if model_name == 'exc' else scores.argmin()
    best_imgs.append(imgs[idx,:,:,:])

#   Save best imgs
mtg_exp = transforms.ToPILImage()(make_grid(best_imgs, nrow=10))
mtg_exp.save(join(savedir, f"{evolve_mode}_besteachgen_thingsvision_module_{module_name}_unit_{unit_id}.jpg"))