import sys
import os
import argparse
import numpy as np
from PIL import Image
from time import time
import torch
from torchvision import models, transforms

# os.chdir(r'/home/andre/evolve-code')
# print(os.getcwd())
# exit()
sys.path.append(r'/home/andre/evolve-code')
sys.path.append(r'/home/andre/evolve-code/circuit_toolkit')
# print(sys.path)
# exit()

from ActMax_Optimizer_Dev.core.insilico_exps import ExperimentEvolution
# from ActMax_Optimizer_Dev.core.Optimizers import CholeskyCMAES, ZOHA_Sphere_lr_euclid, Genetic, pycma_optimizer
from circuit_toolkit.Optimizers import CholeskyCMAES
from selectivity_codes.insilico_RF_save import get_center_pos_and_rf
from selectivity_codes.utils import normalize

from modify_weights import clamp_ablate_unit


# un-comment to use our new one! 
# optim = ZOHA_Sphere_lr_euclid(4096, population_size=40, select_size=20, lr=1.5, sphere_norm=300)
# optim.lr_schedule(n_gen=75, mode="exp", lim=(50, 7.33) ,)

parser = argparse.ArgumentParser()
parser.add_argument('--network', type=str)
parser.add_argument('--neuron', type=int)
parser.add_argument('--layer_name', type=str)
parser.add_argument('--neuron_coord', type=int)
parser.add_argument('--type', type=str, required=False)
parser.add_argument('--ablate', action='store_true', default=False)
parser.add_argument('--weight_name', type=str, required=False)
args = parser.parse_args()

type_label = ''
states = None
if args.type == 'alexnet-untrained':
    type_label = 'untrained_'
    states = torch.load("/home/andre/tuning_curves/untrained_alexnet/random_weights.pth")
elif args.type == 'resnet18-untrained':
    type_label = 'untrained_'
    states = torch.load("/home/andre/tuning_curves/untrained_resnet18/random_weights.pth")
elif args.type == 'resnet18-robust':
    type_label = 'robust_'
    states = torch.load("/home/andre/model_weights/resnet-18-l2-eps3.pt")

explabel = f"new_{args.network}_{type_label}{args.layer_name}"
model_unit = (args.network, args.layer_name, args.neuron, args.neuron_coord, args.neuron_coord)

savedir = None
if args.ablate:
    savedir = r"/home/andre/evolved_data_ablated"
else:
    savedir = r"/home/andre/evolved_data"

savedir = os.path.join(savedir, explabel + f"_unit{model_unit[2]}")
if not os.path.isdir(savedir):
    os.mkdir(savedir)
    os.mkdir(os.path.join(savedir, "exc"))
    os.mkdir(os.path.join(savedir, "exc", "no_mask"))
    os.mkdir(os.path.join(savedir, "inh"))
    os.mkdir(os.path.join(savedir, "inh", "no_mask"))
else:
    exit()

#   Center crop prototypes to 224 (set these to 224).
avg_exc_prototype = np.zeros((224, 224, 3))
avg_inh_prototype = np.zeros((224, 224, 3))

cent_pos, corner, imgsize, Xlim, Ylim, gradAmpmap = (0,0), (0,0), (0,0), (0,0), (0,0), np.random.randn(224,224)

runs = 9
for i in range(runs):
    print(f"-----     Run {i}     -----")
    optim = CholeskyCMAES(
        4096, population_size=40, init_sigma=2.0,
        Aupdate_freq=10, init_code=np.zeros([1, 4096]),
        maximize=True)
    Exp = ExperimentEvolution(model_unit, savedir=os.path.join(savedir, "exc"), explabel=explabel, optimizer=optim)

    if states is not None:
        Exp.CNNmodel.model.load_state_dict(states)

    if args.ablate:
        Exp.CNNmodel.model.load_state_dict(
            clamp_ablate_unit(Exp.CNNmodel.model.state_dict(), args.weight_name, args.neuron, min=0, max=None)
        )

    if i == 0:
        cent_pos, corner, imgsize, Xlim, Ylim, gradAmpmap = get_center_pos_and_rf(
            Exp.CNNmodel.model, args.layer_name, input_size=(3, 224, 224), device="cuda"
        )
        gradAmpmap = np.expand_dims(normalize(gradAmpmap), -1)

    t1 = time()
    Exp.run(optim.get_init_pop())
    t2 = time()
    print(t2 - t1, "sec")
    # Exp.visualize_trajectory(show=False)

    avg_exc_prototype += Exp.save_prototype(mask=gradAmpmap, evo_run=i)
    Exp.save_last_gen(evo_run=i)

    del(optim)
    del(Exp)

    optim = CholeskyCMAES(
        4096, population_size=40, init_sigma=2.0,
        Aupdate_freq=10, init_code=np.zeros([1, 4096]),
        maximize=False)
    Exp = ExperimentEvolution(model_unit, savedir=os.path.join(savedir, "inh"), explabel=explabel, optimizer=optim)

    if states is not None:
        Exp.CNNmodel.model.load_state_dict(states)

    if args.ablate:
        Exp.CNNmodel.model.load_state_dict(
            clamp_ablate_unit(Exp.CNNmodel.model.state_dict(), args.weight_name, args.neuron, min=None, max=0)
        )

    # if i == 0:
    #     cent_pos, corner, imgsize, Xlim, Ylim, gradAmpmap = get_center_pos_and_rf(
    #         Exp.CNNmodel.model, args.layer_name, input_size=(3, 224, 224), device="cuda"
    #     )
    #     gradAmpmap = np.expand_dims(normalize(gradAmpmap), -1)

    t1 = time()
    Exp.run(optim.get_init_pop())
    t2 = time()
    print(t2 - t1, "sec")
    # Exp.visualize_trajectory(show=False)

    avg_inh_prototype += Exp.save_prototype(mask=gradAmpmap, evo_run=i, inhib=True)
    Exp.save_last_gen(evo_run=i)

    del(optim)
    del(Exp)

avg_exc_prototype = (255 * avg_exc_prototype / runs)
avg_inh_prototype = (255 * avg_inh_prototype / runs)

avg_exc_pil = Image.fromarray(avg_exc_prototype.astype(np.uint8))
avg_exc_pil.save(os.path.join(savedir, f"Avg_Exc_Prototype_{explabel}.png"))
avg_exc_pil = Image.fromarray((avg_exc_prototype * gradAmpmap).astype(np.uint8))
avg_exc_pil.save(os.path.join(savedir, f"Avg_Exc_Prototype_masked_{explabel}.png"))

avg_inh_pil = Image.fromarray(avg_inh_prototype.astype(np.uint8))
avg_inh_pil.save(os.path.join(savedir, f"Avg_Inh_Prototype_{explabel}.png"))
avg_inh_pil = Image.fromarray((avg_inh_prototype * gradAmpmap).astype(np.uint8))
avg_inh_pil.save(os.path.join(savedir, f"Avg_Inh_Prototype_masked_{explabel}.png"))
