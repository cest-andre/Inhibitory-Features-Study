import os
import math
import torchvision
from torchvision.io import read_image
from torchvision.utils import make_grid


def recurrence_grid(basedir, modelname, layername, num_neurons, num_steps):
    exc_imgs = []
    inh_imgs = []
    for n in range(num_neurons):
        for s in range(num_steps):
            img = read_image(os.path.join(basedir, modelname, f"{layername}_{s}", f"unit{n}", "pos", "0_distill_ch.png"))
            exc_imgs.append(img)

            img = read_image(os.path.join(basedir, modelname, f"{layername}_{s}", f"unit{n}", "neg", "0_distill_ch.png"))
            inh_imgs.append(img)

    grid = make_grid(exc_imgs, nrow=num_steps)
    grid = torchvision.transforms.ToPILImage()(grid)
    grid.save(os.path.join(basedir, modelname, f"{layername}_steps_pos_lucents_distill_ch.png"))

    grid = make_grid(inh_imgs, nrow=num_steps)
    grid = torchvision.transforms.ToPILImage()(grid)
    grid.save(os.path.join(basedir, modelname, f"{layername}_steps_neg_lucents_distill_ch.png"))


def lucent_grid(basedir, modelname, layername, num_neurons, has_negs=True):
    exc_imgs = []
    inh_imgs = []
    for n in range(num_neurons):
        img = read_image(os.path.join(basedir, modelname, layername, f"unit{n}", "pos", "0_distill_channel.png"))
        exc_imgs.append(img)

        if has_negs:
            img = read_image(os.path.join(basedir, modelname, layername, f"unit{n}", "neg", "0_distill_channel.png"))
            inh_imgs.append(img)

    nrow = int(math.sqrt(num_neurons))
    grid = make_grid(exc_imgs, nrow=nrow)
    grid = torchvision.transforms.ToPILImage()(grid)
    grid.save(os.path.join(basedir, modelname, layername, f"all_pos_lucents_distill_channel_{nrow}x{nrow}.png"))

    if has_negs:
        grid = make_grid(inh_imgs, nrow=nrow)
        grid = torchvision.transforms.ToPILImage()(grid)
        grid.save(os.path.join(basedir, modelname, layername, f"all_neg_lucents_distill_channel_{nrow}x{nrow}.png"))


def proto_grid(num_neurons):
    for n in range(num_neurons):
        exc_imgs = []
        inh_imgs = []

        for j in range(9):
            img = read_image(
                f'/home/andrelongon/Documents/inhibition_code/evolved_data/alexnet_.features.Conv2d3_unit{n}/exc/no_mask/Best_Img_alexnet_.features.Conv2d3_{j}.png'
            )
            exc_imgs.append(img)

            img = read_image(
                f'/home/andrelongon/Documents/inhibition_code/evolved_data/alexnet_.features.Conv2d3_unit{n}/inh/no_mask/Best_Img_alexnet_.features.Conv2d3_{j}.png'
            )
            inh_imgs.append(img)

        grid = make_grid(exc_imgs, nrow=3)
        grid = torchvision.transforms.ToPILImage()(grid)
        grid.save(f'/home/andrelongon/Documents/inhibition_code/proto_grids/intact/unit{n}_all_Exc_Prototype_alexnet_.features.Conv2d3.png')

        grid = make_grid(inh_imgs, nrow=3)
        grid = torchvision.transforms.ToPILImage()(grid)
        grid.save(f'/home/andrelongon/Documents/inhibition_code/proto_grids/intact/unit{n}_all_Inh_Prototype_alexnet_.features.Conv2d3.png')


if __name__ == '__main__':
    lucent_grid("/media/andrelongon/DATA/feature_viz/intact", "resnet50", "layer4.1.bn3", 9, has_negs=False)
    # recurrence_grid("/media/andrelongon/DATA/feature_viz/intact", "cornet-s", "IT.conv2_id", 36, 2)