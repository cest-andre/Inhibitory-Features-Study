import torchvision
from torchvision.io import read_image
from torchvision.utils import make_grid


neurons = range(8)
# neurons = [0,]

for n in neurons:
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