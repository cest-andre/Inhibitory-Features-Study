import torchvision
from torchvision.io import read_image
from torchvision.utils import make_grid


for i in range(10):
    exc_imgs = []
    inh_imgs = []

    for j in range(9):
        img = read_image(
            f'/home/andre/evolved_data/alexnet_untrained_.features.Conv2d10_unit{i}/exc/no_mask/Best_Img_alexnet_untrained_.features.Conv2d10_{j}.png'
        )
        exc_imgs.append(img)

        img = read_image(
            f'/home/andre/evolved_data/alexnet_untrained_.features.Conv2d10_unit{i}/inh/no_mask/Best_Img_alexnet_untrained_.features.Conv2d10_{j}.png'
        )
        inh_imgs.append(img)

    grid = make_grid(exc_imgs, nrow=3)
    grid = torchvision.transforms.ToPILImage()(grid)
    grid.save(f'/home/andre/evolve_grid/alexnet_untrained/unit{i}_all_Exc_Prototype_alexnet_untrained_.features.Conv2d10.png')

    grid = make_grid(inh_imgs, nrow=3)
    grid = torchvision.transforms.ToPILImage()(grid)
    grid.save(f'/home/andre/evolve_grid/alexnet_untrained/unit{i}_all_Inh_Prototype_alexnet_untrained_.features.Conv2d10.png')