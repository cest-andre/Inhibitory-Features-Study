import sys
import copy
import argparse
import torch
import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt
import torchvision
from torchvision import models, transforms
from modify_weights import clamp_ablate_unit, random_ablate_unit, binarize_unit
sys.path.append(r'/home/andre/evolve-code/circuit_toolkit')
# sys.path.append(r'/home/andre/evolve-code/selectivity_codes')
from circuit_toolkit.dataset_utils import create_imagenet_valid_dataset
from circuit_toolkit.selectivity import batch_selectivity
from circuit_toolkit.insilico_rf_save import get_center_pos_and_rf
from transplant import get_activations


def validate_tuning_curve_thingsvision(extractor, module_name, selected_neuron=None, neuron_coord=None, subset=None, sort_acts=True):
    batch_size = 1024
    IMAGE_SIZE = 224
    # specify ImageNet mean and standard deviation
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
    ])
    norm_transform = transforms.Normalize(mean=MEAN, std=STD)

    # create a dataset object for the ImageNet dataset
    imnet_folder = r"/home/andrelongon/Documents/data/imagenet/val"
    imagenet_data = torchvision.datasets.ImageFolder(imnet_folder, transform=transform)
    #   Shuffle due to taking subset.
    torch.manual_seed(0)
    dataloader = torch.utils.data.DataLoader(imagenet_data, batch_size=batch_size, shuffle=False, drop_last=False)

    num_batches = len(dataloader)
    if subset is not None:
        num_batches = int(subset * len(dataloader))
    print(f"Running on {num_batches} batches.")
    input_acts = []
    activations = []
    most_images = []
    all_images = []
    act_list = []
    im_count = 0

    for j, (inputs, labels) in enumerate(dataloader):
        print(j)
        if j == num_batches:
            break

        inputs, labels = inputs.cuda(), labels.cuda()
        im_count += inputs.shape[0]

        norm_inputs = norm_transform(inputs)
        acts = get_activations(extractor, norm_inputs, module_name, neuron_coord, selected_neuron, use_center=True)
        activations.append(acts)
        # print(np.min(acts))
        # if extractor is not None:
        #     input_acts.append(get_activations(extractor, norm_inputs, f'features.{selected_layer-2}'))

        inputs = inputs.cpu()
        for input in inputs:
            all_images.append(input)

    all_ord_list = np.arange(im_count).tolist()
    unrolled_act = [num for sublist in activations for num in sublist]
    if sort_acts:
        all_act_list, all_ord_sorted = zip(*sorted(zip(unrolled_act, all_ord_list), reverse=True))
        return all_images, act_list, unrolled_act, all_act_list, all_ord_sorted, input_acts
    else:
        return all_images, act_list, unrolled_act, None, None, input_acts


def validate_tuning_curve(model, layer, selected_layer, selected_neuron, subset=None):
    batch_size = 256
    IMAGE_SIZE = 224
    linear = False
    patch = False
    # specify ImageNet mean and standard deviation
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    # ReceptiveField computation from Binxu code
    model = model.cuda()
    model.eval()
    cent_pos, corner, imgsize, Xlim, Ylim, gradAmpmap = (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), np.random.randn(
        IMAGE_SIZE, IMAGE_SIZE)
    if not linear:
        cent_pos, corner, imgsize, Xlim, Ylim, gradAmpmap = get_center_pos_and_rf(model,
                                                                                  layer, input_size=(3, 224, 224),
                                                                                  device="cuda")
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    norm_transform = transforms.Normalize(mean=MEAN, std=STD)

    # create a dataset object for the ImageNet dataset
    imnet_folder = r"/home/andre/data/imagenet/val/valid"
    imagenet_data = torchvision.datasets.ImageFolder(imnet_folder, transform=transform)
    #   Shuffle due to taking subset.
    torch.manual_seed(0)
    dataloader = torch.utils.data.DataLoader(imagenet_data, batch_size=batch_size, shuffle=False, drop_last=False)

    num_batches = len(dataloader)
    if subset is not None:
        num_batches = int(subset * len(dataloader))
    print(f"Running on {num_batches} batches.")
    activations = []
    most_images = []
    all_images = []
    act_list = []
    im_count = 0

    for j, (inputs, labels) in enumerate(dataloader):
        if j == num_batches:
            break

        inputs, labels = inputs.cuda(), labels.cuda()
        im_count += inputs.shape[0]

        norm_inputs = norm_transform(inputs)

        batch_activations, most_act_image, most_supp_img, best_act = batch_selectivity(corner, gradAmpmap,
                                                                                       imgsize, norm_inputs, labels,
                                                                                       model=model, layer=layer,
                                                                                       dataloader=dataloader,
                                                                                       batch_size=batch_size,
                                                                                       selected_layer=selected_layer,
                                                                                       selected_neuron=selected_neuron,
                                                                                       linear=linear, patch=patch)

        activations.append(batch_activations)
        # most_images.append(most_act_image)
        # most_images.append(most_supp_img)
        act_list.append(best_act[0])
        act_list.append(best_act[1])

        norm_inputs = norm_inputs.cpu()
        for input in norm_inputs:
            all_images.append(input)

    # ord_list = np.arange(len(act_list)).tolist()
    # act_list, ord_list = zip(*sorted(zip(act_list, ord_list), reverse=True))

    all_ord_list = np.arange(im_count).tolist()
    unrolled_act = [num for sublist in activations for num in sublist]
    all_act_list, all_ord_sorted = zip(*sorted(zip(unrolled_act, all_ord_list), reverse=True))
    # print(f"Imnet order for top act image:  {all_ord_sorted[0]}\nActivation:  {unrolled_act[all_ord_sorted[0]]}")
    # print(f"Act:  {all_act_list[0]}")

    #   TODO
    #   all_ord_sorted[0] indexes the top activating image.  So I think I just need to index the ablated unrolled_act
    #   with all_ord_sorted index from intact ([0:9]).
    #
    #   I can turn this into a function which will return these lists.  Then elsewhere, I can run this for intact and
    #   ablate, then retrieve their respective lists.  To plot, I can remove the top 9 intacts from the ablate list
    #   and make a separate plt.scatter call with just these points in a different color.
    #
    #   For rank order, I think I just run this on the respective all_ord_sorteds.  Or maybe the activations.
    return all_images, act_list, unrolled_act, all_act_list, all_ord_sorted, gradAmpmap

def validate(model):
    # specify ImageNet mean and standard deviation
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    torch.manual_seed(0)

    model = model.cuda()
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    norm_transformation = transforms.Normalize(mean=MEAN, std=STD)

    # create a dataset object for the ImageNet dataset
    imnet_folder = r"/home/andrelongon/Documents/data/imagenet/val"
    imagenet_data = torchvision.datasets.ImageFolder(imnet_folder, transform=transform)
    dataloader = torch.utils.data.DataLoader(imagenet_data, batch_size=batch_size, shuffle=False)
    # print(len(dataloader))

    correct = 0
    total = 0

    for j, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.cuda(), labels.cuda()

        norm_inputs = norm_transformation(inputs)
        outputs = model(norm_inputs)

        # # Compute the top-5 predictions
        _, predicted = torch.topk(outputs, k=1, dim=1)

        # # Check if the true label is in the top-5 predictions
        correct += (predicted == labels.unsqueeze(1)).any(dim=1).sum().item()
        total += labels.size(0)

    # Compute the top-5 accuracy
    accuracy = 100 * correct / total
    print('Top 5 accuracy ', accuracy)

    return accuracy


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str)
    args = parser.parse_args()

    batch_size = 1024
    IMAGE_SIZE = 224

    model = None
    if args.network == 'alexnet':
        model = models.alexnet(True)
    elif args.network == 'vgg16':
        model = models.vgg16(True)
    elif args.network == 'resnet18':
        model = models.resnet18(True)

    # model.load_state_dict(ablate_unit(model.state_dict(), "features.10.weight", np.random.randint(0, 256, 64), min=0, max=0))
    # model.load_state_dict(ablate_unit(model.state_dict(), "features.10.weight", np.arange(32), min=0, max=0))
    # validate(model)
    # exit()

    all_ablate_accs = []
    inh_ablate_accs = []
    random_ablate_accs = []
    increased_bias_accs = []
    units = np.arange(32)
    original_states = copy.deepcopy(model.state_dict())

    for i in units:
        model.load_state_dict(clamp_ablate_unit(model.state_dict(), "layer4.1.conv2.weight", i, min=0, max=0))
        all_ablate_accs.append(validate(model))
        model.load_state_dict(original_states)
        
        model.load_state_dict(clamp_ablate_unit(model.state_dict(), "layer4.1.conv2.weight", i, min=0, max=None))
        inh_ablate_accs.append(validate(model))
        model.load_state_dict(original_states)

        # model.load_state_dict(random_ablate_unit(model.state_dict(), "features.10.weight", i, bias_name='features.10.bias'))
        # random_ablate_accs.append(validate(model))
        # model.load_state_dict(original_states)

        #   Increase bias to effectively shift up tuning curve without disturbing rank order.
        # model.load_state_dict(
        #     clamp_ablate_unit(model.state_dict(), "layer4.1.conv2.weight", i, min=0, max=None, bias_name='features.10.bias')
        # )
        # model.state_dict()['features.10.bias'][i] = -10
        # increased_bias_accs.append(validate(model, batch_size))
        # model.load_state_dict(original_states)

        # model.load_state_dict(binarize_unit(model.state_dict(), "features.10.weight", i))
        # increased_bias_accs.append(validate(model))
        # model.load_state_dict(original_states)

    # plt.scatter(units, np.array(all_ablate_accs),  color='g')
    # plt.scatter(units, np.array(random_ablate_accs), color='r')
    # plt.scatter(units, np.array(inh_ablate_accs), color='b')
    # plt.savefig(f'/home/andre/imagenet_val_data/{args.network}_features.10.weight_ablate_imnet_val_accuracy.png')
    # plt.close()

    np.save(
        f'/home/andrelongon/Documents/inhibition_code/imagenet_val_data/{args.network}_layer4.1.conv2.weight_all_ablate_imnet_val_accuracy.npy',
        np.array(all_ablate_accs)
    )
    print(np.mean(np.array(all_ablate_accs)))
    # np.save(
    #     f'/home/andre/imagenet_val_data/{args.network}_features.10.weight_random_ablate_imnet_val_accuracy.npy',
    #     np.array(random_ablate_accs)
    # )
    np.save(
        f'/home/andrelongon/Documents/inhibition_code/imagenet_val_data/{args.network}_layer4.1.conv2.weight_inh_ablate_imnet_val_accuracy.npy',
        np.array(inh_ablate_accs)
    )
    print(np.mean(np.array(inh_ablate_accs)))
    # np.save(
    #     f'/home/andre/imagenet_val_data/{args.network}_features.10.increased_bias_imnet_val_accuracy.npy',
    #     np.array(increased_bias_accs)
    # )