import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from thingsvision import get_extractor, get_extractor_from_model
import functools
import sys
import argparse
import lpips
import numpy as np
# from brainscore_core import score_model

from transplant import get_activations
from modify_weights import rescale_weights
from train_imnet import AlexNet

sys.path.append(r'/home/andrelongon/Documents/inhibition_code/vision')
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images
from brainscore_vision.model_helpers.activations.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from brainscore_vision import score
from brainscore_vision import model_registry


parser = argparse.ArgumentParser()
parser.add_argument('--network', type=str)
parser.add_argument('--layer_name', type=str)
args = parser.parse_args()

model_params = None#{'weights': 'IMAGENET1K_V1'}# if model_name == 'resnet152' or model_name == 'resnet50' else None

source_name = 'custom' if args.network == 'cornet-s' else 'torchvision'
extractor = get_extractor(
    model_name=args.network,
    source=source_name,
    device='cuda',
    pretrained=False,
    model_parameters=model_params
)


#   Create my alexnet
# model = AlexNet().cuda()

states = torch.load(f"/media/andrelongon/DATA/imnet_weights/overlap_finetune/{args.network}_untrained.pth")
# states = torch.load(f"/media/andrelongon/DATA/imnet_weights/overlap_finetune/resnet18_layer4.1.conv2_baseline_weightdecay_1ep.pth")
extractor.model.load_state_dict(states)
del states

# extractor = get_extractor_from_model(model=model, device='cuda', backend='pt')
#   NOTE:  untrained weights to later train without overlap and compare brainscores.
# torch.save(extractor.model.state_dict(), f"/media/andrelongon/DATA/imnet_weights/overlap_finetune/{args.network}_untrained.pth")

# states = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x8d_wsl').state_dict()
# model_label = '14'
# states = torch.load(f"/home/andrelongon/Documents/inhibition_code/weights/overlap_finetune/{args.network}_{args.layer_name}_{model_label}.pth")

#pre_finetune
#input_layer_tuned
# preprocessing = functools.partial(load_preprocess_images, image_size=224)
# activations_model = PytorchWrapper(identifier=f'{args.network}-{args.layer_name}-pre_finetune', model=extractor.model, preprocessing=preprocessing)
# activations_model.image_size = 224

# model_registry[f'{args.network}-{args.layer_name}-pre_finetune'] = lambda: ModelCommitment(identifier=f'{args.network}-{args.layer_name}-pre_finetune', activations_model=activations_model, layers=[args.layer_name])
# it_score = score(model_identifier=f'{args.network}-{args.layer_name}-pre_finetune', benchmark_identifier='MajajHong2015public.IT-pls')#, model=model)
# print(f'Score pre-finetune:  {it_score}')

#  Insert overlapped weights for each conv layer.
# model_states = extractor.model.state_dict()
# states = torch.clone(torch.load(f"/home/andrelongon/Documents/inhibition_code/weights/overlap_finetune/cornet-s_IT.conv_input_7_3ep.pth")['IT.conv_input.weight'])
# model_states['IT.conv_input.weight'] = states#['IT.conv_input.weight']

# states = torch.clone(torch.load(f"/home/andrelongon/Documents/inhibition_code/weights/overlap_finetune/cornet-s_IT.conv1_2_1ep.pth")['IT.conv1.weight'])
# model_states['IT.conv1.weight'] = states#['IT.conv1.weight']

# states = torch.clone(torch.load(f"/home/andrelongon/Documents/inhibition_code/weights/overlap_finetune/cornet-s_IT.conv2_2_1ep.pth")['IT.conv2.weight'])
# model_states['IT.conv2.weight'] = states#['IT.conv2.weight']

# states = torch.clone(torch.load(f"/home/andrelongon/Documents/inhibition_code/weights/overlap_finetune/cornet-s_IT.conv3_1_3ep.pth")['IT.conv3.weight'])
# model_states['IT.conv3.weight'] = states#['IT.conv3.weight']

# extractor.model.load_state_dict(model_states)

# preprocessing = functools.partial(load_preprocess_images, image_size=224)
# activations_model = PytorchWrapper(identifier=f'{args.network}-{args.layer_name}-conv_input_7_conv1_2_conv2_2', model=extractor.model, preprocessing=preprocessing)
# activations_model.image_size = 224

# model_registry[f'{args.network}-{args.layer_name}-conv_input_7_conv1_2_conv2_2'] = lambda: ModelCommitment(identifier=f'{args.network}-{args.layer_name}-conv_input_7_conv1_2_conv2_2', activations_model=activations_model, layers=[args.layer_name])
# it_score = score(model_identifier=f'{args.network}-{args.layer_name}-conv_input_7_conv1_2_conv2_2', benchmark_identifier='MajajHong2015public.IT-pls')#, model=model)
# print(f'Score pre-finetune:  {it_score}')

# exit()

#   Overlap fine tune.
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
imnet_folder = r"/media/andrelongon/DATA/imagenet/train"
imagenet_data = torchvision.datasets.ImageFolder(imnet_folder, transform=transform)
#   Shuffle due to taking subset.
torch.manual_seed(0)
dataloader = torch.utils.data.DataLoader(imagenet_data, batch_size=batch_size, shuffle=True, drop_last=False)

supervised_train = True
val = True
criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.AdamW(extractor.model.parameters())

# states = torch.load("/media/andrelongon/DATA/imnet_opt_states/overlap_finetune/resnet18_layer4.1.conv2_baseline_weightdecay_1ep.pth")
# optimizer.load_state_dict(states)
# del states

# num_batches = 256
# if subset is not None:
#     num_batches = int(subset * len(dataloader))

# print(f"Running on {num_batches} batches.")

#   Get weights for layer to fine-tune.
# states = extractor.model.state_dict()
# layer_weights = torch.clone(states[args.layer_name + '.weight'])
# weights_copy = torch.clone(layer_weights)
#   Make copy of weights?


#   Get layer acts for each batch.  For each neuron, get lpips for top and bot imgs in batch.
#   Then, tweak the top 9 pos/neg channel weights (double them?).  Repass images through layer, recompute
#   top bot lpips.  If lpips decreases from earlier one, tweak original weights by a smaller increase (10%).
#   If lpips increases, decrease original weights (10%).

#   TODO:  Is a double pass each epoch necessary?  Maybe just for the first batch, but after, can just
#          keep a single distance array and update it each batch.

# states = torch.load(f"/home/andrelongon/Documents/inhibition_code/weights/overlap_finetune/myalexnet_features.16_1_1ep.pth")
# states = torch.load("/media/andrelongon/DATA/imnet_weights/overlap_finetune/myalexnet_features.16_8_1ep.pth")
# extractor.model.load_state_dict(states)

metric = lpips.LPIPS(net='alex').cuda()

epochs = 3
top_ch = 3
distances = []
strikes = 0
offset = None

for e in range(0, epochs):
    for j, (inputs, labels) in enumerate(dataloader):
        inputs = inputs.cuda()
        labels = labels.cuda()

        norm_inputs = norm_transform(inputs)

        if supervised_train:
            optimizer.zero_grad()

            # old_w = torch.clone(extractor.model.state_dict()[args.layer_name + '.weight'])

            outputs = extractor.model(norm_inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # continue

            # new_w = torch.clone(extractor.model.state_dict()[args.layer_name + '.weight'])
            # offset = torch.mean(torch.abs(old_w - new_w)) / 10

        states = extractor.model.state_dict()
        layer_weights = torch.clone(states[args.layer_name + '.weight'])
        weights_copy = torch.clone(layer_weights)
        
        print(j)
        # if j == num_batches:
        #     break

        acts = get_activations(extractor, norm_inputs, args.layer_name, use_center=True)
        acts = torch.transpose(torch.tensor(acts).cpu(), 0, 1)

        img_idx = torch.argsort(acts)

        top_ch = np.random.randint(3, high=33)
        
        if True:
        # if len(distances) == 0:
            distances = []
            for n in range(img_idx.shape[0]):
                top_imgs = norm_inputs[torch.flip(img_idx[n][-9:], (0,))]
                bot_imgs = norm_inputs[img_idx[n][:9]]

                dist = metric(top_imgs, bot_imgs)
                distances.append(torch.mean(dist).cpu().detach().numpy())

                channel_idx = torch.argsort(torch.sum(layer_weights[n], (1, 2)))

                layer_weights[n, channel_idx[:top_ch]] *= 2
                layer_weights[n, channel_idx[-top_ch:]] *= 2

                # layer_weights[n, channel_idx[:top_ch]] -= 0.01
                # layer_weights[n, channel_idx[-top_ch:]] += 0.01

            states[args.layer_name + '.weight'] = layer_weights
            extractor.model.load_state_dict(states)

            acts = get_activations(extractor, norm_inputs, args.layer_name, use_center=True)
            acts = torch.transpose(torch.tensor(acts).cpu(), 0, 1)

            img_idx = torch.argsort(acts)

            states[args.layer_name + '.weight'] = weights_copy
            extractor.model.load_state_dict(states)
            layer_weights = torch.clone(states[args.layer_name + '.weight'])

        increase_count = 0

        for n in range(img_idx.shape[0]):
            top_imgs = norm_inputs[torch.flip(img_idx[n][-9:], (0,))]
            bot_imgs = norm_inputs[img_idx[n][:9]]

            dist = metric(top_imgs, bot_imgs)
            dist = torch.mean(dist).cpu().detach().numpy()

            channel_idx = torch.argsort(torch.sum(layer_weights[n], (1, 2)))

            if dist > distances[n]:
                layer_weights[n, channel_idx[:top_ch]] *= 1.001 #(1 + offset)
                layer_weights[n, channel_idx[-top_ch:]] *= 1.001 #(1 + offset)

                # layer_weights[n, channel_idx[:top_ch]] -= 0.001
                # layer_weights[n, channel_idx[-top_ch:]] += 0.001

                distances[n] = dist

                increase_count += 1
            else:
                pass
                # layer_weights[n, channel_idx[:top_ch]] *= 0.999 #(1 - offset)
                # layer_weights[n, channel_idx[-top_ch:]] *= 0.999 #(1 - offset)

                # layer_weights[n, channel_idx[:top_ch]] += 0.001
                # layer_weights[n, channel_idx[-top_ch:]] -= 0.001

                #   Also bump up the channels in the middle to encourage new opportunities?
                #   Likewise decrease middle channels in above condition?

        print(f"Increase count:  {increase_count}")
        # if j == 300:
        #     break
        # if increase_count < 5:
        #     strikes += 1

        #     if strikes == 10:
        #         break

        states[args.layer_name + '.weight'] = layer_weights
        extractor.model.load_state_dict(states)

    model_label = f'3_inverse_{e+1}ep'
    torch.save(extractor.model.state_dict(), f"/media/andrelongon/DATA/imnet_weights/overlap_finetune/{args.network}_{args.layer_name}_{model_label}.pth")
    torch.save(optimizer.state_dict(), f"/media/andrelongon/DATA/imnet_opt_states/overlap_finetune/{args.network}_{args.layer_name}_{model_label}.pth")

    if val:
        # states = torch.load("/home/andrelongon/Documents/inhibition_code/weights/overlap_finetune/resnet18_layer4.0.conv1_6_1ep.pth")

        #   TODO:  Rescale based on each neuron's min max weight from pre-finetuned state.
        # states = rescale_weights(states, extractor.model.state_dict(), 'layer4.0.conv1.weight')
        # extractor.model.load_state_dict(states)

        # layer_weights = extractor.model.state_dict()['layer4.0.conv1.weight']
        # layer_weights = torch.flatten(layer_weights, start_dim=1, end_dim=-1)
        # print(torch.min(layer_weights, 0)[0][:10])
        # print(torch.max(layer_weights, 0)[0][:10])
        # exit()

        imnet_val_folder = r"/media/andrelongon/DATA/imagenet/val"
        imnet_val_data = torchvision.datasets.ImageFolder(imnet_val_folder, transform=transform)
        torch.manual_seed(0)
        val_loader = torch.utils.data.DataLoader(imnet_val_data, batch_size=batch_size, shuffle=True, drop_last=False)

        top1_correct = 0
        top5_correct = 0
        total = 0
        with torch.no_grad():
            for data in val_loader:
                images, labels = data
                images, labels = images.cuda(), labels.cuda()
                images = norm_transform(images)
                # calculate outputs by running images through the network
                outputs = extractor.model(images)
                # the class with the highest energy is what we choose as prediction
                _, top1_predicted = torch.topk(outputs, k=1, dim=1)
                _, top5_predicted = torch.topk(outputs, k=5, dim=1)
                
                top1_correct += (top1_predicted == labels.unsqueeze(1)).any(dim=1).sum().item()
                top5_correct += (top5_predicted == labels.unsqueeze(1)).any(dim=1).sum().item()
                total += labels.size(0)

        print(f'Imnet val top 1: {100 * top1_correct // total} %')
        print(f'Imnet val top 5: {100 * top5_correct // total} %')

    # exit()

    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    activations_model = PytorchWrapper(identifier=f'{args.network}-{args.layer_name}-post_finetune_{model_label}', model=extractor.model, preprocessing=preprocessing)
    activations_model.image_size = 224

    model_registry[f'{args.network}-{args.layer_name}-post_finetune_{model_label}'] = lambda: ModelCommitment(identifier=f'{args.network}-{args.layer_name}-post_finetune_{model_label}', activations_model=activations_model, layers=[args.layer_name])
    it_score = score(model_identifier=f'{args.network}-{args.layer_name}-post_finetune_{model_label}', benchmark_identifier='MajajHong2015public.IT-pls')#, model=model)
    print(f'Score post_finetune:  {it_score}')