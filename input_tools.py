import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from thingsvision import get_extractor
from PIL import Image

from imnet_val import validate_tuning_curve_thingsvision
from tuning_curve import get_max_stim_acts

#   TODO:  Extract entire post-relu input tensor for a given layer.
#          For instance, features.2 for second conv layer of AlexNet.
#          Compute histogram of this tensor with zero as a bucket (b/t -1e-16 to 1e-16 or something).
#          Other buckets are all positive:  1e-16 to 0.01, 0.01 to 0.1, 0.1 to 1, 1 to 10, 10 to 100?
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str)
    parser.add_argument('--layer_name', type=str)
    parser.add_argument('--weight_name', type=str)
    parser.add_argument('--layer_number', type=int)
    parser.add_argument('--neuron', type=int)
    # parser.add_argument('--neuron_coord', type=int)
    parser.add_argument('--type', type=str, required=False)
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

    extractor = get_extractor(
        model_name=args.network,
        source='torchvision',
        device='cpu',
        pretrained=True
    )

    if states is not None:
        extractor.model.load_state_dict(states)

    weights = extractor.model.state_dict()[args.weight_name][args.neuron]
    idx_offset = weights.shape[-1] // 2
    weights = torch.flatten(weights)

    #   TODO:  Move this code into function.  Then create a new function with computes positive-weighted
    #          input norms of intact vs ablated optimized images.  Save these somewhere and pair plot later.
    #          First, obtain prototypes and lucent opt images from 32 ABLATED (negative) units across layers
    #          in trained and untrained AlexNet and ResNet18.

    savedir = "/home/andre/max_stim_inputs/intact_v_ablate/resnet18_trained"

    # topdir = f'/home/andre/tuning_curves/alexnet/intact/layer{args.layer_number}_neuron{args.neuron}/max/exc'
    # botdir = f'/home/andre/tuning_curves/alexnet/intact/layer{args.layer_number}_neuron{args.neuron}/max/inh'
    protodir = f'/home/andre/evolved_data/resnet18_.layer{args.layer_number}.1.Conv2dconv2_unit{args.neuron}/exc/no_mask'
    # antiprotodir = f'/home/andre/evolved_data/alexnet_untrained_.features.Conv2d{args.layer_number - 1}_unit{args.neuron}/inh/no_mask'
    # poslucentdir = f"/home/andre/lucent_imgs/alexnet_trained/features.{args.layer_number-1}_unit{args.neuron}/pos"
    # poslucentdir = f"/home/andre/lucent_imgs/alexnet_trained/{args.neuron}/pos"
    # neglucentdir = f"/home/andre/lucent_imgs/alexnet_trained/{args.neuron}/neg"

    abl_protodir = f'/home/andre/evolved_data_ablated/resnet18_.layer{args.layer_number}.1.Conv2dconv2_unit{args.neuron}/exc/no_mask'
    # abl_poslucentdir = f"/home/andre/lucent_imgs_ablated/alexnet_trained/features.{args.layer_number-1}_unit{args.neuron}/pos"

    # top_imgs = [Image.open(os.path.join(topdir, file)) for file in os.listdir(topdir)]
    # bot_imgs = [Image.open(os.path.join(botdir, file)) for file in os.listdir(botdir)]
    # print(os.listdir(protodir))
    proto_imgs = [Image.open(os.path.join(protodir, file)) for file in os.listdir(protodir)]
    # antiproto_imgs = [Image.open(os.path.join(antiprotodir, file)) for file in os.listdir(antiprotodir)]
    # poslucent_imgs = [Image.open(os.path.join(poslucentdir, file)) for file in os.listdir(poslucentdir)]
    # neglucent_imgs = [Image.open(os.path.join(neglucentdir, file)) for file in os.listdir(neglucentdir)]
    # print(os.listdir(abl_protodir))
    abl_proto_imgs = [Image.open(os.path.join(abl_protodir, file)) for file in os.listdir(abl_protodir)]
    # abl_poslucent_imgs = [Image.open(os.path.join(abl_poslucentdir, file)) for file in os.listdir(abl_poslucentdir)]

    # _, _, acts, _, _, _ =\
    #     validate_tuning_curve_thingsvision(extractor, args.layer_name, subset=0.1, sort_acts=False)
    # acts = torch.tensor(np.array(acts))#[:, :, 2:5, 2:5]
    # acts = torch.clamp(acts, min=0, max=None)

    #   Calculate middle index and offset based on shape of preflattened weights and acts.
    intact_acts = torch.tensor(get_max_stim_acts(extractor, proto_imgs, args.layer_name))
    ablated_acts = torch.tensor(get_max_stim_acts(extractor, abl_proto_imgs, args.layer_name))

    mid_idx = intact_acts.shape[-1] // 2
    print(f"IDX: {mid_idx-idx_offset}:{mid_idx+idx_offset+1}")
    intact_acts = intact_acts[:, :, :, mid_idx-idx_offset:mid_idx+idx_offset+1, mid_idx-idx_offset:mid_idx+idx_offset+1]
    ablated_acts = ablated_acts[:, :, :, mid_idx-idx_offset:mid_idx+idx_offset+1, mid_idx-idx_offset:mid_idx+idx_offset+1]

    intact_acts = torch.flatten(intact_acts, start_dim=1, end_dim=-1)
    ablated_acts = torch.flatten(ablated_acts, start_dim=1, end_dim=-1)

    intact_acts = torch.clamp(intact_acts, min=0, max=None)
    ablated_acts = torch.clamp(ablated_acts, min=0, max=None)

    # intact_pos_norms = torch.linalg.vector_norm(intact_acts[:, torch.nonzero(weights > 0)], ord=1, dim=1)
    # ablated_pos_norms = torch.linalg.vector_norm(ablated_acts[:, torch.nonzero(weights > 0)], ord=1, dim=1)

    # np.save(
    #     os.path.join(savedir, f"layer{args.layer_number}_unit{args.neuron}_intact_poslucent_norms.npy"),
    #     intact_pos_norms.numpy()
    # )
    #
    # np.save(
    #     os.path.join(savedir, f"layer{args.layer_number}_unit{args.neuron}_ablated_poslucent_norms.npy"),
    #     ablated_pos_norms.numpy()
    # )

    intact_outputs = [torch.dot(act, weights) for act in intact_acts]
    ablated_outputs = [torch.dot(act, weights) for act in ablated_acts]
    print(f"Intact acts of intact protos: {np.mean(np.array(intact_outputs))}")
    print(f"Intact acts of ablated protos: {np.mean(np.array(ablated_outputs))}")

    np.save(
        os.path.join(savedir, f"layer{args.layer_number}_unit{args.neuron}_intact_proto_intact_act.npy"),
        np.array(intact_outputs)
    )

    np.save(
        os.path.join(savedir, f"layer{args.layer_number}_unit{args.neuron}_ablated_proto_intact_act.npy"),
        np.array(ablated_outputs)
    )

    intact_outputs = [torch.dot(act, torch.clamp(weights, min=0, max=None)) for act in intact_acts]
    ablated_outputs = [torch.dot(act, torch.clamp(weights, min=0, max=None)) for act in ablated_acts]
    print(f"Ablated acts of intact protos: {np.mean(np.array(intact_outputs))}")
    print(f"Ablated acts of ablated protos: {np.mean(np.array(ablated_outputs))}")

    np.save(
        os.path.join(savedir, f"layer{args.layer_number}_unit{args.neuron}_intact_proto_ablated_act.npy"),
        np.array(intact_outputs)
    )

    np.save(
        os.path.join(savedir, f"layer{args.layer_number}_unit{args.neuron}_ablated_proto_ablated_act.npy"),
        np.array(ablated_outputs)
    )

    # print(f"Mean act: {np.mean(acts)}")
    #
    # zero_perc = 1 - (torch.nonzero(torch.tensor(acts) > 0).shape[0] / acts.shape[0])
    # print(f"Percent zero inputs: {zero_perc}")
    #
    # bins = [-1e-32, 1e-32, 1e-16, 1e-8, 1e-4, 1e-2, 1e-1, 1, 10, 100, 1000]
    # hist, _, _ = plt.hist(acts, bins)
    # print(hist / np.sum(hist))

    # plt.savefig(f"/home/andre/histograms/input_histos/{args.type}_{args.layer_name}.png")
