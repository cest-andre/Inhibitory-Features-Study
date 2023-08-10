import os
from thingsvision import get_extractor
from torchvision import transforms
import copy
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

from modify_weights import clamp_ablate_unit, random_ablate_unit, shuffle_unit


if __name__ == "__main__":
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    norm_trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])

    savedir = '/home/andre/confidence_data'

    model_name = 'alexnet'
    source = 'torchvision'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    neuron_coord = None
    module_name = 'classifier.6'
    unit_id = 25

    extractor = get_extractor(
        model_name=model_name,
        source=source,
        device=device,
        pretrained=True
    )

    original_states = copy.deepcopy(extractor.model.state_dict())

    intact_scores = []
    full_ablate_scores = []
    inh_ablate_scores = []
    random_ablate_scores = []
    shuffled_scores = []
    noise_scores = []

    # full_ablate_diffs = []
    # inh_ablate_diffs = []
    # shuffled_diffs = []
    # noise_diffs = []

    for i in range(1):
        # full_ablate_diff_count = 0
        # inh_ablate_diff_count = 0
        # shuffled_diff_count = 0
        # noise_diff_count = 0

        # unit_id = i
        img_folder = f"/home/andre/tuning_curves/intact/layer11_neuron{unit_id}/max/exc"
        imgs = [Image.open(os.path.join(img_folder, file)) for file in os.listdir(img_folder)]

        for img in imgs:
            img = norm_trans(img)
            img = torch.unsqueeze(img, 0)
            img = torch.unsqueeze(img, 0)

            #   Intact.
            features = extractor.extract_features(
                batches=img,
                module_name=module_name,
                flatten_acts=False
            )
            intact_scores.append(features[0])
            # intact_top5 = torch.topk(torch.Tensor(features), k=5, dim=1)[1].numpy()
            # noise_features = features + np.random.normal(size=features.shape)
            # noise_scores.append(noise_features)
            # noise_top5 = torch.topk(torch.Tensor(noise_features), k=5, dim=1)[1].numpy()

            #   Full ablate.
            extractor.model.load_state_dict(
                clamp_ablate_unit(extractor.model.state_dict(), "features.10.weight", unit_id, min=0, max=0)
            )

            features = extractor.extract_features(
                batches=img,
                module_name=module_name,
                flatten_acts=False
            )
            full_ablate_scores.append(features[0])
            # full_ablate_top5 = torch.topk(torch.Tensor(features), k=5, dim=1)[1].numpy()
            extractor.model.load_state_dict(original_states)

            #   Inhib ablate
            extractor.model.load_state_dict(
                clamp_ablate_unit(extractor.model.state_dict(), "features.10.weight", unit_id, min=0, max=None)
            )

            features = extractor.extract_features(
                batches=img,
                module_name=module_name,
                flatten_acts=False
            )
            inh_ablate_scores.append(features[0])
            # inh_ablate_top5 = torch.topk(torch.Tensor(features), k=5, dim=1)[1].numpy()
            extractor.model.load_state_dict(original_states)

            #   Shuffled
            extractor.model.load_state_dict(
                shuffle_unit(extractor.model.state_dict(), "features.10.weight", unit_id)
            )

            features = extractor.extract_features(
                batches=img,
                module_name=module_name,
                flatten_acts=False
            )
            shuffled_scores.append(features[0])
            # shuffled_top5 = torch.topk(torch.Tensor(features), k=5, dim=1)[1].numpy()
            extractor.model.load_state_dict(original_states)

            #   Random Ablate
            extractor.model.load_state_dict(
                random_ablate_unit(extractor.model.state_dict(), "features.10.weight", unit_id)
            )

            features = extractor.extract_features(
                batches=img,
                module_name=module_name,
                flatten_acts=False
            )
            random_ablate_scores.append(features[0])
            # shuffled_top5 = torch.topk(torch.Tensor(features), k=5, dim=1)[1].numpy()
            extractor.model.load_state_dict(original_states)

            #   TODO:  Calculate diffs and append to lists here.
            #   Perform 1d set diff between intact top 5 and each other condition.
            #   Accumulate diffs across images.  Then append result in the outer loop.
            # full_ablate_diff_count += np.setdiff1d(full_ablate_top5, intact_top5).shape[0]
            # inh_ablate_diff_count += np.setdiff1d(inh_ablate_top5, intact_top5).shape[0]
            # shuffled_diff_count += np.setdiff1d(shuffled_top5, intact_top5).shape[0]
            # noise_diff_count += np.setdiff1d(noise_top5, intact_top5).shape[0]

        #   Append diff result for unit i.
        # full_ablate_diffs.append(full_ablate_diff_count)
        # inh_ablate_diffs.append(inh_ablate_diff_count)
        # shuffled_diffs.append(shuffled_diff_count)
        # noise_diffs.append(noise_diff_count)


    intact_scores = np.mean(np.array(intact_scores), axis=0)
    # noise_scores = np.mean(np.array(noise_scores), axis=0)
    full_ablate_scores = np.mean(np.array(full_ablate_scores), axis=0)
    inh_ablate_scores = np.mean(np.array(inh_ablate_scores), axis=0)
    random_ablate_scores = np.mean(np.array(random_ablate_scores), axis=0)
    shuffled_scores = np.mean(np.array(shuffled_scores), axis=0)

    #   Plot raw activations.
    # plt.scatter(np.arange(1000), intact_scores, color='b', label='Intact')
    # # plt.scatter(np.arange(1000), full_ablate_scores, color='r', label='Full Ablate')
    # # plt.scatter(np.arange(1000), inh_ablate_scores, color='r', label='Inh Ablate')
    # plt.scatter(np.arange(1000), random_ablate_scores, color='r', label='Rand Ablate')
    # plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1.15))
    # plt.xlabel("Unit ID")
    # plt.ylabel("Activation")
    # plt.title(f"Imnet Class Label Output For Unit {unit_id} Top 9")
    # plt.savefig(os.path.join(savedir, f'rand_ablated_confidence_{unit_id}.png'))
    # plt.close()

    #   Plot delta wrt intact activations.
    full_ablate_delta = np.absolute(full_ablate_scores - intact_scores)
    inh_ablate_delta = np.absolute(inh_ablate_scores - intact_scores)
    shuffled_ablate_delta = np.absolute(shuffled_scores - intact_scores)
    random_ablate_delta = np.absolute(random_ablate_scores - intact_scores)

    plt.scatter(full_ablate_delta, inh_ablate_delta, color='b', label='Inh Ablate')
    plt.scatter(full_ablate_delta, random_ablate_delta, color='y', label='Rand Ablate')
    plt.legend(loc='upper left')
    plt.xlabel('Full Ablate Abs Delta')
    plt.ylabel('Alt Ablate Abs Delta')
    min_delta = np.min(full_ablate_delta)
    max_delta = np.max(full_ablate_delta)
    plt.plot(np.arange(min_delta, max_delta), np.arange(min_delta, max_delta), color='r', label='Full Ablate')
    plt.savefig(os.path.join(savedir, f'ablated_confidence_delta_{unit_id}.png'))

    #   Pearson Correlation
    # intact_all = np.array([intact_scores, all_ablate_scores])
    # R = np.corrcoef(intact_all)
    # print(R)
    #
    # intact_inh = np.array([intact_scores, inh_ablate_scores])
    # R = np.corrcoef(intact_inh)
    # print(R)
    #
    # intact_shuffle = np.array([intact_scores, shuffled_scores])
    # R = np.corrcoef(intact_shuffle)
    # print(R)
    #
    # intact_noise = np.array([intact_scores, noise_scores])
    # R = np.corrcoef(intact_noise)
    # print(R)

    # print("Intact")
    # print(intact_top5)
    # print("\nFull Ablate")
    # print(full_ablate_top5)
    # print("\nInh Ablate")
    # print(inh_ablate_top5)
    # print("\nShuffled")
    # print(shuffled_top5)
    # print("\nNoise")
    # print(noise_top5)

    #   Plot diff accumulations.
    # plt.scatter(np.arange(256), full_ablate_diffs, color='r', label='Full Ablate')
    # plt.scatter(np.arange(256), inh_ablate_diffs, color='b', label='Inh Ablate')
    # plt.scatter(np.arange(256), noise_diffs, color='g', label='Noise')
    # plt.scatter(np.arange(256), shuffled_diffs, color='y', label='Shuffled')
    # plt.legend(loc="upper right")
    # plt.xlabel("Channel Number")
    # plt.ylabel("Intact Label Diff")
    # plt.savefig(os.path.join(savedir, f'label_diff.png'))

    #   Plot QQs for normal distribution test.
    # sp.stats.probplot(full_ablate_diffs, dist="norm", plot=plt)
    # plt.savefig(os.path.join(savedir, f'full_ablate_diffs_qq.png'))
    # plt.close()
    # sp.stats.probplot(inh_ablate_diffs, dist="norm", plot=plt)
    # plt.savefig(os.path.join(savedir, f'inh_ablate_diffs_qq.png'))
    # plt.close()
    # sp.stats.probplot(noise_diffs, dist="norm", plot=plt)
    # plt.savefig(os.path.join(savedir, f'noise_diffs_qq.png'))
    # plt.close()
    # sp.stats.probplot(shuffled_diffs, dist="norm", plot=plt)
    # plt.savefig(os.path.join(savedir, f'shuffled_diffs_qq.png'))
    # plt.close()

    #   Ratio of standard devs for homogeneity of variance test.
    # stds = np.array([np.std(full_ablate_diffs), np.std(inh_ablate_diffs), np.std(noise_diffs), np.std(shuffled_diffs)])
    # print(np.max(stds) / np.min(stds))

    #   ANOVA using F-test.
    # print(sp.stats.f_oneway(full_ablate_diffs, inh_ablate_diffs))
    # print(sp.stats.f_oneway(noise_diffs, inh_ablate_diffs))
    # print(sp.stats.f_oneway(shuffled_diffs, inh_ablate_diffs))

    # print(sp.stats.kruskal(full_ablate_diffs, inh_ablate_diffs))
    # print(sp.stats.kruskal(noise_diffs, inh_ablate_diffs))
    # print(sp.stats.kruskal(shuffled_diffs, inh_ablate_diffs))