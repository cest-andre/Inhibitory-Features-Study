import sys
import os
import copy

from thingsvision import get_extractor
from scipy.stats import spearmanr, pearsonr
from scipy.spatial.distance import cosine
import numpy as np

from imnet_val import validate_tuning_curve_thingsvision
from modify_weights import clamp_ablate_unit, invert_weights


def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


#   TODO - how to measure similarity?  Cosine between the two 50k tuning vectors?
#          Spearman corr of ascending order of intact tuning curve (order inverted curve according to this)?
#          Compute both for now.  What about Pearson?  Is the monotonicity of Spearman preferred?
#          Perform on ablated units (for invert, ablate positives then invert).
#          If inverted is closer than a random unit in the trained case, will this hold for untrained?
def tuning_curve_similarity(extractor, unit, model_name="alexnet", curve_dir="/home/andre/tuning_curves"):
    exc_curve = np.load(os.path.join(curve_dir, f"{model_name}/inh_abl", f"layer11_unit{unit}_inh_abl_unrolled_act.npy"))
    exc_order = np.load(os.path.join(curve_dir, f"{model_name}/inh_abl", f"layer11_unit{unit}_inh_abl_all_ord_sorted.npy"))
    exc_curve = normalize(exc_curve[exc_order])

    #   Get exc_abl curve of same unit, multiply by -1 to obtain inverted curve.
    #   Order inverted curve to match.  Calculate cosine and correlation.
    inverted_curve = -1 * np.load(os.path.join(curve_dir, f"{model_name}/exc_abl", f"layer11_unit{unit}_exc_abl_unrolled_act.npy"))
    inverted_curve = normalize(inverted_curve)
    inverted_cos = cosine(exc_curve, inverted_curve[exc_order])
    print(f"Cosine sim of same unit invert:  {inverted_cos}")

    inverted_corr = pearsonr(exc_curve, inverted_curve[exc_order])
    print(f"Corr of same unit invert:  {inverted_corr}")

    all_exc_cos = []
    all_inh_cos = []
    all_exc_corr = []
    all_inh_corr = []
    #   Loop through all neurons i =/= unit and repeat wrt exc curve.
    for i in range(64):
        if i == unit:
            continue

        alt_exc_curve = np.load(os.path.join(curve_dir, f"{model_name}/inh_abl", f"layer11_unit{i}_inh_abl_unrolled_act.npy"))
        alt_exc_curve = normalize(alt_exc_curve)
        all_exc_cos.append(cosine(exc_curve, alt_exc_curve[exc_order]))
        all_exc_corr.append(pearsonr(exc_curve, alt_exc_curve[exc_order])[0])

        alt_inverted_curve = -1 * np.load(os.path.join(curve_dir, f"{model_name}/exc_abl", f"layer11_unit{i}_exc_abl_unrolled_act.npy"))
        alt_inverted_curve = normalize(alt_inverted_curve)
        all_inh_cos.append(cosine(exc_curve, alt_inverted_curve[exc_order]))
        all_inh_corr.append(pearsonr(exc_curve, alt_inverted_curve[exc_order])[0])

    print(f"Cosine sim average over all exc units:  {np.mean(np.array(all_exc_cos))}")
    print(f"Corr average over all exc units:  {np.mean(np.array(all_exc_corr))}")

    print(f"Cosine sim average over all inh units:  {np.mean(np.array(all_inh_cos))}")
    print(f"Corr average over all inh units:  {np.mean(np.array(all_inh_corr))}")


if __name__ == "__main__":
    # extractor = get_extractor(
    #     model_name='alexnet',
    #     source='torchvision',
    #     device='cuda',
    #     pretrained=True
    # )

    tuning_curve_similarity(None, 0)