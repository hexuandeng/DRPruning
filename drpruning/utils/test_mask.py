import sys
import math
import torch
from collections import defaultdict

def get_masks(path):
    def load_weights(path):
        """ load model weights from a path """
        state_dict = {}
        p_weight = torch.load(path, map_location=torch.device('cpu'))
        if "state" in p_weight:
            state_dict.update(p_weight["state"]["model"])
        else:
            state_dict.update(p_weight)

        print("Loaded model from path: ", path)
        return state_dict

    def cdf_qz(z_loga: torch.Tensor = None):
        """Implements the CDF of the 'stretched' concrete distribution"""
        limit_a, limit_b, epsilon = -.1, 1.1, 1e-6
        temperature = 2./3.

        xn = (0 - limit_a) / (limit_b - limit_a)
        logits = math.log(xn) - math.log(1 - xn)
        return torch.sigmoid(logits * temperature - z_loga).clamp(min=epsilon, max=1 - epsilon)

    def deterministic_z(z_loga, target_mask_size=None):
        temperature = 2./3.
        magical_number = 0.8

        if target_mask_size is None:
            expected_score = 1 - cdf_qz(z_loga)
            expected_num_nonzeros = expected_score.sum()
            expected_num_zeros = z_loga.nelement() - expected_num_nonzeros.item()
        else:
            expected_num_zeros = z_loga.shape[-1] - target_mask_size 
        try:
            num_zeros = round(expected_num_zeros)
        except:
            print("num of zeros is nan....")
            sys.exit()
        soft_mask = torch.sigmoid(z_loga / temperature * magical_number)
        if num_zeros > 0:
            if soft_mask.ndim == 0:
                soft_mask = torch.tensor(0).to(z_loga.device)
            else:
                _, indices = torch.topk(soft_mask, k=num_zeros, largest=False)
                soft_mask[indices] = 0.
        return soft_mask

    model = load_weights(path)
    remains = defaultdict(list)

    for sub in ["head", "intermediate"]:
        module = model[f'model.l0_module.masks.{sub}.z_loga']
        for i in range(module.shape[0]):
            remains[sub].append(deterministic_z(module[i], target_mask_size=int(module.shape[-1] / 2)))

    for sub in ["layer", "hidden"]:
        module = model[f'model.l0_module.masks.{sub}.z_loga']
        remains[sub].append(deterministic_z(module, target_mask_size=24 if sub == "layer" else 2048))

    return remains

def test_mask(data):
    import numpy as np
    from scipy import stats
    values = [float(line.split('(')[1].rstrip(')')) * 100 for line in data.strip().split('\n')]
    
    mean = np.mean(values)
    sem = stats.sem(values)
    print(f"Mean ± SEM: {mean} ± {sem}")

def compare_tensor_lists(list1, list2):
    assert len(list1) == len(list2), "Lists must have the same length"

    equal_elements = 0
    total_elements = 0
    for tensor1, tensor2 in zip(list1, list2):
        assert tensor1.shape == tensor2.shape, "Tensors must have the same shape"
        equal_elements += torch.sum((tensor1 == 0.) == (tensor2 == 0.))
        total_elements += tensor1.numel()

    return equal_elements.float() / total_elements
