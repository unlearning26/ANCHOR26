# This code is adapted from the original DINOv2 repository: https://github.com/facebookresearch/dinov2
# This code is licensed under the CC BY-NC-ND 4.0 license
# found in the LICENSE file in the root directory of this source tree.

import torch
import random


def collate_data_and_cast(samples_list, mask_ratio_tuple, mask_probability, dtype, n_tokens=None, mask_generator=None):
    # dtype = torch.half  # TODO: Remove
    # suppose len(samples_list)=4 (batch size)
    
    # Handle both dict format (from MONAI transform chain) and tuple format (from adapter)
    # samples_list contains augmentation output dicts from DataAugmentationDINO3d
    first_sample = samples_list[0][0] if isinstance(samples_list[0], (tuple, list)) else samples_list[0]

    n_global_crops = len(first_sample["global_crops"])
    n_local_crops = len(first_sample["local_crops"])

    # Extract augmentation dict from each sample (handle both tuple and dict formats)
    def get_aug_dict(s):
        return s[0] if isinstance(s, (tuple, list)) else s

    collated_global_crops = torch.stack([get_aug_dict(s)["global_crops"][i] for i in range(n_global_crops) for s in samples_list])

    collated_local_crops = torch.stack([get_aug_dict(s)["local_crops"][i] for i in range(n_local_crops) for s in samples_list])

    B = len(collated_global_crops)
    N = n_tokens
    n_samples_masked = int(B * mask_probability)
    probs = torch.linspace(*mask_ratio_tuple, n_samples_masked + 1)
    # mask_ratio_tuple = (0.2, 0.75), n_samples_masked = 4
    # probs = [0.200, 0.3375, 0.475, 0.6125, 0.75]
    #         ↑       ↑        ↑       ↑        ↑
    #        min   sample0  sample1  sample2   max
    upperbound = 0
    masks_list = []
    for i in range(0, n_samples_masked):
        prob_min = probs[i]
        prob_max = probs[i + 1]
        masks_list.append(torch.BoolTensor(mask_generator(int(N * random.uniform(prob_min, prob_max)))))
        upperbound += int(N * prob_max)
    for i in range(n_samples_masked, B):
        masks_list.append(torch.BoolTensor(mask_generator(0)))

    random.shuffle(masks_list)

    collated_masks = torch.stack(masks_list).flatten(1)
    mask_indices_list = collated_masks.flatten().nonzero().flatten() # Returns the indices of the masked patches in the batch (non-zeros in the mask)

    # DEFENSIVE GUARD: Validate mask indices are within bounds
    total_patches = B * N
    if mask_indices_list.numel() > 0:
        assert mask_indices_list.max() < total_patches, (
            f"Mask index out of bounds: max_idx={mask_indices_list.max()}, total_patches={total_patches}"
        )

    # DEFENSIVE GUARD: Detect unexpectedly high zero-mask rate
    masks_per_sample = collated_masks.sum(-1)
    zero_mask_samples = (masks_per_sample == 0).sum().item()
    expected_zero = int(B * (1 - mask_probability))  # Expected zeros from mask_probability
    # Warn only if significantly more zeros than expected (tolerance of 2)
    if zero_mask_samples > expected_zero + 2:
        import logging
        logging.getLogger("dinov2").warning(
            f"[MaskGuard] {zero_mask_samples}/{B} samples have zero masks (expected ~{expected_zero})"
        )
    
    masks_weight = (1 / masks_per_sample.clamp(min=1.0)).unsqueeze(-1).expand_as(collated_masks)[collated_masks]
    
    # DEFENSIVE GUARD: Validate masks_weight has no inf/nan from division
    assert torch.isfinite(masks_weight).all(), "masks_weight contains inf/nan after division"

    return {
        "collated_global_crops": collated_global_crops.to(dtype),
        "collated_local_crops": collated_local_crops.to(dtype),
        "collated_masks": collated_masks,
        "mask_indices_list": mask_indices_list,
        "masks_weight": masks_weight,
        "upperbound": upperbound,
        "n_masked_patches": torch.full((1,), fill_value=mask_indices_list.shape[0], dtype=torch.long),
    }
