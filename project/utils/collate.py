import torch

def collate_fn(batch):
    slices = []
    targets = []
    img_paths = []
    cmb_counts = []
    
    for item in batch:
        item_slices, item_targets, item_img_path, item_cmb_counts = item
        slices.extend(item_slices)
        targets.extend(item_targets)
        img_paths.append(item_img_path)
        cmb_counts.append(item_cmb_counts)

    slices = [torch.stack(tuple(slice_set)) for slice_set in slices]

    return slices, targets, img_paths, cmb_counts