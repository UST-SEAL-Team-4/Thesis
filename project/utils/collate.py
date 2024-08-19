import torch

def collate_fn(batch):
    slices = []
    targets = []
    img_paths = []
    cmb_counts = []

    for item in batch:
        if item is not None:  # Skip None items
            item_slices, item_targets, item_img_path, item_cmb_counts = item
            slices.extend(item_slices)
            targets.extend(item_targets)
            img_paths.append(item_img_path)
            cmb_counts.append(item_cmb_counts)

    if slices:
        cases = torch.stack(slices, dim=0)
        masks = torch.stack(targets, dim=0)
        return cases, masks, img_paths, cmb_counts
    else:
        return None, None, [], []