import pandas as pd
from .dataset import Dataset, VALDODataset
from .preprocessing import get_transform
from .training import split_train_val_datasets, run_training

# prepare dataset
all_labels, all_ids = Dataset().get_all_labels_and_ids()

transform = get_transform(height=256, width=256, p=1.0)

dataset = VALDODataset(img_paths=all_ids, 
                       ann_paths=all_labels, 
                       transform=transform)

# create dataframe
has_cmb = [1 if count > 0 else 0 for count in dataset.cmb_counts]

df_dataset = pd.DataFrame({
    'MRI Scans': dataset.img_paths,
    'Segmented Masks': dataset.ann_paths,
    'CMB Count': dataset.cmb_counts,
    'Has CMB': has_cmb
})

# training and validation split
train_dataset, val_dataset = split_train_val_datasets(df=df_dataset, transform=transform)

# run training
best_val_loss, summary_loss_over_itr_train, summary_loss_over_itr_val, history = run_training()