import pandas as pd
import os
from project.dataset import Dataset, VALDODataset
from project.preprocessing import get_transform
from project.training import split_train_val_datasets, run_training
        
def main():
        
    ds = Dataset()
    print('Step 0 check')

    labels = ds.load_cmb_masks()
    ids = ds.load_raw_mri()
    print('Step 1 check')

    transform = get_transform(height=256, width=256, p=1.0)
    print('Step 2 check')

    dataset = VALDODataset(img_paths=ids,
                           ann_paths=labels,
                           transform=transform,
                           normalization=None)
                            # normalization='z_score')
                            # normalization='min_max')
    print('Step 3 check')

    # create dataframe
    has_cmb = [1 if count > 0 else 0 for count in dataset.cmb_counts]
    print('Step 4 check')

    df_dataset = pd.DataFrame({
        'MRI Scans': dataset.img_paths,
        'Segmented Masks': dataset.ann_paths,
        'CMB Count': dataset.cmb_counts,
        'Has CMB': has_cmb
    })
    print('Step 5 check')

    # training and validation split
    train_dataset, val_dataset = split_train_val_datasets(df=df_dataset, transform=transform)
    print('Step 6 check')

    # run training
    best_val_loss, summary_loss_over_itr_train, summary_loss_over_itr_val, history = run_training(train_dataset, val_dataset)
    print('Step 7 check')
    
if __name__ == '__main__':
    main()