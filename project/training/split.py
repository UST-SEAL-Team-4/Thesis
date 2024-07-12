from sklearn.model_selection import train_test_split
from dataset.dataset import VALDODataset

def split_train_val_datasets(df, transform, test_size=0.2, random_state=42):
    train_df, val_df = train_test_split(
        df, test_size=test_size, stratify=df['Has CMB'], random_state=random_state)
    
    train_dataset = VALDODataset(train_df['MRI Scans'].tolist(), train_df['Segmented Masks'].tolist(), transform=transform)
    
    val_dataset = VALDODataset(val_df['MRI Scans'].tolist(), val_df['Segmented Masks'].tolist(), transform=transform)

    return train_dataset, val_dataset

