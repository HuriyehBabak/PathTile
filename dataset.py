from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


# Pascal / L3S dataset path
DATA_DIR = (
    Path.home()
    / "s4convd-paper"
    / "pre_preprocessed_data"
)


class ASHRAEDataset(Dataset):
    def __init__(
        self,
        df,
        input_length=48,
        means=None,
        stds=None,
    ):
        self.input_length = input_length

        # Input features
        self.df = df[
            [
                "meter_reading",
                "air_temperature",
                "cloud_coverage",
                "dew_temperature",
            ]
        ].dropna()

        # Keep normalization identical to the original code
        self.means = self.df.mean()
        self.stds = self.df.std()

        self.data = (
            self.df - self.means
        ) / self.stds

    def __getitem__(self, index):
        input_start = index
        input_end = input_start + self.input_length

        if input_end >= len(self.data):
            raise IndexError("Index exceeds")

        input_data = self.data.iloc[
            input_start:input_end
        ].values

        label_data = self.data.iloc[
            input_end, 0
        ]

        return {
            "input_sequence": torch.tensor(
                input_data,
                dtype=torch.float32,
            ),
            "label_sequence": torch.tensor(
                label_data,
                dtype=torch.float32,
            ),
        }

    def __len__(self):
        return len(self.data) - self.input_length


def load_local_data(
    name,
    fraction=0.1,
):
    path = DATA_DIR / f"{name}.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    df = pd.read_csv(path)

    # Keep sampling identical to the original Kaggle experiment
    df = df.sample(
        frac=fraction,
        random_state=42,
    )

    return df


if __name__ == "__main__":

    FRACTION = 0.1
    INPUT_LENGTH = 48
    BATCH_SIZE = 8192
    NUM_WORKERS = 4
    PREFETCH = 2

    train_df = load_local_data(
        "train",
        FRACTION,
    )

    val_df = load_local_data(
        "val",
        FRACTION,
    )

    test_df = load_local_data(
        "test",
        FRACTION,
    )

    train_dataset = ASHRAEDataset(
        train_df,
        input_length=INPUT_LENGTH,
    )

    val_dataset = ASHRAEDataset(
        val_df,
        input_length=INPUT_LENGTH,
    )

    test_dataset = ASHRAEDataset(
        test_df,
        input_length=INPUT_LENGTH,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        drop_last=True,
        pin_memory=True,
        prefetch_factor=PREFETCH,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        drop_last=False,
        pin_memory=True,
        prefetch_factor=PREFETCH,
        persistent_workers=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        drop_last=False,
        pin_memory=True,
        prefetch_factor=PREFETCH,
        persistent_workers=True,
    )

    print("Dataset test OK")
    print(
        f"Train batches: {len(train_loader)} | "
        f"Val: {len(val_loader)} | "
        f"Test: {len(test_loader)}"
    )

    sample = train_dataset[0]

    print(
        "Input shape:",
        sample["input_sequence"].shape,
    )

    print(
        "Label shape:",
        sample["label_sequence"].shape,
    )