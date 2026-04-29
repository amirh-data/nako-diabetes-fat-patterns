from pathlib import Path
import pandas as pd


CASE_LIST = Path(
    "/mnt/qdata/projects/StudentsMarius/194_preds/studhaget1/encoder_3d_dataset/encoder_3d_cases.csv"
)

OUT_DIR = Path(
    "/mnt/qdata/projects/StudentsMarius/194_preds/studhaget1/encoder_3d_dataset"
)

N_NORMAL = 1500
N_POSITIVE = 1500
RANDOM_SEED = 42


def main():
    df = pd.read_csv(CASE_LIST)

    normal = df[df["label"] == 0].copy()
    positive = df[df["label"] == 1].copy()

    print("Original counts:")
    print(df["label"].value_counts())

    normal_sample = normal.sample(
        n=min(N_NORMAL, len(normal)),
        random_state=RANDOM_SEED
    )

    positive_sample = positive.sample(
        n=min(N_POSITIVE, len(positive)),
        random_state=RANDOM_SEED
    )

    balanced = pd.concat([normal_sample, positive_sample], axis=0)
    balanced = balanced.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    out_csv = OUT_DIR / "encoder_3d_cases_balanced_t2d_3000.csv"
    balanced.to_csv(out_csv, index=False)

    print("\nBalanced counts:")
    print(balanced["label"].value_counts())

    print(f"\nSaved to:\n{out_csv}")


if __name__ == "__main__":
    main()
