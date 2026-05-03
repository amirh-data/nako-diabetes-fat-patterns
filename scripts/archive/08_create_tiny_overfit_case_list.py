from pathlib import Path
import pandas as pd


CASE_LIST = Path(
    "/mnt/qdata/projects/StudentsMarius/194_preds/studhaget1/encoder_3d_dataset/encoder_3d_cases.csv"
)

OUT_DIR = Path(
    "/mnt/qdata/projects/StudentsMarius/194_preds/studhaget1/encoder_3d_dataset"
)

N_NORMAL = 50
N_T2D = 50
RANDOM_SEED = 123


def main():
    df = pd.read_csv(CASE_LIST)

    normal = df[df["label"] == 0].copy()
    t2d = df[df["label"] == 1].copy()

    print("Original counts:")
    print(df["label"].value_counts())

    normal_sample = normal.sample(n=N_NORMAL, random_state=RANDOM_SEED)
    t2d_sample = t2d.sample(n=N_T2D, random_state=RANDOM_SEED)

    tiny = pd.concat([normal_sample, t2d_sample], axis=0)
    tiny = tiny.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    out_csv = OUT_DIR / "encoder_3d_cases_tiny_overfit_100.csv"
    tiny.to_csv(out_csv, index=False)

    print("\nTiny counts:")
    print(tiny["label"].value_counts())
    print(f"\nSaved to:\n{out_csv}")


if __name__ == "__main__":
    main()
