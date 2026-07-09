import argparse
import re
import zipfile
import io
from pathlib import Path

import pandas as pd

# The four activity classes 
VALID_ACTIVITIES = {"standing", "walking", "jumping", "still"}

TRIAL_NAME_RE = re.compile(r"^(?P<activity>[a-zA-Z]+)_trial(?P<trial>\d+)", re.IGNORECASE)


def parse_trial_filename(filename: str):
    """Pulls the activity label and trial number out of a zip's filename."""
    match = TRIAL_NAME_RE.match(filename)
    if not match:
        return None, None
    activity = match.group("activity").lower()
    trial_num = int(match.group("trial"))
    return activity, trial_num


def load_sensor_csv(zf: zipfile.ZipFile, name: str) -> pd.DataFrame:
    """Reads one sensor CSV (Accelerometer.csv / Gyroscope.csv) out of the zip."""
    with zf.open(name) as f:
        df = pd.read_csv(io.BytesIO(f.read()))
    return df


def process_one_zip(zip_path: Path) -> pd.DataFrame:
    """
    Aligns Accelerometer + Gyroscope readings for a single trial on their
    shared 'time' (nanosecond epoch) column and returns one tidy dataframe.
    """
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        if "Accelerometer.csv" not in names or "Gyroscope.csv" not in names:
            raise FileNotFoundError(f"{zip_path.name} is missing Accelerometer/Gyroscope CSV")

        acc = load_sensor_csv(zf, "Accelerometer.csv")
        gyro = load_sensor_csv(zf, "Gyroscope.csv")

        acc = acc.rename(columns={"x": "acc_x", "y": "acc_y", "z": "acc_z"})
        gyro = gyro.rename(columns={"x": "gyro_x", "y": "gyro_y", "z": "gyro_z"})

        acc_sorted = acc.sort_values("time")
        gyro_sorted = gyro.sort_values("time")
        merged = pd.merge_asof(
            acc_sorted,
            gyro_sorted[["time", "gyro_x", "gyro_y", "gyro_z"]],
            on="time",
            direction="nearest",
        )

        merged = merged.rename(columns={"seconds_elapsed": "t_seconds"})
        merged = merged[["time", "t_seconds", "acc_x", "acc_y", "acc_z",
                          "gyro_x", "gyro_y", "gyro_z"]]
        merged = merged.dropna().reset_index(drop=True)
        return merged


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/zips_raw",
                         help="Folder containing the raw Sensor Logger trial zips")
    parser.add_argument("--output", default="data/processed",
                         help="Folder to write clean per-trial CSVs into")
    args = parser.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    zip_paths = sorted(in_dir.glob("*.zip"))
    if not zip_paths:
        raise SystemExit(f"No zip files found in {in_dir.resolve()}")

    summary_rows = []
    skipped = []

    for zp in zip_paths:
        activity, trial_num = parse_trial_filename(zp.name)
        if activity not in VALID_ACTIVITIES:
            skipped.append(zp.name)
            continue

        df = process_one_zip(zp)
        df.insert(0, "activity", activity)
        df.insert(1, "trial_id", f"{activity}_{trial_num:02d}")

        out_name = f"{activity}_trial{trial_num:02d}.csv"
        df.to_csv(out_dir / out_name, index=False)

        duration_s = df["t_seconds"].iloc[-1] - df["t_seconds"].iloc[0]
        sample_rate_hz = round(len(df) / duration_s) if duration_s > 0 else float("nan")
        summary_rows.append({
            "file": out_name,
            "activity": activity,
            "trial": trial_num,
            "n_samples": len(df),
            "duration_s": round(duration_s, 2),
            "approx_sample_rate_hz": sample_rate_hz,
        })
        print(f"[ok] {zp.name:55s} -> {out_name:30s} "
              f"({len(df)} rows, {duration_s:.2f}s, ~{sample_rate_hz} Hz)")

    if skipped:
        print(f"\nSkipped {len(skipped)} file(s) that didn't match the naming pattern:")
        for s in skipped:
            print(f"  - {s}")

    summary = pd.DataFrame(summary_rows).sort_values(["activity", "trial"])
    summary_path = out_dir / "_dataset_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\n=== Dataset summary ===")
    print(summary.groupby("activity").agg(
        n_trials=("file", "count"),
        total_duration_s=("duration_s", "sum"),
        mean_sample_rate_hz=("approx_sample_rate_hz", "mean"),
    ))
    print(f"\nWrote {len(summary_rows)} labelled CSVs to {out_dir.resolve()}")
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
