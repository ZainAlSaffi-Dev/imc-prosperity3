import pandas as pd
import os
import sys
from typing import List


def combine_market_data(data_type: str, round_num: str) -> None:
    """Aggregates daily CSV files into a unified time-series dataset.

    Performs vertical concatenation of daily files to preserve time continuity.
    Injects a 'day' column to differentiate overlapping timestamps between days.

    Args:
        data_type: The mode of operation. 'p' for prices, 't' for trades.
        round_num: The round identifier used for directory resolution.
    """
    is_price = data_type == "p"
    file_prefix = "prices" if is_price else "trades"
    id_col = "product" if is_price else "symbol"

    # Path is relative to the project root
    base_dir = os.path.join("data", f"round{round_num}")

    if not os.path.exists(base_dir):
        print(f"Directory not found: {base_dir}")
        return

    frames: List[pd.DataFrame] = []

    # Iterate through standard competition days (-2 to 0)
    for day in range(-2, 1):
        filename = f"{file_prefix}_round_{round_num}_day_{day}.csv"
        filepath = os.path.join(base_dir, filename)

        if not os.path.exists(filepath):
            continue

        print(f"Reading: {filename}")

        try:
            df = pd.read_csv(filepath, sep=";")
        except pd.errors.EmptyDataError:
            print(f"Skipping empty file: {filename}")
            continue
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            continue

        # Standardize headers
        df.columns = df.columns.str.strip().str.lower()

        if "timestamp" not in df.columns or id_col not in df.columns:
            print(f"Skipping {filename}: Missing critical columns")
            continue

        # Inject day identifier if missing
        if "day" not in df.columns:
            df.insert(0, "day", day)

        frames.append(df)

    if not frames:
        print("No valid dataframes found to combine.")
        return

    # Stack frames vertically
    combined_df = pd.concat(frames, axis=0, ignore_index=True)

    # Sort chronologically to ensure linear time progression
    combined_df = combined_df.sort_values(by=["day", "timestamp", id_col]).reset_index(
        drop=True
    )

    output_filename = f"combined_{file_prefix}_round_{round_num}.csv"
    output_path = os.path.join(base_dir, output_filename)

    try:
        combined_df.to_csv(output_path, index=False, sep=";")
        print(f"Successfully saved {len(combined_df)} rows to: {output_path}")
    except PermissionError:
        print(f"Write failed: Permission denied for {output_path}")


# Script execution
try:
    user_type = input("Enter 'p' for prices or 't' for trades: ").strip().lower()
    if user_type not in ["p", "t"]:
        print("Invalid input.")
        sys.exit(1)

    user_round = input("Enter round number (e.g., 1): ").strip()
    combine_market_data(user_type, user_round)

except KeyboardInterrupt:
    print("\nOperation cancelled.")
    sys.exit(0)
