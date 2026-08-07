import os
import gc
import numpy as np
from pathlib import Path
import sys
import argparse
import tomllib
# --- Resolving Package Path ---
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
# --- Importing API Files ---
from random_forest.forest import RandomForest

#> ---------------------------------------------------------------------------------------

def _load_toml_config(config_path: Path) -> dict:
    """Read binary TOML file using Python 3.11+ standard library."""
    with open(config_path, "rb") as f:
        raw_config = tomllib.load(f)
        return (raw_config["path"],
                raw_config["name"], 
                raw_config["random_forest"], 
                raw_config["iterable_config"])

#> ---------------------------------------------------------------------------------------

def _validate_iterable_lengths(iterable_cfg: dict) -> list[dict]:
    """
    Strips '_ls' from parameter keys and transposes list values into 
    individual column configuration dictionaries.
    """
    fi = len(iterable_cfg["n_trees"])
    si = len(iterable_cfg["min_leaf_purity"])
    ti = len(iterable_cfg["min_samples_split"])
    foi = len(iterable_cfg["min_samples_leaf"])
    if fi != si or fi != ti or fi != foi:
        raise ValueError("Config's iterable lenghts does not match!")
    # --- Return ---
    return fi

#> ---------------------------------------------------------------------------------------

def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Random Forest models using TOML configurations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Config path flag
    parser.add_argument(
        "config",
        type=Path,
        help="Path to the TOML configuration file",
    )
    # --- Return ---
    return parser.parse_args()

#> ---------------------------------------------------------------------------------------

def _prediction_loop(x_test, base_rf_params, iterable_cfg, config_paths, config_names, n_runs):
    for i in range(n_runs):
        # 1. Fixed leading slash in path join
        model_dir = os.path.join(config_paths["model"], f"config_{i+1}")
        # --- Loading Model ---
        RandomForestModel = RandomForest.load_model(
            model_dir,
            silent_load=True
        )
        # --- Predicting ---
        predicted_y = RandomForestModel.predict(
            x_test,
            n_jobs=base_rf_params["n_jobs"],
            timer=False
        )
        # --- Saving The Predictions ---
        # 2. Create target directory (not the file path itself)
        pred_dir = os.path.join(config_paths["prediction"], f"config_{i+1}")
        os.makedirs(pred_dir, exist_ok=True)
        # 3. Save array into the folder
        prediction_save_path = os.path.join(pred_dir, "predicted_y.npy")
        np.save(prediction_save_path, predicted_y)
        # --- Clean Up ---
        del RandomForestModel
        gc.collect()

#> ---------------------------------------------------------------------------------------

def main():
    # --- Receiving Arguments ---
    args = _parse_cli_args()
    # --- File Existance Guard ---
    if not args.config.exists():
        raise FileNotFoundError(f"Configuration file not found: {args.config}")
    # --- Parsing Argument ---
    config_paths, config_names, base_rf_params, iterable_cfg = _load_toml_config(args.config)
    # --- Getting Values ---
    n_runs = _validate_iterable_lengths(iterable_cfg)
    x_test = os.path.join(config_paths["data"],"x_test.npy")
    # --- Predicting ---
    _prediction_loop(x_test, base_rf_params, iterable_cfg, config_paths, config_names, n_runs)
    
#> ---------------------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[!] Process interrupted by user (^C). Force quitting...")
        gc.collect()
        raise SystemExit(1)

#> ---------------------------------------------------------------------------------------