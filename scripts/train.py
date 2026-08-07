import argparse
import tomllib
import os
from pathlib import Path
import gc
import sys
# --- Resolving Package Path ---
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
# --- Importing API Files ---
from random_forest.forest import RandomForest
from random_forest.utils import MemoryTracker

#> ---------------------------------------------------------------------------------------

def _load_toml_config(config_path: Path):
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

def _train_loop(x_train, y_train, base_rf_params, iterable_cfg, config_paths, config_names, n_runs):
    # --- Memory Tracking ---
    with MemoryTracker(sample_interval_sec=0.01) as tracker:
        # --- Loop ---
        for i in range(n_runs):
            # --- Getting an Instance from the Model ---
            RandomForestModel = RandomForest()
            # --- Fitting ---
            RandomForestModel.fit(
                x_train, y_train,
                cc_indices = base_rf_params["cc_indices"] if len(base_rf_params["cc_indices"])!=0 else None,
                n_trees = iterable_cfg["n_trees"][i], 
                trees_max_depth = base_rf_params["trees_max_depth"],
                min_leaf_purity = iterable_cfg["min_leaf_purity"][i],
                bootstrap_balance = base_rf_params["bootstrap_balance"],
                criterion = base_rf_params["criterion"],
                min_samples_split = iterable_cfg["min_samples_split"][i],
                min_samples_leaf = iterable_cfg["min_samples_leaf"][i],
                random_state = base_rf_params["random_state"],
                n_jobs = base_rf_params["n_jobs"]
                )
            # --- Saving the Model ---
            RandomForestModel.save_model(
                directory=os.path.join(config_paths["model"], f"config_{i+1}"), 
                silent_save=True)
            # --- Clean Up ---
            del RandomForestModel
            gc.collect()
    # --- Saving Profile ---
    tracker.save_report(
        path =  config_paths["memory_profile"],
        file_name = config_names["profile_filename"], 
        label = config_names["profile_label"])

#> ---------------------------------------------------------------------------------------

def main():
    # --- Receiving Arguments ---
    args = _parse_cli_args()
    # --- File Existance Guard ---
    if not args.config.exists():
        raise FileNotFoundError(f"Configuration file not found: {args.config}")
    # --- Parse Info From TOML ---
    config_paths, config_names, base_rf_params, iterable_cfg = _load_toml_config(args.config)
    # --- Getting Values ---
    n_runs = _validate_iterable_lengths(iterable_cfg)
    x_train = os.path.join(config_paths["data"],"x_train.npy")
    y_train = os.path.join(config_paths["data"],"y_train.npy")
    # --- Training ---
    _train_loop(x_train, y_train, base_rf_params, iterable_cfg, config_paths, config_names, n_runs)

#> ---------------------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[!] Process interrupted by user (^C). Force quitting...")
        gc.collect()
        raise SystemExit(1)

#> ---------------------------------------------------------------------------------------
