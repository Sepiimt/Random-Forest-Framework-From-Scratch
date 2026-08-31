import argparse
import numpy as np
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
        return (raw_config["model"]["model"],
                raw_config["path"],
                raw_config["ram_profiler_names"], 
                raw_config["model_params"], 
                raw_config["iterable_config"])
    
#> ---------------------------------------------------------------------------------------

def _validate_rf_iterable_lengths(config_iterable_params: dict) -> list[dict]:
    fi = len(config_iterable_params["n_trees"])
    si = len(config_iterable_params["trees_max_depth"])
    ti = len(config_iterable_params["min_leaf_purity"])
    foi = len(config_iterable_params["min_samples_split"])
    fivi = len(config_iterable_params["min_samples_leaf"])
    criteria = len(np.unique([fi, si, ti, foi, fivi]))
    if criteria != 1:
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

def _rf_train_loop(x_train, y_train, 
                   selected_model, 
                   config_paths, 
                   config_names, 
                   config_static_params, 
                   config_iterable_params, 
                   n_runs):
    # --- Memory Tracking ---
    with MemoryTracker(sample_interval_sec=0.25) as tracker:
            # --- Loop ---
        for i in range(n_runs):
            # --- Getting an Instance from the Model ---
            RandomForestModel = RandomForest()
            # --- Fitting ---
            RandomForestModel.fit(
                x_train, y_train,
                cc_indices = config_static_params["cc_indices"] if len(config_static_params["cc_indices"])!=0 else None,
                n_trees = config_iterable_params["n_trees"][i], 
                trees_max_depth = config_iterable_params["trees_max_depth"][i],
                min_leaf_purity = config_iterable_params["min_leaf_purity"][i],
                bootstrap_balance = config_static_params["bootstrap_balance"],
                criterion = config_static_params["criterion"],
                min_samples_split = config_iterable_params["min_samples_split"][i],
                min_samples_leaf = config_iterable_params["min_samples_leaf"][i],
                random_state = config_static_params["random_state"],
                n_jobs = config_static_params["n_jobs"]
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
    (selected_model, config_paths, 
     config_names, config_static_params, 
     config_iterable_params) = _load_toml_config(args.config)
    # --- Getting Values ---
    n_runs = _MODEL_ITER_VALIDATOR[selected_model](config_iterable_params)
    x_train = os.path.join(config_paths["data"],"x_train.npy")
    y_train = os.path.join(config_paths["data"],"y_train.npy")
    # --- Training ---
    if selected_model == "random_forest":
        _rf_train_loop(x_train, y_train, selected_model,
                       config_paths, config_names,
                       config_static_params,
                       config_iterable_params, 
                       n_runs)
    else:
        raise ValueError("Selected model is not available!")

#> ---------------------------------------------------------------------------------------

_MODEL_ITER_VALIDATOR = {
    "random_forest" : _validate_rf_iterable_lengths
}

#> ---------------------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[!] Process interrupted by user (^C). Force quitting...")
        gc.collect()
        raise SystemExit(1)

#> ---------------------------------------------------------------------------------------