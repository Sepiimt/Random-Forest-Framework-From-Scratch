import matplotlib.pyplot as plt
import os
from pathlib import Path
import numpy as np
from typeguard import typechecked

#> ---------------------------------------------------------------------------------------

@typechecked
def model_metrics_plotter(metrics_foler_path: str = r"../results/benchmarks/",
                          plot_title: str = "Results vs. Model Configs",
                          save_plot: bool = True,
                          save_name: str = "metrics_bar_plot.png", 
                          save_to: str = r"../results/figures/"):
    # --- Documentation ---
    """
    #> Usage and Information
    Use this function to turn "metrics.txt"(s) into comparetive figures.
    
    #> Parameters Documentation:
    1. metrics_foler_path: Path to "metrics.txt".
    2. plot_title: Title for the entire plot.
    3. save_plot: Flag for saving the plot.
    4. save_name: Name for the saved plot file.
    5. save_to: Save path for the plotted figure.
    """
    # --- Reading Metrics Process ---
    zip_list, file_names = _metrics_extractor(metrics_foler_path)
    # --- Creating Plots ---
    _metrics_plotter(zip_list, file_names, save_name, save_plot, save_to, plot_title)

#> ---------------------------------------------------------------------------------------

def _metrics_extractor(metrics_foler_path):
    # --- Reading Metrics Process --- 
    #> Note: Parses by matching known field labels rather than fixed line numbers. The
    #> previous approach counted physical lines and hardcoded which indices to skip - it
    #> broke silently (wrong values landing in the wrong subplot, no error raised) the
    #> moment the writer's output structure changed at all, e.g. whether an RFModel was
    #> provided, or how many lines the confusion matrix repr happened to span.
    folder_path = Path(metrics_foler_path)
    field_labels = ["Cross_Entropy", "Accuracy", "Precision", "Recall", "F1_Score", "Brier_Score"]
    rows = []
    file_names = []
    # --- Iterating Over Metric Files in a stable, reproducible order ---
    #> Note: Path.glob() order is not guaranteed - sorting here means bar position on the
    #> plot is consistent across machines/runs, not an artifact of filesystem listing order.
    for file_path in sorted(folder_path.glob('*.txt')):
        found = {}
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                for label in field_labels:
                    prefix = f"> {label}:"
                    if line.startswith(prefix):
                        found[label] = line.split(":", 1)[1].strip()
        missing = [label for label in field_labels if label not in found]
        if missing:
            raise ValueError(f'"{file_path.name}" is missing expected field(s): {missing}')
        rows.append([found[label] for label in field_labels])
        file_names.append(file_path.stem)
    if not rows:
        raise ValueError(f'No ".txt" metric files found in "{metrics_foler_path}"!')
    # --- Transposing so each row is one metric, across all files ---
    zip_list = np.array(rows).T
    # --- Return ---
    return zip_list, file_names

#> ---------------------------------------------------------------------------------------

def _metrics_plotter(zip_list, file_names, save_name, save_plot, save_to, plot_title):
    #> Note: zip_list rows are, in order: Cross_Entropy, Accuracy, Precision, Recall,
    #> F1_Score, Brier_Score (see field_labels in _metrics_extractor) - indices 0-5, not
    #> 4-9. The previous 4-9 indexing only worked by accident (see review notes) and would
    #> now be wrong regardless, since zip_list no longer carries the RFModel info fields.
    n_configs = zip_list.shape[1]
    x_positions = list(range(1, n_configs + 1))

    def _bar(axis, row_index, title, color, alpha):
        values = [round(float(v), 8) for v in zip_list[row_index]]
        axis.bar(x=x_positions, height=values, color=color, alpha=alpha)
        axis.set_title(title)
        axis.set_xticks(x_positions)
        axis.set_xticklabels(file_names, rotation=0, ha='center', fontsize=10)
        axis.grid(True, linestyle='--', alpha=0.7)
        #> Note: no hardcoded ylim here anymore - the old bounds (e.g. accuracy 0.94-1.0)
        #> were fit to one prior dataset's results. Feed this a new sweep whose values fall
        #> outside those bounds and matplotlib clips or hides the bars entirely, with no
        #> error - it just looks like the run produced nothing, or produced garbage. Letting
        #> matplotlib auto-scale per figure is the safer default; pass explicit ylims back
        #> in if you want a fixed scale for comparing across multiple saved figures later.

    # --- Creating Plots ---
    fig, ax = plt.subplots(3, 2, figsize=(25, 10))
    fig.suptitle(plot_title, fontsize=18)
    _bar(ax[0,0], 0, 'Cross Entropy', "red", 0.3)
    _bar(ax[0,1], 1, 'Accuracy', "palegreen", 0.6)
    _bar(ax[1,0], 2, 'Precision', "deepskyblue", 0.3)
    _bar(ax[1,1], 3, 'Recall', "deepskyblue", 0.3)
    _bar(ax[2,0], 4, 'F1 Score', "palegreen", 0.6)
    _bar(ax[2,1], 5, 'Brier Score', "red", 0.3)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    # --- Saving the Plot ---
    if save_plot:
        os.makedirs(save_to, exist_ok=True)
        save_path = os.path.join(save_to, save_name)
        plt.savefig(save_path, dpi=300)
    # --- Display the Plot ---
        plt.show()
    # --- Closing pyplot ---
    plt.close()

#> ---------------------------------------------------------------------------------------
