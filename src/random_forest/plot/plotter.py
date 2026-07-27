import matplotlib.pyplot as plt
import os
from pathlib import Path
import numpy as np

def model_metrics_plotter(metrics_foler_path = r"..\\results\\benchmarks\\", 
                          save_plot = True, 
                          save_to = r"..\\results\\figures\\"):
    # --- Documentation ---
    """
    #> Usage and Information
    Use this function to turn "metrics.txt"(s) into comparetive figures.
    
    #> Parameters Documentation:
    1. metrics_foler_path: Path to "metrics.txt".
    2. save_plot: Flag for saving the plot.
    3. save_to: Save path for the plotted figure.
    """
    # --- Reading Metrics Process ---
    zip_list = _metrics_extractor(metrics_foler_path)
    # --- Creating Plots ---
    _metrics_plotter(zip_list, save_plot, save_to)
    

def _metrics_extractor(metrics_foler_path):
    #> --- Reading Metrics Process --- <#
    folder_path = Path(metrics_foler_path)
    #> total_trees, tree_length, minimum_leaf_purity, Cross_Entropy, Accuracy, Precision, Recall, F1_Score, Brier_Score
    zip_list=[]
    # --- »Lines to Ignore ---
    ignore_line=[0, 4, 6, 7, 9, 10, 11, 12]
    # --- Iterating Over Metric Files ---
    for file_path in folder_path.glob('*.txt'):
        with open(file_path, 'r', encoding='utf-8') as file:
            temp_list=[]
            count=0
            for line in file:
                if count in ignore_line:
                    count+=1
                    continue
                splitted_words = (line.strip()).split(" ")
                temp_list.append(splitted_words[-1])
                count+=1
            zip_list.append(temp_list)
    # --- Transposing the Extracted Array ---
    zip_list=np.array(zip_list).T
    # --- Return ---
    return zip_list


def _metrics_plotter(zip_list, save_plot, save_to):
    # --- Creating Plots ---
    fig, ax = plt.subplots(3, 2, figsize=(25, 10))
    fig.suptitle('Results vs. Model Configs', fontsize=16)
    # --- Axis 0 ---
    ax[0,0].bar(x=[i for i in range(1,7)], height=[round(float(x), 8) for x in zip_list[4]], color ="red", alpha=0.3)
    ax[0,0].set_title('Cross Entropy')
    ax[0,0].grid(True, linestyle='--', alpha=0.7)
    ax[0,0].set_ylim(0.12, 0.2)
    # --- Axis 0 ---
    ax[0,1].bar(x=[i for i in range(1,7)], height=[round(float(x), 8) for x in zip_list[5]], color ="palegreen", alpha=0.6)
    ax[0,1].set_title('Accuracy')
    ax[0,1].grid(True, linestyle='--', alpha=0.7)
    ax[0,1].set_ylim(0.94, 1)
    # --- Axis 1 ---
    ax[1,0].bar(x=[i for i in range(1,7)], height=[round(float(x), 8) for x in zip_list[6]], color ="deepskyblue", alpha=0.3)
    ax[1,0].set_title('Precision')
    ax[1,0].grid(True, linestyle='--', alpha=0.7)
    ax[1,0].set_ylim(0.9, 1)
    # --- Axis 1 ---
    ax[1,1].bar(x=[i for i in range(1,7)], height=[round(float(x), 8) for x in zip_list[7]], color ="deepskyblue", alpha=0.3)
    ax[1,1].set_title('Recall')
    ax[1,1].grid(True, linestyle='--', alpha=0.7)
    ax[1,1].set_ylim(0.9, 1)
    # --- Axis 2 ---
    ax[2,0].bar(x=[i for i in range(1,7)], height=[round(float(x), 8) for x in zip_list[8]], color ="palegreen", alpha=0.6)
    ax[2,0].set_title('F1 Score')
    ax[2,0].grid(True, linestyle='--', alpha=0.7)
    ax[2,0].set_ylim(0.9, 1)
    # --- Axis 2 ---
    ax[2,1].bar(x=[i for i in range(1,7)], height=[round(float(x), 8) for x in zip_list[9]], color ="red", alpha=0.3)
    ax[2,1].set_title('Brier Score')
    ax[2,1].grid(True, linestyle='--', alpha=0.7)
    ax[2,1].set_ylim(0.025, 0.05)
    # --- Saving the Plot ---
    if save_plot:
        save_path = os.path.join(save_to, "metrics_bar_plot.png")
        plt.savefig(save_path, dpi=300)
    # --- Display the Plot ---
        plt.show()
    # --- Closing pyplot ---
    plt.close()