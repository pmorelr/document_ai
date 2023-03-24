import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set()

# Retrieve the scores and plot them
def plot_scores(scores):
    # Get the scores
    precisions = [scores[label]['precision'] for label in scores.keys() if label != 'overall_precision' and label != 'overall_recall' and label != 'overall_f1' and label != 'overall_accuracy']
    recalls = [scores[label]['recall'] for label in scores.keys() if label != 'overall_precision' and label != 'overall_recall' and label != 'overall_f1' and label != 'overall_accuracy']
    f1s = [scores[label]['f1'] for label in scores.keys() if label != 'overall_precision' and label != 'overall_recall' and label != 'overall_f1' and label != 'overall_accuracy']
    accuracies = [scores[label]['number'] for label in scores.keys() if label != 'overall_precision' and label != 'overall_recall' and label != 'overall_f1' and label != 'overall_accuracy']
    labels = [label for label in scores.keys() if label != 'overall_precision' and label != 'overall_recall' and label != 'overall_f1' and label != 'overall_accuracy']
    
    # Set the figure size
    plt.figure(figsize=(22, 10))

    plt.rcParams['text.usetex'] = True
    palette = sns.color_palette().as_hex()
    
    # Plot the scores
    plt.subplot(2, 2, 1)
    plt.title(r'\textbf{Precision}', fontsize = 20)
    plt.bar(labels, precisions, color=palette, width=0.6)
    plt.subplot(2, 2, 2)
    plt.title(r'\textbf{Recall}', fontsize = 20)
    plt.bar(labels, recalls, color=palette, width=0.6)
    plt.subplot(2, 2, 3)
    plt.title(r'\textbf{f1 Score}', fontsize = 20)
    plt.bar(labels, f1s, color=palette, width=0.6)
    plt.subplot(2, 2, 4)
    plt.title(r'\textbf{Number of Elements}', fontsize = 20)
    plt.bar(labels, accuracies, color=palette, width=0.6)
    
    plt.show()
