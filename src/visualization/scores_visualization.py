import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


s = {'Caption': {'precision': 0.42857142857142855, 'recall': 0.42857142857142855, 'f1': 0.42857142857142855, 'number': 7}, 'Footnote': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'number': 4}, 'Page-Footer': {'precision': 1.0, 'recall': 0.9655172413793104, 'f1': 0.9824561403508771, 'number': 29}, 'Page-Header': {'precision': 1.0, 'recall': 0.9090909090909091, 'f1': 0.9523809523809523, 'number': 33}, 'Picture': {'precision': 0.3956639566395664, 'recall': 0.7192118226600985, 'f1': 0.5104895104895105, 'number': 203}, 'Table': {'precision': 0.9581005586592178, 'recall': 0.9985443959243085, 'f1': 0.9779044903777621, 'number': 1374}, 'Text': {'precision': 0.9930615784908933, 'recall': 0.8339402767662054, 'f1': 0.9065716547901822, 'number': 1373}, 'Title': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'number': 1}, 'overall_precision': 0.90244708994709, 'overall_recall': 0.90244708994709, 'overall_f1': 0.90244708994709, 'overall_accuracy': 0.90244708994709}


# Retrieve the scores and plot them
def plot_scores(scores):
    # Get the scores
    precisions = [scores[label]['precision'] for label in scores.keys() if label != 'overall_precision' and label != 'overall_recall' and label != 'overall_f1' and label != 'overall_accuracy']
    recalls = [scores[label]['recall'] for label in scores.keys() if label != 'overall_precision' and label != 'overall_recall' and label != 'overall_f1' and label != 'overall_accuracy']
    f1s = [scores[label]['f1'] for label in scores.keys() if label != 'overall_precision' and label != 'overall_recall' and label != 'overall_f1' and label != 'overall_accuracy']
    accuracies = [scores[label]['number'] for label in scores.keys() if label != 'overall_precision' and label != 'overall_recall' and label != 'overall_f1' and label != 'overall_accuracy']
    labels = [label for label in scores.keys() if label != 'overall_precision' and label != 'overall_recall' and label != 'overall_f1' and label != 'overall_accuracy']
    
    # Set the figure size
    plt.figure(figsize=(20, 10))
    
    # Plot the scores
    plt.subplot(2, 2, 1)
    plt.title("Precision")
    plt.bar(labels, precisions)
    plt.subplot(2, 2, 2)
    plt.title("Recall")
    plt.bar(labels, recalls)
    plt.subplot(2, 2, 3)
    plt.title("F1")
    plt.bar(labels, f1s)
    plt.subplot(2, 2, 4)
    plt.title("Number of elements")
    plt.bar(labels, accuracies)
    
    plt.show()

plot_scores(s)
