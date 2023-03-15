import evaluate
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_seqeval(predictions, labels, ner_labels):
    """
    Evaluate the predictions using the seqeval metric
    
    Args:
        predictions (np.array): Predictions of the model
        labels (np.array): Labels of the dataset
        ner_labels (list): List of the NER labels

    Returns:
        metrics (dict): Dictionary containing the metrics
    """
    # Load seqeval metric
    metric = evaluate.load("seqeval")

    # Arrange predictions and labels
    predictions = np.argmax(predictions, axis=2)

    all_predictions = []
    all_labels = []
    for prediction, label in zip(predictions, labels):
        for predicted_idx, label_idx in zip(prediction, label):
            if label_idx == -100:
                continue
            all_predictions.append(ner_labels[predicted_idx])
            all_labels.append(ner_labels[label_idx])

    # Return the metrics
    return metric.compute(predictions=[all_predictions], references=[all_labels])

def plot_metrics(metrics):
    """ Plot the metrics
    
    Args:
        metrics (dict): Dictionary containing the metrics
        
    Returns:
        None
    """

    # Plot the metrics
    plt.figure(figsize=(12, 6))
    sns.barplot(x=list(metrics.keys()), y=list(metrics.values()))
    plt.title("Metrics")
    plt.show()
