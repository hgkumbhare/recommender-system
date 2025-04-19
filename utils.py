import matplotlib.pyplot as plt

def plot_curves(training_metric_data, validation_metric_data, title, x_label, y_label):
    
    """Plot training and validation metrics over epochs"""
    
    plt.figure(figsize=(10, 5))
    plt.plot(training_metric_data, label='Training')
    plt.plot(validation_metric_data, label='Validation')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig('loss_curves.png', dpi=300)