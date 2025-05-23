import matplotlib.pyplot as plt
import pandas as pd

def plot_prediction(preds, trues, dates, title='Prediction vs True'):
    plt.figure(figsize=(12, 6))
    plt.plot(dates, trues, label='True')
    plt.plot(dates, preds, '--', label='Predicted')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()