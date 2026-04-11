
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt
from helpers.common import printshare
import mlflow
import sys

def prepare_confusion_matrix(targets, predictions):
    cm = confusion_matrix(y_true=targets, y_pred=predictions, normalize="true")
    cm = np.round(cm, 3)
    class_names = [
        'а', 'б', 'в', 'г', 'ґ', 'д', 'е', 'є',
        'ж', 'з', 'и', 'і', 'ї', 'й', 'к', 'л',
        'м', 'н', 'о', 'п', 'р', 'с', 'т', 'у',
        'ф', 'х', 'ц', 'ч', 'ш', 'щ', 'ь', 'ю', 'я'
    ]
    cmp = ConfusionMatrixDisplay(cm, display_labels=class_names)

    fig, ax = plt.subplots(figsize=(14, 14))

    label_font = {'size': '13'}
    ax.set_xlabel('Predicted labels', fontdict=label_font)
    ax.set_ylabel('Observed labels', fontdict=label_font)

    title_font = {'size': '16'}
    ax.set_title('Confusion Matrix', fontdict=title_font)
    cmp.plot(ax=ax, text_kw={'size': 6})
    return fig
