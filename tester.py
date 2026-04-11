import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def perform_testing(net, batch_size, testing_set, weights_file: Path):
    print("performing testing...")
    testing_loader = DataLoader(testing_set, batch_size=batch_size, shuffle=True, num_workers=6)

    net_data = torch.load(str(weights_file))
    net.load_state_dict(net_data['model'], strict=True)

    net.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        targets = []
        predictions = []
        for inputs, labels in testing_loader:
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = net(inputs)

            pred_vals, pred_classes = torch.max(outputs.data, 1)
            correct += (pred_classes == labels).sum().item()
            total += labels.size(0)

            targets.extend(labels.detach().cpu().numpy())
            predictions.extend(pred_classes.detach().cpu().numpy())

    cm = confusion_matrix(y_true=targets, y_pred=predictions, normalize="true")
    cm = np.round(cm, 3)
    class_names = [
        'а', 'б', 'в', 'г', 'ґ', 'д', 'е', 'є',
        'ж', 'з', 'и', 'і', 'ї', 'й', 'к', 'л',
        'м', 'н', 'о', 'п', 'р', 'с', 'т', 'у',
        'ф', 'х', 'ц', 'ч', 'ш', 'щ', 'ь', 'ю', 'я'
    ]
    cmp = ConfusionMatrixDisplay(cm, display_labels=class_names)

    fig, ax = plt.subplots()
    fig.rcParams.update({'font.size': 6})
    label_font = {'size': '13'}
    ax.set_xlabel('Predicted labels', fontdict=label_font)
    ax.set_ylabel('Observed labels', fontdict=label_font)

    title_font = {'size': '16'}
    ax.set_title('Confusion Matrix', fontdict=title_font)
    cmp.plot(ax=ax)
    plt.show()

    print(classification_report(y_true=targets, y_pred=predictions, target_names=class_names))
