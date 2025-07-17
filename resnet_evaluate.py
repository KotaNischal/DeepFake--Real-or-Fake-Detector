import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from resnet_model import ModifiedResNet

def evaluate_model(test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ModifiedResNet().to(device)
    model.load_state_dict(torch.load("/content/drive/MyDrive/Data/image_classification_model.pth"))
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("Accuracy:", accuracy_score(all_labels, all_preds))
    print("Precision:", precision_score(all_labels, all_preds))
    print("Recall:", recall_score(all_labels, all_preds))
    print("F1 Score:", f1_score(all_labels, all_preds))
    print(classification_report(all_labels, all_preds))
