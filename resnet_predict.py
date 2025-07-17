import torch
import cv2
import torchvision.transforms as transforms
from resnet_model import ModifiedResNet

def predict_image(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ModifiedResNet().to(device)
    model.load_state_dict(torch.load("/content/drive/MyDrive/Data/image_classification_model.pth"))
    model.eval()

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    return "real" if predicted.item() == 0 else "fake"