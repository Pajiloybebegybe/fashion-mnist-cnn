import argparse
from PIL import Image
import torch
from torchvision import transforms

from models.cnn import CNN
from train import FASHION_MEAN, FASHION_STD

CLASS_NAMES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = CNN().to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device)["model_state"])
    model.eval()

    tfm = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((FASHION_MEAN,), (FASHION_STD,)),
    ])
    x = tfm(Image.open(args.image)).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1).squeeze()
    conf, pred = probs.max(0)
    print(f"Predicted: {CLASS_NAMES[pred.item()]}  (p={conf.item():.4f})")
    for i, p in enumerate(probs.tolist()):
        print(f"  {CLASS_NAMES[i]:<12} {p:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True, help="путь к изображению .png/.jpg")
    p.add_argument("--weights", default="models/best_cnn.pth")
    p.add_argument("--cpu", action="store_true")
    main(p.parse_args())