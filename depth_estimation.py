import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def estimate_depth(image_path):
    # Load DPT_Large model
    model_type = "DPT_Large"  # Can also try "DPT_Hybrid" for faster
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    midas.eval()

    # Load transform
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.dpt_transform

    # Load image
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)

    # Apply transform (returns [1,3,H,W])
    input_batch = transform(img_np)

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    midas.to(device)
    input_batch = input_batch.to(device)

    # Predict depth
    with torch.no_grad():
        depth = midas(input_batch)
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(1),
            size=img_np.shape[:2],
            mode="bicubic",
            align_corners=False
        ).squeeze()

    depth_map = depth.cpu().numpy()

    # Optional: visualize
    plt.imshow(depth_map, cmap='plasma')
    plt.title("Depth Map")
    plt.axis('off')
    plt.show()

    return depth_map, img
