import torch
import clip
from PIL import Image

# Run model on cpu or GPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load clip model
model, preprocess = clip.load("ViT-B/32", device=device)

# Define category descriptions
categories = ["A car on the road", "A car in the desert", "A boat on water", "A lion in the jungle"]
text = clip.tokenize(categories).to(device)

# Define function to predict new images
def predict(image_path):

    # preprocess the image
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

    with torch.no_grad():
        
        # find consine similarity of the image to the categories
        logits_per_image, logits_per_text = model(image, text)
    
    # retrieve index of category with highest cosine similarity
    class_index = logits_per_image.argmax().item()

    # retrieve score of best class
    logit_score = logits_per_image.max().item()

    class_description = categories[class_index]

    print(f"Category: {class_description}, Score: {logit_score}")

if __name__ == "__main__":

    predict("../images/image1.jpg")
    predict("../images/image2.jpg")
    predict("../images/image3.jpg")
    predict("../images/image4.jpg")