# Preview
![Screenshot 2025-01-19 092712](https://github.com/user-attachments/assets/198c6817-945f-4876-861a-d65718be544b)
![Screenshot 2025-01-19 092953](https://github.com/user-attachments/assets/da0a70dd-ebf8-4c51-a25e-5756596f4e1f)

# How it works
1. User has to give the fandom name and his comic pages as input, the fandom's models are fetched from models directory if the name's models are available, in a much bigger system these models should be fetched from database.
2. The input images are resized to Sementation model's input shape and passed through it which generates a mask, this mask and resized image is used to get all character faces in that image.
3. These faces are then passed to fetched Classification model to identify characters.
4. The output characeters along with that image is shown.

# Assumptions
- It is assumed that there is no crossover of multiple verses in given comic, only a single fandom's models and character labels are fetched.
- Some characters are known by multiple names like Iron Man and Tony Stark, I have kept both labels seperate.

# Model Traning
- I have used  [VGG Image Annotator](https://www.robots.ox.ac.uk/~vgg/software/via/via_demo.html) to annotate comic pages and stored them in a json format.
- From annotations and comic pages masks are created around faces of characters, I also cropped and saved faces for classifier traning.
- For segmentation model, U-Net with pretrained VGG16 layers as Encoder were used, it was giving validation accuracy around `98%` but Iou was around `0.15`.
- Data augmentation was applied for images and masks to reduce overfitting.
- For classification pretrained VGG16 was used, it is giving `100%` accuracy within `10` epochs even with augmented faces data.
