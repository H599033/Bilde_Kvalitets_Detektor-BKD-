import torch
import torch.nn.functional as F
import cv2

def is_blur(image_path):
    """
    This function convolves a grayscale image with
    a Laplacian kernel and calculates its variance.
    """
    image_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    tensor = torch.tensor(image_gray, dtype=torch.float32) / 255.0
    image = tensor.unsqueeze(0).unsqueeze(0) 
    
    threshold = 0.05

    # Laplacian kernel
    tesnsor = torch.Tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]])

    # Convolving image with Laplacian kernel
    new_img = F.conv2d(image, tesnsor.unsqueeze(0).unsqueeze(0), padding=1)

    # Calculating variance
    img_var = torch.var(new_img)
    print (f"Variance: {img_var}")

    return img_var.item()<threshold

#------------------------------------------TEST--------------------------------------------------------
#lag historgram#
# print(is_blur("Prosjekt/Resourses/Output_sources/Bilnr_3/_MG_6175.JPG"))
# Load and process the image using OpenCV
"""


    
image_path_uten_blur = "Prosjekt/Resourses/Temp_sources/temp_frame_1.png"
image_path_medlitt_blur = "Prosjekt/Resourses/Temp_sources/temp_frame_12.png"
image_path_med_blur = "Prosjekt/Resourses/Temp_sources/temp_frame_13.png"

image_uten_blur = cv2.imread(image_path_uten_blur, cv2.IMREAD_GRAYSCALE)
image_uten_blur = torch.tensor(image_uten_blur, dtype=torch.float32) / 255.0
image_uten_blur = image_uten_blur.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions

image_litt_blur = cv2.imread(image_path_medlitt_blur, cv2.IMREAD_GRAYSCALE)
image_litt_blur = torch.tensor(image_litt_blur, dtype=torch.float32) / 255.0
image_litt_blur = image_litt_blur.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions

image_mer_blur = cv2.imread(image_path_med_blur, cv2.IMREAD_GRAYSCALE)
image_mer_blur = torch.tensor(image_mer_blur, dtype=torch.float32) / 255.0
image_mer_blur = image_mer_blur.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions

result_uten_blur = is_blur(image_uten_blur)
result_litt_blur = is_blur(image_litt_blur)
result_mer_blur = is_blur(image_mer_blur)

print("Bilde uten blur =", result_uten_blur)
print("Bilde med litt blur =", result_litt_blur)
print("Bilde med mer blur =", result_mer_blur)
"""
# Test the function

#----------------------------------------Testing av threshold-----------------------------

# threshold trenger å være på ett nivå slik at den setter bilder uten MB (Motion blur) som false og alle andre bilder som true
# Hvis nivået er for lavt kan den sette Uklare(MB) bilder som klare 
# Hvis nivået er for høy kan den sette klare bilder som Uklare. 

# Testen kjøres på bilder som har 3 forskjlelige nivåer av MB