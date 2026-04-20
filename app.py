import streamlit as st
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as tr

@st.cache_resource
def load_model():
    package_file = "package.pt"
    model_file = "gan_model.pt"
    model = torch.jit.load(model_file)
    model.eval()
    with open(package_file, "rb") as f:
        package = torch.load(f, weights_only=False)
    return model, package

model, package = load_model()
transform_a = package['transform_a']
transform_b = package['transform_b']
generator_a = model.generator_a
generator_b = model.generator_b

mean = torch.tensor([0.5, 0.5, 0.5])
std = torch.tensor([0.5, 0.5, 0.5])
def de_normalize(tensor: torch.Tensor):
    return tr.ToPILImage()(tensor.cpu() * std.reshape(3, 1, 1) + mean.reshape(3, 1, 1))

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

button_beard = st.button("Shave a beard!")
button_no_beard = st.button("Put a beard!")

if uploaded_file is not None and (button_beard or button_no_beard):
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    if button_beard:
        img_transformed = transform_a(np.asarray(image))
        output = generator_a(img_transformed.unsqueeze(0)).squeeze(0)
    else:
        img_transformed = transform_b(np.asarray(image))
        output = generator_b(img_transformed.unsqueeze(0)).squeeze(0)
    output_image = de_normalize(output)
    st.image(output_image, caption='Output image', use_container_width=True)

