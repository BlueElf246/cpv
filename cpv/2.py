from timm import *
model = pipeline("image-segmentation")
model("/Users/datle/Desktop/CPV/cpv/workshop4/watershed/cock_bird.jpeg")