import mtcnn
import matplotlib.pyplot as plt
from mtcnn import MTCNN
import cv2
from PIL import Image
import numpy as np
import os

#Folders for I/O
input_folder = 'test'
output_folder = 'output'

# extract a single face from a given photograph
def extract_face(pixels, required_size=(150, 150)):
    detector = MTCNN()
    results = detector.detect_faces(pixels)
    if len(results) > 0:
        x1, y1, width, height = results[0]['box']
        x2, y2 = x1 + width, y1 + height
        face = pixels[y1:y2, x1:x2]
        if len(face[0]) > 0:
            return cv2.resize(face, required_size)
    return []

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = plt.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

mask_images = load_images_from_folder(input_folder)
output = []
for image in mask_images:
    pixels = extract_face(image)
    output.append(pixels)

counter = 0
for image in output:
    name = 'img' + str(counter) + '.jpg'
    temp = os.path.join(output_folder, name)
    if(len(image) > 0):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(temp, np.array(image))
        counter = counter + 1