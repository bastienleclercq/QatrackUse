import cv2
import math
import numpy as np
import matplotlib.pyplot as plt

def create_circle(Roi : dict, img : np.ndarray, centre : list, PS : float) -> np.ndarray:
    mask = np.zeros_like(img, dtype=np.uint8)
    x = int(centre[0] + (Roi['distance'] * math.cos(math.radians(Roi['angle']))) / PS)
    y = int(centre[1] +  (Roi['distance'] * math.sin(math.radians(Roi['angle']))) / PS)
    cv2.circle(mask, (x, y), int(Roi['radius']), 255, -1)
    return mask

def distance_moy(images: list) -> float:
    distances = []
    objects_prev = cv2.findContours(images[0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    for image in images[1:]:
        objects_curr = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

        for obj_prev in objects_prev:
            for obj_curr in objects_curr:
                # Calculez la distance entre les centres des objets
                center_prev = np.mean(obj_prev, axis=0)[0]
                center_curr = np.mean(obj_curr, axis=0)[0]
                distance = np.linalg.norm(center_prev - center_curr)
                distances.append(distance)

        objects_prev = objects_curr

    distance_moyenne = np.mean(distances)
    return distance_moyenne


def radial_profile(data):
    Nx, Ny = data.shape
    M = int((Nx - 1) / 2)
    Xo = M
    Yo = M

    theta = np.linspace(0, 2.0 * np.pi, 360)
    dj = np.linspace(0, M - 1, M, dtype=int)

    radialprofile = np.zeros(M)
    for ang in theta:
        for j in dj:
            X = int(round(Xo + j * np.cos(ang)))
            Y = int(round(Yo + j * np.sin(ang)))

            radialprofile[j] += data[X][Y]

    return radialprofile / 360.

def draws_image(image : np.array) -> None:
    plt.imshow(image, cmap=plt.cm.bone)
    plt.show()

def draw_diff_image(image1 : np.array, image2 : np.array) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(image1, cmap='gray')
    ax2.imshow(image2, cmap='gray')
    plt.show()