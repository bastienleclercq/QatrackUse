from pylinac import CatPhan504
from catpy504 import fonctionglob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
from scipy.integrate import simps
import math

class CBCT :
    def __init__(self, cbct : CatPhan504):
        self.cbct = cbct

        self.img528 = self.cbct.ctp528.mtf
        self.img486 = self.cbct.ctp486.image.array
        self.img515 = self.cbct.ctp515.image.array
        self.img404 = self.cbct.ctp404.image.array
        self.centre = [int(self.cbct.ctp486.phan_center.x), int(self.cbct.ctp486.phan_center.y)]
        self.PS = self.cbct.mm_per_pixel
        self.Clim = None
        self.hufond = None

    def noise_water(self) -> float:
        mask = np.zeros_like(self.img486, dtype=np.uint8)
        rayon = int(self.cbct.catphan_radius_mm / self.cbct.mm_per_pixel * 0.4) # Convertir le rayon en entier
        cv2.circle(mask, (self.centre[0], self.centre[1]), rayon, 255, -1)
        img_cercle = self.img486[mask == 255]
        noise = np.std(img_cercle)
        self.hufond = np.mean(img_cercle)
        SSB = self.hufond/noise
        self.Clim = abs(noise - self.hufond) /self.hufond * 100
        return SSB, self.hufond

    def Uniformite(self) -> list:
        Rois = self.cbct.ctp486.roi_settings
        Hu_value = []
        for roi in Rois:
            Roi = Rois[roi]
            mask = fonctionglob.create_circle(Roi, self.img486, self.centre, self.PS)
            imgT = self.img486[mask == 255]
            Hu_value.append(np.mean(imgT))
            ImgT = []
            #fonctionglob.draw_diff_image(self.img486 * mask, self.img486)
        return Hu_value

    def puissance_noise(self):
        #fonctionglob.draws_image(self.img486)
        Roitop = self.img486[131:181, 231:281]
        Roibot = self.img486[231:281, 231:281]
        Roidroite = self.img486[231:281, 131:181]
        Roigauche = self.img486[231:281, 331:381]

        Rois = [Roitop, Roigauche, Roibot, Roidroite]
        result = 0
        for roi in Rois:
            mean = np.mean(roi)
            diff = roi-mean
            fft_Roi=np.fft.fft2(diff)
            power_spectrum = np.abs(fft_Roi) ** 2
            fft_shifted = np.fft.fftshift(power_spectrum)
            result +=  fft_shifted
        NPSb = result/4*(len(Rois))  * (self.PS*self.PS) / (Roitop.shape[0] * Roitop.shape[1])
        NPS = fonctionglob.radial_profile(NPSb)
        dim_freq_roi = int((Roigauche.shape[0]-1)/2)
        x_freq= np.zeros(dim_freq_roi)
        for i in range(dim_freq_roi):
            x_freq[i] = i / (dim_freq_roi * self.PS * 2.0)
        x_freq = np.array(x_freq)
        ########CHECK#####################
        #plt.plot(x_freq, NPS)       #
        #plt.show()                      #
        #fonctionglob.draws_image(NPSb)#
        ##################################
        area = simps(NPS, x_freq)
        return area

    def test(self):
        mask = np.zeros_like(self.img486, dtype=np.uint8)
        rayon = int(self.cbct.catphan_radius_mm / self.cbct.mm_per_pixel*0.9) # Convertir le rayon en entier
        cv2.circle(mask, (self.centre[0], self.centre[1]), rayon, 1, -1)
        img = self.img515 * mask
        fonctionglob.draws_image(img)

    def reso_spatial(self):
        Rois = self.cbct.ctp515.roi_settings
        Hu_value = []
        #fonctionglob.draws_image(self.img515)
        for roi in Rois :
            Roi = Rois[roi]
            mask = fonctionglob.create_circle(Roi, self.img515, self.centre, self.PS)
            imgT = self.img515[mask == 255]
            contrastes = (np.mean(imgT - self.hufond) / self.hufond) * 100
            if contrastes < self.Clim:
                break
            else:
                ImgT = []
            ######Check Erreur position######
            #imgBug = self.img515 * mask    #
            # Hu_value.append(np.mean(imgT))#
            #if roi == '7':                 #
            #    print('bug')               #
            #################################
        return roi

    def linearite_space(self):
        lignes = self.cbct.ctp404.lines
        distance = []
        for idx, lane in enumerate(lignes):
            ligne = lignes[lane]
            l = math.sqrt((ligne.point1.x-ligne.point2.x)**2+(ligne.point1.y-ligne.point2.y)**2)
            distance.append(l)
        distance_moy = np.mean(distance)
        distorsion = ((max(distance) - min(distance))/distance_moy)*100
        return distance_moy, distorsion

    def disto_radial(self):
        # normaliser l'image pour cv2
        normalized_img = cv2.normalize(self.img404, None, 0, 255, cv2.NORM_MINMAX)
        normalized_img = cv2.convertScaleAbs(normalized_img)

        #détection de tous le contous
        edges = cv2.Canny(normalized_img, threshold1=100, threshold2=200)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        exterior_contour = max(contours, key=cv2.contourArea)

        # remplissage binaire
        exterior_image = np.zeros_like(edges)
        cv2.drawContours(exterior_image, [exterior_contour], -1, 255, thickness=cv2.FILLED)
        final = cv2.Canny(exterior_image, threshold1=100, threshold2=200)

        white_pixels = np.argwhere(final == 255)
        diametre = []
        for i in range(10):
            random_index = random.randint(0, len(white_pixels) - 1)
            random_coords = tuple(white_pixels[random_index])
            distance = 2*np.sqrt((random_coords[0] - self.centre[0]) ** 2 + (random_coords[1] - self.centre[1]) ** 2)
            diametre.append(distance)
        dist_radM = np.mean(diametre)
        dist_rad = (max(diametre)-min(diametre))/dist_radM * 100
        return dist_radM, dist_rad


    def result(self):
        SSB, huwater = self.noise_water()
        Hu_value = self.Uniformite()
        linearite_space, distorcion_space = self.linearite_space()
        calcul_rad, distorcion_rad = self.disto_radial()
        result = {
            'SSB': SSB,
            'HU eau': huwater,
            'Uniformite': {
                'Top': Hu_value[0],
                'Droite': Hu_value[1],
                'Bot': Hu_value[2],
                'Gauche': Hu_value[3],
                'Mid': Hu_value[4],
            },
            'Spectre de puissance': self.puissance_noise(),
            'Spatial': {
                'Résolution spatial': self.reso_spatial(),
                'linéarité spacial': linearite_space,
                'distorsion spacial': distorcion_space,
            },
            'Radial': {
                'Calcul radial': calcul_rad,
                'distorsion radial': distorcion_rad,
            },
            'Epaisseur de coupe' : self.cbct.ctp404.meas_slice_thickness,
            'Resolution spatiale HC': {
                'MTF 2%': self.img528.relative_resolution(x=2)*10,
                'MTF10%': self.img528.relative_resolution(x=10)*10,
                'MTF50%': self.img528.relative_resolution(x=50)*10
            },
        }
        return result
