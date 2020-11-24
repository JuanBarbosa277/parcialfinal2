import cv2
import os
from sklearn.cluster import KMeans
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt


class Bandera:
    def __init__(self,mensaje):
        self.path = 'C:/Users/Juan Sebastian/Documents/Pontificia Universidad Javeriana/Decimo Semestre/Procesamiento de Imagenes/Talleres/Taller4'
        self.image_name = mensaje
        self.path_file = os.path.join(self.path, self.image_name)
        self.image = cv2.imread(self.path_file)
        cv2.imshow("Imagen",self.image)
        cv2.waitKey(0)



    def Colores(self):
        image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        image = np.array(image, dtype=np.float64) / 255
        sw=False
        rows, cols, ch = image.shape
        assert ch == 3
        image_array = np.reshape(image, (rows * cols, ch))
        result  = np.ndarray(shape=(11,11), dtype=float)
        for i in range(0, len(result)):
            for j in range(0, len(result)):
                result[i][j] = 0
        graficar =  np.ndarray(shape=(4), dtype=float)
        vectores = np.ndarray(shape=(20), dtype=float)
        i=1
        k = 0
        z = 0
        for i in range(1,5,1):
            image_array_sample = shuffle(image_array, random_state=0)[:10000]
            model  = KMeans(n_clusters=i, random_state=0).fit(image_array_sample)
            modelo = KMeans(n_clusters=i, random_state=0).fit_transform(image_array_sample)
            centros = KMeans(n_clusters=i, random_state=0).fit_predict(image_array_sample)
            acum=0
            for p in np.arange(0,i):
                for j in np.arange(0,len(centros)):
                    if p==centros[j]:
                        acum=acum+modelo[j][p]
                print("El cluster: ", p+1, "la distancia es: ", acum)
                result[i][p]=acum
                vectores[k]=acum
                acum=0
        acum=0
        for i in range(1, 5):
            for j in range(0, len(result)):
                acum = acum + result[j][i]
            graficar[i-1] = acum
            acum = 0


        cv2.imshow("Bandera usuario", self.image)
        cv2.waitKey(0)









    def Porcentaje(self):
        image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        image = np.array(image, dtype=np.float64) / 255
        sw=False
        rows, cols, ch = image.shape
        assert ch == 3
        image_array = np.reshape(image, (rows * cols, ch))
        image_array_sample = shuffle(image_array, random_state=0)[:10000]
        centros = KMeans(n_clusters=4, random_state=0).fit_predict(image_array_sample)
        cont1=0
        cont2=0
        cont3=0
        cont4=0
        for i in np.arange(0,len(centros)):
            if centros[i] == 0:
                cont1=cont1+1
            if centros[i] == 1:
                cont2=cont2+1
            if centros[i] == 2:
                cont3 = cont3 + 1
            if centros[i] == 3:
                cont4 = cont4 + 1
            por1 =int(cont1/(cont1+cont1+cont1+cont1))*100
            por2 = int(cont1 / (cont1 + cont2 + cont3 + cont4))*100
            por3 = int(cont1 / (cont1 + cont2 + cont3 + cont4))*100
            por4 = int(cont1 / (cont1 + cont2 + cont3 + cont4))*100

        print("Porcentaje color 1: ", por1)
        print("Porcentaje color 1: ", por2)
        print("Porcentaje color 1: ", por3)
        print("Porcentaje color 1: ", por4)





    def Orientacion(self):
        print("Si Si")



#mensaje=input("Introduzca el nombre de la bandera: ")
mensaje="flag2.png"
parcial=Bandera(mensaje=mensaje)
parcial.Porcentaje()
