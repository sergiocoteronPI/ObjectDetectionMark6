
import numpy as np
import cv2
import os

from random import shuffle
from copy import deepcopy

from classCOCODetec import classCOCODetec

def leerDatosTXT(ruta):

    image_label_nomb = []

    for ruta, _, ficheros in os.walk(ruta):
        for nombre_fichero in ficheros:
            rut_comp = os.path.join(ruta, nombre_fichero)
            
            if(rut_comp.endswith("txt")):
                image_label_nomb.append(rut_comp)

    return image_label_nomb

def retocar(self, img, bbox):
    
    zeros = np.zeros([self.dim_fil,self.dim_col,3])
    im_sha_1, im_sha_2, _ = img.shape
    
    if im_sha_1 >= self.dim_fil:
        if im_sha_2 >= self.dim_col:
            zeros = cv2.resize(img,(self.dim_col,self.dim_fil))
            for obj in bbox:
                obj[2],obj[1],obj[4],obj[3] = int(obj[2]*self.dim_fil/im_sha_1), int(obj[1]*self.dim_col/im_sha_2), int(obj[4]*self.dim_fil/im_sha_1), int(obj[3]*self.dim_col/im_sha_2)
        else:
            zeros[:,0:im_sha_2,:] = cv2.resize(img,(im_sha_2,self.dim_fil))
            for obj in bbox:
                obj[2],obj[4] = int(obj[2]*self.dim_fil/im_sha_1), int(obj[4]*self.dim_fil/im_sha_1)
    elif im_sha_2 >= self.dim_col:
        zeros[0:im_sha_1,:,:] = cv2.resize(img,(self.dim_col,im_sha_1))
        for obj in bbox:
            obj[1],obj[3] = int(obj[1]*self.dim_col/im_sha_2), int(obj[3]*self.dim_col/im_sha_2)
    else:
        zeros[0:im_sha_1, 0:im_sha_2,:] = img

    return zeros, bbox


def normalizar_imagen(img):

    return img/255 * 2 - 1

def iou(boxA, boxB):

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
 
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
 
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou

def _batch(self, img ,allobj):

    H, W = self.H, self.W
    B, C = self.B, self.C
    
    #anchors = self.anchors
    labels = self.labels
    
    cellx = 1. * self.dim_col / W
    celly = 1. * self.dim_fil / H

    y_true_etiqueta = np.zeros([H,W,C*(1+ B*(1+4))])
    for obj in allobj:

        if obj[0] not in labels:
            continue
        
        centerx = .5*(obj[1]+obj[3]) #xmin, xmax
        centery = .5*(obj[2]+obj[4]) #ymin, ymax
        cx = centerx / cellx
        cy = centery / celly

        if cx >= W or cy >= H: return []
        
        if obj[1] < 1:
            obj[1] = 1
        if obj[2] < 1:
            obj[2] = 1
        if obj[3] > self.dim_col:
            obj[3] = self.dim_col - 1
        if obj[4] > self.dim_fil:
            obj[4] = self.dim_fil - 1

        ### Ahora le asignamos una caja.

        objACaja = -1
        if (obj[3] - obj[1])/(obj[4] - obj[2]) > 0.8 and (obj[3] - obj[1])/(obj[4] - obj[2]) < 1.2:
            objACaja = 0
        elif (obj[3] - obj[1])/(obj[4] - obj[2]) <= 0.8:
            objACaja = 1
        elif (obj[3] - obj[1])/(obj[4] - obj[2]) >= 1.2:
            objACaja = 2

        if objACaja == -1:
            input("EEEh que ha salido -1")

        ##################################
            
        obj[3] = float(obj[3]-obj[1]) / self.dim_col
        obj[4] = float(obj[4]-obj[2]) / self.dim_fil

        for cog in range(3,5):
            if obj[cog] < 0:
                obj[cog] = 0.001
        
        obj[3] = np.sqrt(obj[3])
        obj[4] = np.sqrt(obj[4])
        
        obj[1] = cx - np.floor(cx) # centerx
        obj[2] = cy - np.floor(cy) # centery

        cuadradoReticula = int(np.floor(cy) * W + np.floor(cx))

        resto = int(cuadradoReticula%W)
        cociente = int((cuadradoReticula - resto) / W)

        posicionReticulaClase = labels.index(obj[0])
        
        y_true_etiqueta[cociente, resto, posicionReticulaClase] = 1.
        y_true_etiqueta[cociente, resto, C + B*posicionReticulaClase + objACaja] = 1.
        y_true_etiqueta[cociente, resto, C + B*C + posicionReticulaClase*B*4 + 4*objACaja: C + B*C + posicionReticulaClase*B*4 + 4*objACaja + 4] = obj[1:5]
    
    return normalizar_imagen(img), y_true_etiqueta

def aumentarBboxes(classCOCODetec, image, bboxes, IOUminimo = 0.8):

    H, W = classCOCODetec.H, classCOCODetec.W

    sha1, sha2, _ = image.shape

    cellx = 1. * sha2 / W
    celly = 1. * sha1 / H

    new_bboxes = deepcopy(bboxes)
    for box in bboxes:

        mess, left, top, right, bot = box[0], int(box[1]), int(box[2]), int(box[3]), int(box[4])

        centerx = .5*(left+right)
        centery = .5*(top+bot)
        cx = centerx / cellx
        cy = centery / celly
        
        ml_x, ml_y = (right - left)/2, (bot - top)/2

        for i, j in [(0,1), (1,1), (1,0), (0,-1), (-1,-1), (-1,0), (1,-1), (-1,1)]:
            
            if (cx + i) > 0 and (cx + i) < W and (cy + j) > 0 and (cy + j) < H:

                new_cx, new_cy = cx + i, cy + j

                if new_cx < cx:
                    new_cx += np.ceil(cx) - cx
                elif new_cx > cx:
                    new_cx -= cx  - np.floor(cx)

                
                if new_cy < cy:
                    new_cy += np.ceil(cy) - cy
                elif new_cy > cy:
                    new_cy -= cy  - np.floor(cy)

                new_left, new_right = int(new_cx*cellx - ml_x), int(new_cx*cellx + ml_x)
                new_top, new_bot = int(new_cy*celly - ml_y), int(new_cy*celly + ml_y)

                if iou([new_left, new_top, new_right, new_bot],[left, top, right, bot]) < IOUminimo:
                    continue

                new_bboxes.append([mess, new_left, new_top, new_right, new_bot])

    return new_bboxes
    
def leerImagenEnEsacalaDeGriseOEnColor(nombre):

    if True:#np.random.randint(10)%2:

        return cv2.imread(nombre)

    else:
        
        eg = cv2.imread(nombre, 0)
        largo, ancho = eg.shape

        eg_triple = np.zeros([largo, ancho,3])

        eg_triple[:,:,0] = eg
        eg_triple[:,:,1] = eg
        eg_triple[:,:,2] = eg

        return eg_triple.astype('uint8')


def retocarImagenCoordenadas(imagen, bboxes):

    sha1, sha2, _ = imagen.shape

    if np.random.randint(10)%2:
        sha_y,sha_x,_= imagen.shape
        noise = np.random.rand(sha_y,sha_x,3)
        imagen = imagen + noise*np.random.randint(3,10)

    #Suavizar imagen
    if np.random.randint(10)%2:
        imagen = cv2.filter2D(imagen,-1,np.ones((5,5),np.float32)/25)

    #Difuminada
    if np.random.randint(10)%2:
        imagen = cv2.blur(imagen,(3,3))

    #flip
    if np.random.randint(10)%2:

        imagen = cv2.flip(imagen, 1)
        for box in bboxes:
            aux = sha2 - deepcopy(box[1])
            box[1] = sha2 - box[3]
            box[3] = aux

    #crop
    if np.random.randint(10)%2:

        box = bboxes[np.random.randint(len(bboxes))]
        
        new_x = np.random.randint(0,box[1])
        new_y = np.random.randint(0,box[2])

        new_w = np.random.randint(box[3], sha2)
        new_h = np.random.randint(box[4], sha1)

        newbboxes = []
        for box in bboxes:

            box[1] = box[1] - new_x
            box[2] = box[2] - new_y

            box[3] = box[3] - new_x
            box[4] = box[4] - new_y

            if box[1] >= 0:
                if box[2] >= 0:
                    newbboxes.append([box[0],box[1],box[2],box[3],box[4]])
                elif box[4] > 1:
                    box[2] = 1
                    newbboxes.append([box[0],box[1],box[2],box[3],box[4]])
            elif box[3] > 1:
                box[1] = 1
                newbboxes.append([box[0],box[1],box[2],box[3],box[4]])

        bboxes = newbboxes
        imagen = imagen[new_y:new_h, new_x:new_w, :]

    #rotate

    return imagen, bboxes

def cargarLote(self, loteNombresTXT):

    imTrainArray = []
    yTrueArray = []

    for name in loteNombresTXT:
        
        vector = []
        with open(name, 'r') as f:
            for line in f:
                linea = line.rstrip('\n').split(',')
                if linea[1] in classCOCODetec.labels:
                    vector.append(linea)

        if vector == []:
            return [], []

        #Abrimos la imagen en blanco y negro o en color.
        image = leerImagenEnEsacalaDeGriseOEnColor(classCOCODetec.rpi + vector[0][0])

        bboxes = []
        for mini_vector in vector:
            bboxes.append([mini_vector[1], float(mini_vector[2]),float(mini_vector[3]),float(mini_vector[4]),float(mini_vector[5])])
        
        # Aquí se introduce la aumentación artifical de datos.
        if np.random.randint(100)%2 == 0:
            image, bboxes = retocarImagenCoordenadas(image, bboxes)

        # Se redimensiona la imagen al tamaño que admite la red neuronal (416x416, 512x512 ..... )
        image, bboxes = retocar(self, image, bboxes)
        
        # Este código aumenta las posibles cajas que puede tener cada etiqueta.
        if False:
            bboxes = aumentarBboxes(self, image, bboxes)

        # Por último se crea el ground truth y la imagen final.
        imgOutBatch, yTrueBatch = _batch(self, image, bboxes)
        
        imTrainArray.append(imgOutBatch)
        yTrueArray.append(yTrueBatch)

    return imTrainArray, yTrueArray