
import tensorflow as tf

import numpy as np
import cv2
import os

from copy import deepcopy

from classCOCODetec import classCOCODetec
from neuralnetwork import lossFunction, neuralNetwork

try:
    font = cv2.FONT_HERSHEY_SIMPLEX
except:
    print("Error: No se ha podido ejecutar - cv2.FONT_HERSHEY_SIMPLEX")

class BoundBox:
    def __init__(self, classes):
        self.x, self.y = float(), float()
        self.w, self.h = float(), float()
        self.label = ''
        self.probs = float()

class ClassTestCOCODetec():

    def __init__(self):

        if os.path.exists(classCOCODetec.h5):

            print('')
            print('Cargando modelo...')
            print('')
            self.model = tf.keras.models.load_model(classCOCODetec.h5, custom_objects={'lossFunction': self.lossFunction})
            
        else:

            print('')
            print('No ha sido posible cargar el modelo...')
            print('')

            self.model, _ = self.neuralNetwork()
            self.model.compile(loss=self.lossFunction,optimizer=tf.keras.optimizers.Adam(lr = classCOCODetec.learningRatio))

        print('')
        print(self.model.summary())
        print('')

    def COCODetecFunction(self, imagen):

        _imagen = deepcopy(imagen)

        try:
            origShapeY, origShapeX, _ = _imagen.shape
            
            multY, multX = origShapeY, origShapeX
            if origShapeY < classCOCODetec.dim_fil:
                multY = classCOCODetec.dim_fil
            if origShapeX < classCOCODetec.dim_col:
                multX = classCOCODetec.dim_col
        except:
            
            return None, []

        frameAdaptado = self.retocar(_imagen)
        frameNormalizado = self.normalizar_imagen(frameAdaptado)

        neuralNetworkOut = self.model.predict(x=np.array([frameNormalizado]))
        box, imgOut = self.postprocess(neuralNetworkOut, imagen, multY, multX)


        return imgOut, box


    def leerImagenesRuta(self, ruta):
    
        imagePath = []

        for ruta, _, ficheros in os.walk(ruta):
            for nombre_fichero in ficheros:
                rut_comp = os.path.join(ruta, nombre_fichero)
                
                if(rut_comp.endswith("jpg") or rut_comp.endswith("png") or rut_comp.endswith("JPG") or rut_comp.endswith("jpeg")):
                    imagePath.append(rut_comp)

        return imagePath

    # Normalización de la imagen en caso de entrenar de este modo #
    # =========================================================== #
    def normalizar_imagen(self, img):
        return img/255 * 2 - 1
    # =========================================================== #

    # Pre procesamiento de la imagen para darsela a la red neuronal #
    # ============================================================= #
    def retocar(self, img):
    
        zeros = np.zeros([classCOCODetec.dim_fil,classCOCODetec.dim_col,3])
        im_sha_1, im_sha_2, _ = img.shape
        if im_sha_1 >= classCOCODetec.dim_fil:
            if im_sha_2 >= classCOCODetec.dim_col:
                try:
                    zeros = cv2.resize(img,(classCOCODetec.dim_col,classCOCODetec.dim_fil))
                except:
                    return None
            else:
                try:
                    zeros[:,0:im_sha_2,:] = cv2.resize(img,(im_sha_2,classCOCODetec.dim_fil))
                except:
                    return None
        elif im_sha_2 >= classCOCODetec.dim_col:
            try:
                zeros[0:im_sha_1,:,:] = cv2.resize(img,(classCOCODetec.dim_col,im_sha_1))
            except:
                return None
        else:
            zeros[0:im_sha_1, 0:im_sha_2,:] = img
        return zeros
    # ============================================================= #

    # Funcion pérdida y función para calcular la intersección sobre la unión #
    # ====================================================================== #
            
    def lossFunction(self, yTrue, yPred):

        return lossFunction(yTrue, yPred)

    # ====================================================================== #

    # Red neuronal para la deteción de matrículas en imágenes. También están aquí las funciones necesarias para que funcione #
    # ====================================================================================================================== #

    def neuralNetwork(self):

        return neuralNetwork()

    # ====================================================================================================================== # 


    # Post procesamiento. Tomamos lo devuelto por la red neuronal y lo convertimos en una imagen de salida y un array con los datos encontrados #
    # ========================================================================================================================================= #

    def overlap_c(self, x1, w1 , x2 , w2):
        l1 = x1 - w1 /2.
        l2 = x2 - w2 /2.
        left = max(l1,l2)
        r1 = x1 + w1 /2.
        r2 = x2 + w2 /2.
        right = min(r1, r2)
        return right - left

    def box_intersection_c(self, ax, ay, aw, ah, bx, by, bw, bh):
        w = self.overlap_c(ax, aw, bx, bw)
        h = self.overlap_c(ay, ah, by, bh)
        if w < 0 or h < 0: return 0
        area = w * h
        return area

    def box_union_c(self, ax, ay, aw, ah, bx, by, bw, bh):
        i = self.box_intersection_c(ax, ay, aw, ah, bx, by, bw, bh)
        u = aw * ah + bw * bh -i
        return u

    def box_iou_c(self, ax, ay, aw, ah, bx, by, bw, bh):
        return self.box_intersection_c(ax, ay, aw, ah, bx, by, bw, bh) / self.box_union_c(ax, ay, aw, ah, bx, by, bw, bh)

    def expit_c(self, x):
        return 1/(1+np.exp(-np.clip(x,-10,10)))
        

    def NMS(self, final_probs , final_bbox):

        labels, C = classCOCODetec.labels, classCOCODetec.C
        
        boxes = []
        indices = []
    
        pred_length = final_bbox.shape[0]
        class_length = final_probs.shape[1]

        for class_loop in range(class_length):
            for index in range(pred_length):
                if final_probs[index,class_loop] == 0: continue
                
                for index2 in range(index+1,pred_length):
                    if final_probs[index2,class_loop] == 0: continue
                    if index==index2 : continue
                    
                    if self.box_iou_c(final_bbox[index,0],final_bbox[index,1],final_bbox[index,2],final_bbox[index,3],
                                      final_bbox[index2,0],final_bbox[index2,1],final_bbox[index2,2],final_bbox[index2,3]) >= 0.1:
                        if final_probs[index2,class_loop] > final_probs[index, class_loop] :
                            final_probs[index, class_loop] = 0
                            break
                        final_probs[index2,class_loop]=0
                if index not in indices:

                    bb=BoundBox(C)
                    bb.x = final_bbox[index, 0]
                    bb.y = final_bbox[index, 1]
                    bb.w = final_bbox[index, 2]
                    bb.h = final_bbox[index, 3]

                    bb.label = labels[class_loop]
                    bb.probs = final_probs[index,class_loop]

                    boxes.append(bb)
                    
                    indices.append(index)
                    
        return boxes


    def new_NMS(self, obj):
    
        if obj == []:
            return []
        
        sizeObjetos = len(obj)
        
        indices = []
        for index in range(sizeObjetos):
            for index2 in range(index+1, sizeObjetos):

                if (index2 in indices) or (index in indices): continue 

                if self.box_iou_c(obj[index].x,obj[index].y,obj[index].w,obj[index].h,obj[index2].x,obj[index2].y,obj[index2].w,obj[index2].h) > 0.1 and obj[index].clase == obj[index2].clase:
                    if obj[index].prob > obj[index2].prob:
                        indices.append(index2)
                    else:
                        indices.append(index)

        newObjetos = []
        for index in range(sizeObjetos):
            if index in indices: continue
            newObjetos.append(obj[index])

        
        return newObjetos
        
    def box_constructor(self, net_out_in):
        
        threshold = classCOCODetec.threshold
        anchors = classCOCODetec.anchors

        H, W = classCOCODetec.H, classCOCODetec.W
        C, B = classCOCODetec.C, classCOCODetec.B

        boxes = []
        
        probs = np.zeros((H, W, C), dtype=np.float32)
        _Bbox_pred = np.zeros((H, W, C, 5), dtype=np.float32)
        
        Classes = net_out_in[:,:,:,:C].reshape([H, W, C])
        Confs_pred = net_out_in[:,:,:,C: C + B*C].reshape([H, W, C, B])
        Bbox_pred = net_out_in[:,:,:,C + B*C:].reshape([H, W, C, B, 4])
        
        for row in range(H):
            for col in range(W):

                Classes[row, col, :] = self.expit_c(Classes[row, col, :])
                if np.max(Classes[row, col, :]) < threshold:
                    continue
                
                Confs_pred[row, col, :, :] = self.expit_c(Confs_pred[row, col, :, :])
                if np.max(Confs_pred[row, col, :, :]) < threshold:
                    continue

                for class_loop in range(C):
                    
                    for boxLoop in range(B):

                        tempc = Classes[row, col, class_loop] * Confs_pred[row, col, class_loop, boxLoop]
                        if(tempc < threshold):
                            continue
                        
                        bb=BoundBox(C)

                        bb.x = (col + self.expit_c(Bbox_pred[row, col, class_loop, boxLoop, 0])) / W
                        bb.y = (row + self.expit_c(Bbox_pred[row, col, class_loop, boxLoop, 1])) / H
                        bb.w = np.exp(np.clip(Bbox_pred[row, col, class_loop, boxLoop, 2],-15,8)) * anchors[2 * boxLoop + 0] / W
                        bb.h = np.exp(np.clip(Bbox_pred[row, col, class_loop, boxLoop, 3],-15,8)) * anchors[2 * boxLoop + 1] / H

                        bb.prob = tempc

                        bb.clase = class_loop
                        bb.box = boxLoop

                        boxes.append(bb)

        return self.new_NMS(boxes)

    def box_constructor_sin_nms(self, net_out_in):

        threshold = classCOCODetec.threshold
        anchors = classCOCODetec.anchors

        H, W = classCOCODetec.H, classCOCODetec.W
        C, B = classCOCODetec.C, classCOCODetec.B
        
        boxes = []

        Classes = net_out_in[:,:,:,:C].reshape([H, W, C])
        Confs_pred = net_out_in[:,:,:,C: C + B*C].reshape([H, W, C, B])
        Bbox_pred = net_out_in[:,:,:,C + B*C:].reshape([H, W, C, B, 4])
        
        for row in range(H):
            for col in range(W):

                Classes[row, col, :] = self.expit_c(Classes[row, col, :])
                if np.max(Classes[row, col, :]) < threshold:
                    continue
                
                Confs_pred[row, col, :, :] = self.expit_c(Confs_pred[row, col, :, :])
                if np.max(Confs_pred[row, col, :, :]) < threshold:
                    continue

                for class_loop in range(C):
                    
                    for boxLoop in range(B):

                        tempc = Classes[row, col, class_loop] * Confs_pred[row, col, class_loop, boxLoop]
                        if(tempc < threshold):
                            continue

                        bb=BoundBox(C)

                        bb.x = (col + self.expit_c(Bbox_pred[row, col, class_loop, boxLoop, 0])) / W
                        bb.y = (row + self.expit_c(Bbox_pred[row, col, class_loop, boxLoop, 1])) / H
                        bb.w = np.exp(np.clip(Bbox_pred[row, col, class_loop, boxLoop, 2],-15,8))* anchors[2 * boxLoop + 0] / W
                        bb.h = np.exp(np.clip(Bbox_pred[row, col, class_loop, boxLoop, 3],-15,8))* anchors[2 * boxLoop + 1] / H

                        bb.prob = tempc

                        bb.clase = class_loop
                        bb.box = boxLoop

                        boxes.append(bb)

        return boxes


    def findboxes(self, net_out):
        
        boxes = []
        if classCOCODetec.nms:
            boxes = self.box_constructor(net_out)
        else:
            boxes = self.box_constructor_sin_nms(net_out)
        
        return boxes


    def process_box(self, b, h, w):
    
        left  = int ((b.x - b.w/2.) * w)
        right = int ((b.x + b.w/2.) * w)
        top   = int ((b.y - b.h/2.) * h)
        bot   = int ((b.y + b.h/2.) * h)
        if left  < 0    :  left = 0
        if right > w - 1: right = w - 1
        if top   < 0    :   top = 0
        if bot   > h - 1:   bot = h - 1

        return (left, right, top, bot, classCOCODetec.labels[b.clase], b.prob)


    def postprocess(self, net_out, im, h, w):

        labels = classCOCODetec.labels
        colors = classCOCODetec.colors

        boxes = self.findboxes(net_out)
        
        imgcv = im.astype('uint8')

        resultsForJSON = []
        for b in boxes:
            
            boxResults = self.process_box(b, h, w)
            if boxResults is None:
                continue
            
            left, right, top, bot, mess, confidence = boxResults
            resultsForJSON.append({"label": mess, "confidence": float('%.2f' % confidence), "topleft": {"x": left, "y": top}, "bottomright": {"x": right, "y": bot}})

            try:
                cv2.rectangle(imgcv,(left, top), (right, bot),colors[labels.index(mess)], 2)
            except:
                print("los cv2 en try-except")
                
            confi = confidence*100

            if classCOCODetec.verProbs:
                if top - 16 > 0:
                    try:
                        cv2.rectangle(imgcv,(left-1, top - 16), (left + (len(mess)+9)*5*2-1, top),colors[labels.index(mess)], -1)
                        cv2.putText(imgcv,mess + ' -> ' + "%.2f" % confi  + '%' ,(left, top - 4), font, 0.45,(0,0,0),0,cv2.LINE_AA)
                    except:
                        print("los cv2 en try-except")
                else:
                    try:
                        cv2.rectangle(imgcv,(left-1, top), (left + (len(mess)+9)*5*2-1, top+16),colors[labels.index(mess)], -1)
                        cv2.putText(imgcv,mess + ' -> ' + "%.2f" % confi  + '%' ,(left, top + 12), font, 0.45,(0,0,0),0,cv2.LINE_AA)
                    except:
                        print("los cv2 en try-except")
            else:
                if top - 16 > 0:
                    try:
                        cv2.rectangle(imgcv,(left-1, top - 16), (left + len(mess)*5*2-1, top),colors[labels.index(mess)], -1)
                        cv2.putText(imgcv,mess,(left, top - 4), font, 0.45,(0,0,0),0,cv2.LINE_AA)
                    except:
                        print("los cv2 en try-except")
                else:
                    try:
                        cv2.rectangle(imgcv,(left-1, top), (left + len(mess)*5*2-1, top+16),colors[labels.index(mess)], -1)
                        cv2.putText(imgcv,mess,(left, top + 12), font, 0.45,(0,0,0),0,cv2.LINE_AA)
                    except:
                        print("los cv2 en try-except")

        return resultsForJSON, imgcv

    # ========================================================================================================================================= #

testCOCODetec = ClassTestCOCODetec()

#imagenesPath = testCOCODetec.leerImagenesRuta(classCOCODetec.rpi + "train2017/")
#imagenesPath = testCOCODetec.leerImagenesRuta("/home/sergio/Escritorio/20200311_movi/")

with open("etiquetasEquiparadas.txt", "r") as f:
    imageLabelNombre = f.readline().split(",")[:-1] 

imagenesPath = []
for name in imageLabelNombre:

    with open(name, 'r') as f:
        linea = f.readline().rstrip('\n').split(',')
        imagenesPath.append(classCOCODetec.rpi + linea[0])

print(imagenesPath[0])
             
from random import shuffle
shuffle(imagenesPath)

for name in imagenesPath:

    frame = cv2.imread(name)
    imgOut, _ = testCOCODetec.COCODetecFunction(frame)

    shapey, shapex, _ = imgOut.shape
    if shapex > 1000:
        newShapex = 1000
        imgOut = cv2.resize(imgOut, (1000, shapey - 1000*int(shapey/1000)))

    cv2.imshow("img", imgOut)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break

#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("/home/sergio/Vídeos/londonDrive.mp4")

while True:

    _, frame = cap.read()
    
    imgOut, _ = testCOCODetec.COCODetecFunction(frame)

    cv2.imshow("img", imgOut)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
