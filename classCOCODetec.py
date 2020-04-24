
import numpy as np

class claseParametrosDeteccionDeObjetos:

    def __init__(self, threshold, batch_size, dim_fil, dim_col, H, W, B, learningRatio, nms, verProbs, rpe, rpi, h5):
        
        self.threshold = threshold
        self.batch_size = batch_size

        self.dim_fil = dim_fil
        self.dim_col = dim_col

        self.labels = ['person',
                       'bicycle','car','motorcycle','bus','truck',
                       'traffic light','stop sign',
                       'cat','dog',
                       'backpack','umbrella','handbag']
        
        self.anchors = [1,1, 2,1, 1,2, 7.77052,7.16828,  16.62,10.5][0:2*B]
        #self.anchors = [1,1, 1,1, 1,1, 1,1, 1,1, 1,1][0:2*B]

        self.H = H
        self.W = W
        self.C = len(self.labels)
        self.B = B
        self.HW = H*W

        self.colors = np.random.randint(0,255 ,(self.C,3)).tolist()
        self.colors[0] = [255,0,255]
        self.learningRatio = learningRatio

        self.nms = nms
        self.verProbs = verProbs

        self.clases_visibles = [self.labels.index(v) for v in self.labels]

        self.rpe = rpe
        self.rpi = rpi

        self.h5 = h5

classCOCODetec = claseParametrosDeteccionDeObjetos(threshold = 0.3,
                                            batch_size = 20,
                                            dim_fil = 480, dim_col = 480,
                                            H = 13, W = 13, B = 3,
                                            learningRatio = 1e-3,
                                            nms = True,
                                            verProbs = False,
                                            rpe = '../basededatos/labels_train', #ruta para etiquetas
                                            rpi = '../basededatos/',             #ruta para imágenes
                                            h5 = 'mark1_COCO.h5')
