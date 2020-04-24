
import numpy as np
import os
from random import shuffle

import tensorflow as tf

from classCOCODetec import classCOCODetec

from auxiliarTrain import leerDatosTXT, cargarLote
from neuralnetwork import neuralNetwork, lossFunction

class solo_nombres:
    def __init__(self, _imageLabelNombre):
        self.imageLabelNombre = _imageLabelNombre
            
with open("etiquetasEquiparadas.txt", "r") as f:
    imageLabelNombre = f.readline().split(",")[:-1] 

shuffle(imageLabelNombre)
sn = solo_nombres(imageLabelNombre)

if os.path.exists(classCOCODetec.h5):

    print("")
    print("Modelo: " + classCOCODetec.h5 + " encontrado")
    print("")
    print("Cargarndo modelo...")
    print("")

    #model = tf.keras.models.load_model(classCOCODetec.h5, custom_objects={'lossFunction': lossFunction})

    model, h_out = neuralNetwork()

    #model.compile(loss=lossFunction,optimizer=tf.keras.optimizers.Adam(lr = classCOCODetec.learningRatio))
    model.compile(loss=lossFunction,optimizer=tf.keras.optimizers.RMSprop(lr=classCOCODetec.learningRatio , rho=0.9,epsilon=None,decay=0.0))
    model.load_weights(classCOCODetec.h5)
    
else:

    model, h_out = neuralNetwork()

    #model.compile(loss=lossFunction,optimizer=tf.keras.optimizers.Adam(lr = classCOCODetec.learningRatio))
    model.compile(loss=lossFunction,optimizer=tf.keras.optimizers.RMSprop(lr=classCOCODetec.learningRatio,rho=0.9,epsilon=None,decay=0.0))

print('')
print(model.summary())
print('')


class MY_Generator(tf.keras.utils.Sequence):

    def __init__(self, image_filenames, labels, batch_size):

        self.image_filenames, self.labels = image_filenames, labels
        self.batch_size = batch_size

    def __len__(self):
        #return int(np.ceil(len(self.image_filenames) / (10*float(self.batch_size))))
        return int(np.ceil(len(self.image_filenames) / float(self.batch_size))) #Todo el dataset que haya en image_filenames

    def __getitem__(self, idx):

        batch_x = sn.imageLabelNombre[idx * self.batch_size:(idx + 1) * self.batch_size]
        #self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        image_train = []
        _yTrue = []
        
        for name in batch_x:
            try:
                _imagen_train, def_yTrue = cargarLote(classCOCODetec, [name])
            except:
                continue

            if _imagen_train == []:
                continue
            
            image_train.append(_imagen_train[0])
            _yTrue.append(def_yTrue[0])

        return np.array(image_train), np.array(_yTrue)

##### =================================================================================== #####
my_training_batch_generator = MY_Generator(sn.imageLabelNombre, None, classCOCODetec.batch_size)
##### =================================================================================== #####

"""
Código absurdo porque sino no me deja entrenar.
============================================================================================
"""
def preparar_unlote(image_filenames, batch_size):

    batch_x = image_filenames[:batch_size]
        
    image_train = []
    _yTrue = []
    
    for name in batch_x:
        try:
            _imagen_train, def_yTrue = cargarLote(classCOCODetec, [name])
        except:
            continue

        if _imagen_train == []:
            continue
        
        image_train.append(_imagen_train[0])
        _yTrue.append(def_yTrue[0])

    return np.array(image_train), np.array(_yTrue)

x_train_lote, y_train_lote = preparar_unlote(sn.imageLabelNombre, classCOCODetec.batch_size)
model.fit(x_train_lote, y_train_lote, verbose=1)
"""
============================================================================================
"""

while True:

    try:
        model.fit_generator(generator=my_training_batch_generator,
                            steps_per_epoch= int(len(sn.imageLabelNombre) / (classCOCODetec.batch_size)),
                            epochs=1,
                            verbose=1,
                            use_multiprocessing=True,
                            workers=4,
                            max_queue_size=10)

        print('')
        print(' ===== salvando modelo =====')
        print('')
                
        tf.keras.models.save_model(model, classCOCODetec.h5)
        
        shuffle(imageLabelNombre)
        
    except:

        print("")
        print("")
        guardar = input("Quieres guardar el modelo: ")
        print("")
        if guardar in ["s", "si", "y", "yes", "Y"]:

            print('')
            print(' ===== salvando modelo =====')
            print('')
            
            tf.keras.models.save_model(model, classCOCODetec.h5)
            break
        else:
            break