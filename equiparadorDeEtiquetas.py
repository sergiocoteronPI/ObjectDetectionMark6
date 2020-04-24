
import numpy as np 
import os
import cv2

from auxiliarTrain import leerDatosTXT
from classCOCODetec import classCOCODetec
print(classCOCODetec.labels)
print("")

imageLabelNombre = leerDatosTXT(ruta = classCOCODetec.rpe)

ArrayNombres = []
for _ in range(len(classCOCODetec.labels)):
    ArrayNombres = ArrayNombres + [[]]


quienVale = []

labelsPermitidas = [v for v in range(1,len(classCOCODetec.labels))]

contadorDeEtiquetasTotal = np.zeros(len(classCOCODetec.labels))
contadorDeEtiquetasModificado = np.zeros(len(classCOCODetec.labels))
for name, cont in zip(imageLabelNombre, range(len(imageLabelNombre))):

    contadorDeEtiquetasImg = np.zeros(len(classCOCODetec.labels))

    vector = []
    with open(name, 'r') as f:
        for line in f:
            linea = line.rstrip('\n').split(',')
            if linea[1] in classCOCODetec.labels:
                contadorDeEtiquetasImg[classCOCODetec.labels.index(linea[1])] += 1
                vector.append(linea)
    
    for v in range(len(contadorDeEtiquetasImg)):
        contadorDeEtiquetasTotal[v] += contadorDeEtiquetasImg[v]

    NoContinuar = True
    for v in labelsPermitidas:
        if contadorDeEtiquetasImg[v] > 0:
            NoContinuar = False
            break

    if NoContinuar:
        continue

    for v in range(len(contadorDeEtiquetasImg)):
        contadorDeEtiquetasModificado[v] += contadorDeEtiquetasImg[v]

    quienVale.append(name)

print("")
etiquetasValidas = "["
for etiqueta, v, w in zip(classCOCODetec.labels, contadorDeEtiquetasTotal, contadorDeEtiquetasModificado):

    if int(v) != 0:
        etiquetasValidas += "'" + etiqueta + "',"
    print(etiqueta + ":", int(v), int(w))

print("")
print(etiquetasValidas)

print("")
print(" ############################################################################################## ")
print("")


with open("etiquetasEquiparadas.txt", "w") as f:
    for line in quienVale:
        f.write(line + ",")
