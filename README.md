# Detección de objetos en imagenes mediante redes neuronales convolucionales.

Índice.

* De que trata este repositorio. (objetivo, programas necesarios...)
* De dónde y cómo obtener los datos.
* Como transformar los datos para entrenar.
* Como trabaja el modelo.
* Resultados obtenidos.

# De que trata el repositorio.

Voy a presentar mi modelo para la detección de objetos fuertemente inspirado en YOLO https://pjreddie.com/darknet/yolo/.

A ver si aprendes a referenciar artículos tío.
@article{yolov3,
  title={YOLOv3: An Incremental Improvement},
  author={Redmon, Joseph and Farhadi, Ali},
  journal = {arXiv},
  year={2018}
}

El modelo ha sido entrenado para las siguientes etiquetas extraidas de COCO: 

* person, bicycle, car, motorcycle, bus, truck, traffic light, stop sign, cat, dog, backpack, umbrella, handbag

Utilizaré redes neuronales convolucionales para encontrar objetos en imágenes. La detección se hará mediante cajas.

Se necesitará:

* python3.7.1
* tensorflow 1.13.2
* opencv
* Seguro que más cosas poco a poco

# ¿De dónde son los datos?

He usado las imágenes de http://cocodataset.org/#home

# ¿Que formato tienen los datos?
