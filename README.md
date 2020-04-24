# Detección de objetos en imagenes mediante redes neuronales convolucionales.

Índice.

* De que trata este repositorio. (objetivo, programas necesarios...)
* De dónde y cómo obtener los datos.
* Como transformar los datos para entrenar.
* Como trabaja el modelo.
* Resultados obtenidos.

# De que trata el repositorio.

Voy a presentar mi modelo para la detección de objetos fuertemente inspirado en YOLO https://pjreddie.com/darknet/yolo/

@article{yolov3,
  title={YOLOv3: An Incremental Improvement},
  author={Redmon, Joseph and Farhadi, Ali},
  journal = {arXiv},
  year={2018}
}

Modelo entrenado en las etiquetas: 

* person, bicycle, car, motorcycle, bus, truck, traffic light, stop sign, cat, dog, backpack, umbrella, handbag

# ¿De dónde son los datos?

He usado las imágenes de http://cocodataset.org/#home

# ¿Que formato tienen los datos?
