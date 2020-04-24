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

# De dónde y cómo obtener los datos.

He usado las imágenes de http://cocodataset.org/#home.

Para obtener los datos utilizados se accede a http://cocodataset.org/#download y se descargan las imágenes:

* http://images.cocodataset.org/zips/train2017.zip
* http://images.cocodataset.org/zips/val2017.zip

A continuación descargamos las etiquetas.

* http://images.cocodataset.org/annotations/annotations_trainval2017.zip

Si hay alguna duda en la página del dataset COCO viene todo muy bien detallado, aún así aquí explicaremos como convertir los datos para que los pueda leer el programa.


# Como transformar los datos para entrenar.

El programa lee txt. Para cada imagen se crea un txt con la siguiente estructura.

nombreImagen.txt

rutaImagen,etiqueta1,coordx1,coordy1,coordw1,coordh1\n

rutaImagen,etiqueta2,coordx2,coordy2,coordw2,coordh2\n

    .        .        .

# Como trabaja el modelo.

Como ya se ha comentado el modelo está inspirado en YOLO. YOLO toma una imagen y la divide en regiones que llamaremos celdas. La red neuronal se encarga de transformar una imagen de tamaño 416x416 en 13x13. Para cada celda tendremos b cajas posibles y dentro de cada caja encontraremos un vector de probabilidades de clase, otro vector de coordenadas y por ultimo un número entre 0 y 1 que nos indicará como de certeras son para esa caja las coordenadas predichas.

* Recomendar alguna página del funcionamiento de YOLO.

Lo que he hecho ha sido modificar la estructura. YOLO predice un total de H x W x B*(1 + 4 + C) predicciones pues he transformado esto en H x W x C*(1 + B*(1+4)).

Es decir para una imagen dividida en H x W celdas se crea una sola capa que prediga la clase.

Va a ser mejor que crees una imágenes Sergio para explicarte de palabra es muy complicado para ti.

# Resultados obtenidos.
