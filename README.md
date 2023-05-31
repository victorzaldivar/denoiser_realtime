Elimininación de ruido en tiempo real con el modelo [SudoRmRf](https://github.com/etzinis/sudo_rm_rf#pre-trained-models-and-easy-to-use-recipes).

Éste repositorio proporciona una implementación de PyTorch para realizar denoising en tiempo real. 

El algoritmo propuesto está basado en la arquitectura Sudo RM-Rf para la separación de fuentes, pero está aplicado para que pueda ejecutarse en tiempo real a través de la GPU de los porátiles Apple Silicon con procesador M1.

### Proceso de instalación 
Primero, instalar Python 3.7 (recomendado con Anaconda).


## Instalar Pytorch 
A través de la pàgina web oficial de [Pytorch](https://pytorch.org/), descargar la librería siguiendo los pasos que os indican a través del sistema operativo que vuestro Pc o portàtil disponga en lenguaje Python. 


## Instalar el repositorio
Hay que clonar este repositorio e instalar todas las dependencias. Recomendamos usar un entorno virtual.


## Creación del entorno virtual. 
Para crear un nuevo entorno virtual Conda para poder clonar el repositorio hay que seguir estos pasos.
```bash
conda create --name=test # En 'test' puedes poner el nombre que quieras del entorno virtual
```
```bash
conda activate test #Para activar el entorno virtual
```


Seguidamente, para instalar el repositorio, solamente hay que hacer ejecutar des de la terminal
```bash
git clone https://github.com/victorzaldivar/denoiser_realtime.git
cd denoiser_realtime
pip install -r requirements.txt  # Si no tienes cuda
pip install -r requirements_cuda.txt  #si tienes cuda
pip install -r reqs.txt
```

## Instalación de paquetes y dependencias
Para poder ejecutar el modelo, primero hay que instalar un seguido de paquetes y dependencias.
```bash
conda install pip
```
```bash
conda install -c conda-forge python-sounddevice
```
```bash
pip install torch
```
```bash
pip install pesq
```
```bash
pip install hydra-core --upgrade
```

## Para poder utilizar el modelo en tiempo real

Para ejecutar el modelo `denoiser_realtime` en tiempo real (por ejemplo para una videoconferencia), necesitaremos una interfaz de audio loopback específica.

## BlackHole para Mac OS X

Para Mac OS X, hay que utilizar la interfaz [BlackHole](https://existential.audio/blackhole/).
Hay que intalarse BlackHole, y después en el terminal hay que ejecutar:

```bash
python -m denoiser.live_rt
```

Para utilizarlo en tus videollamadas, solamente hay ejecutar el modelo y escoger "BlackHole" como input para disfrutar de la eliminación del ruido. Con el siguiente comando en la terminal se puede obtener una lista de dispositivos disponibles. 

```bash
python -m sounddevice
```

También es posible ejecutar el modelo eligiendo un dispositivo diferente a los dispositivos de entrada/salida predeterminados. En 'out y en 'in' podemos elegir el número correspondiente a la lista de dispositivos que tenemos disponibles. Aquí hay un ejemplo de cómo hacerlo.
```bash
python -m denoiser.live_rt --out 1 --in 0
```


