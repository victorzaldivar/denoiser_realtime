# denoiser_realtime

Éste repositorio proporciona una implementación de PyTorch para realizar denoising en tiempo real. 

El algoritmo propuesto está basado en la arquitectura Sudo RM-Rf para la separación de fuentes, pero está aplicado para que pueda ejecutarse en tiempo real a través de la GPU de los porátiles Apple Silicon con procesador M1.

## Instalación
Primero, instalar Python 3.7 (recomendado con Anaconda).


#### A través de  pip (es solamente para descargar el modelo pre-entrenado listo para usar)
Solamente hay que hacer ejecutar des de la terminal
```bash
git clone https://github.com/victorzaldivar/denoiser_realtime.git
cd denoiser_realtime
pip install -r requirements.txt  # If you don't have cuda
pip install -r requirements_cuda.txt  # If you have cuda
```


```bash
git clone https://github.com/victorzaldivar/denoiser_realtime.git
cd denoiser_realtime
pip install -r requirements.txt  # If you don't have cuda
pip install -r requirements_cuda.txt  # If you have cuda
```

## Para poder utilizar el modelo en tiempo real

Para ejecutar el modelo `denoiser_realtime` en tiempo real (por ejemplo para una videoconferencia), necesitaremos una interfaz de audio loopback específica.

### Mac OS X

Para Mac OS X, hay que utilizar la interfaz [BlackHole](https://existential.audio/blackhole/).
Hay que intalarse BlackHole, y después en el terminal hay que ejecutar:

```bash
python -m denoiser.live
```

Para utilizarlo en tus videollamadas, solamente hay ejecutar el modelo y escoger "BlackHole" como input para disfrutar de la eliminación del ruido. 

