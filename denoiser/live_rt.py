# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# author: adefossez

import argparse
import sys

import sounddevice as sd
import torch

import causal_improved_sudormrf_v3 
from .utils import bold
from multiprocessing import Pool

import numpy as np
import time



def load_sudormrf_causal_cpu(model_path):
    # 1: declarem el model (instanciem la classe)
    model = causal_improved_sudormrf_v3.CausalSuDORMRF(
        in_audio_channels=1,
        out_channels=512,
        in_channels=256,
        num_blocks=16,
        upsampling_depth=5,
        enc_kernel_size=21,
        enc_num_basis=512,
        num_sources=1,
        )
    # 2: el passem a DataParallel perquè es va guardar com un DataParallel
    model = torch.nn.DataParallel(model)
    # 3: carreguem els pesos
    model.load_state_dict(torch.load(model_path, map_location=torch.device('mps')))
    # 4: El pasem a GPU. Tu podries provar torch.device("mps") que seria la teva GPU
    device = torch.device("mps")
    model = model.module.to(device)
    # 5: posem en mode Evaluació (es desactiva dropout i coses així)
    model.eval()
    return model


def get_parser():
    parser = argparse.ArgumentParser(
        "denoiser.pruebas",
        description="Performs live speech enhancement, reading audio from "
                    "the default mic (or interface specified by --in) and "
                    "writing the enhanced version to 'Soundflower (2ch)' "
                    "(or the interface specified by --out)."
        )
    parser.add_argument(
        "-i", "--in", dest="in_",  #python -m denoiser.live --in "MacBook Air (micrófono)" 
        help="name or index of input interface.")
    parser.add_argument(
        "-o", "--out", default="BlackHole 2ch", #python -m denoiser.live --out "BlackHole 2ch" 
        help="name or index of output interface.")
    #add_model_flags(parser)
    parser.add_argument(
        "--no_compressor", action="store_false", dest="compressor", #python -m denoiser.live --no_compressor
        help="Deactivate compressor on output, might lead to clipping.")
    parser.add_argument(
        "--device", default="mps") #python -m denoiser.live --device "cpu"
    parser.add_argument(
        "--dry", type=float, default=0.04, #python -m denoiser.live --dry 0.04 
        help="Dry/wet knob, between 0 and 1. 0=maximum noise removal "
             "but it might cause distortions. Default is 0.04")
    parser.add_argument(
        "-t", "--num_threads", type=int, #python -m denoiser.live --num_threads 1 
        help="Number of threads. If you have DDR3 RAM, setting -t 1 can "
             "improve performance.")
    parser.add_argument(
        "-f", "--num_frames", type=int, default=1, #python -m denoiser.live --num_frames 1 
        help="Number of frames to process at once. Larger values increase "
             "the overall lag, but will improve speed.")
    return parser


def parse_audio_device(device):
    if device is None:
        return device
    try:
        return int(device)
    except ValueError:
        return device


def query_devices(device, kind): #devuelve en 'caps' el dispositivo que elijamos 
    try:
        caps = sd.query_devices(device, kind=kind) #sd.query_devices() = python -m sounddevice
    except ValueError:
        message = bold(f"Invalid {kind} audio interface {device}.\n")
        message += (
            "If you are on Mac OS X, try installing Soundflower "
            "(https://github.com/mattingalls/Soundflower).\n"
            "You can list available interfaces with `python3 -m sounddevice` on Linux and OS X, "
            "and `python.exe -m sounddevice` on Windows. You must have at least one loopback "
            "audio interface to use this.")
        print(message, file=sys.stderr)
        sys.exit(1)
    return caps


def model_aplication(buffer, model):
    # guardem l'energia de la mixture per poder normalitzar la sortida del model
    ini_nrg = torch.sum(buffer ** 2)
    buffer = (buffer - torch.mean(buffer)) / torch.std(buffer)

    # Código a medir
    inicio = time.time()
    out = model(buffer.unsqueeze(0)).detach() #torch.Size([1,1,256]) con el audio sin ruido
    fin = time.time()
    print(f"Tiempo de ejecución del modelo: {fin-inicio}")

    out /= torch.sqrt(torch.sum(out ** 2) / ini_nrg)
    out = out[0,0,:] #torch.Size([256]) con el audio sin ruido
    return out


def main():

    mps_device = torch.device("mps")

    args = get_parser().parse_args() # en 'args' guardamos todos los argumentos que nosotros pasamos por terminal
    if args.num_threads: # Si añadimos un valor entero en el argumento num_threads
        torch.set_num_threads(args.num_threads) # Se establece el número de threads utilizado para las operaciones intra a la CPU.

    model_path = 'e39_sudo_whamr_16k_enhnoisy_augment.pt'
    model = load_sudormrf_causal_cpu(model_path)
    print("Model loaded.")


    sr=16000
    device_in = parse_audio_device(args.in_) # device_in = #python -m denoiser.live --in "MacBook Air (micrófono)"
    caps = query_devices(device_in, "input") # Guardamos en caps el dispositivo que hemos elegido como input
    channels_in = min(caps['max_input_channels'], 2) # int (mono=1, stereo=2) #default ==2
    stream_in = sd.InputStream(
        device=device_in,
        #samplerate=model.sample_rate, ## model.sample_rate || casualsudoRMF.sample_rate no existe 
        samplerate=sr,
        channels=channels_in)

    device_out = parse_audio_device(args.out) # device_out = #python -m denoiser.live --out "MacBook Air (altavoces)"
    caps = query_devices(device_out, "output") # Guardamos en caps el dispositivo que hemos elegido como output
    channels_out = min(caps['max_output_channels'], 2) # int (mono=1, stereo=2) default == 2
    stream_out = sd.OutputStream(
        device=device_out,
        #samplerate=model.sample_rate, ## model.sample_rate || casualsudoRMF.sample_rate no existe 
        samplerate=sr,
        channels=channels_out)

    stream_in.start()
    stream_out.start()
    first = True
    current_time = 0
    last_log_time = 0
    last_error_time = 0
    cooldown_time = 2
    log_delta = 10
    #sr_ms = model.sample_rate /1000 ## 16
    sr_ms = sr / 1000  
    #stride_ms = streamer.stride / sr_ms  ## 16
    stride_ms = sr /1000

    ## DE PRUEBA
    stride1 = 256
    length1 = 661


    # Inicializamos el buffer como un tensor vacío
    BUFFER_SIZE = 1  # Número de chunks a almacenar en el buffer
    buffer = torch.empty((1,0),dtype=torch.float32, device=mps_device)
    out = torch.empty((1,0),dtype=torch.float32, device=mps_device)

    #print(f"Ready to process audio, total lag: {streamer.total_length / sr_ms:.1f}ms.") #Streamer.total_length = 661
    print("Ready to process audio:")
    while True:
        try:
            if current_time > last_log_time + log_delta: # a cada 10 segundos se ejectua este condicional
                last_log_time = current_time
                print(f"Current time: {current_time:.1f}")
                
            #print(current_time)
            length = length1 if first else stride1 #256 siempre excepto la primera iteración (661)
            first = False #False siempre excepto la primera iteración
            current_time += length / sr #Se actualiza cada 0.016 segundos
            frame, overflow = stream_in.read(length) 
            # overflow = False or True
            # frame = numpy array of size (256,2)

            frame = torch.from_numpy(frame).mean(dim=1).to(args.device) # pasamos de stereo a mono haciendo el valor medio de las dos entradas de audio. También pasamos de un numpy array a un tensor.
            framePrueba = frame[None,:] #torch.Size([1, 256])
 
            with torch.no_grad(): #para eliminar el atributo para calcular el gradiente de los tensors creados (en el caso de que lo tuviera)
                buffer = torch.cat((buffer, framePrueba),1)

                # Si el buffer está lleno, procesamos los chunks de audio y enviamos el output a la salida
                if len(buffer[0,:]) >= length1 + (BUFFER_SIZE-1)*stride1:

                    out = model_aplication(buffer, model) 

                    if not out.numel(): #si la salida está vacía, vuelve al while (para evitar errores)
                        continue
                    if args.compressor: #entra siempre que no pongas --no compressor as input
                        out = 0.99 * torch.tanh(out) # returns a new tensor with the hyperbolic tangent of out. Sigue siendo un tensor de tamaño [256]
                    
                    out = out[:, None].repeat(1, channels_out) #duplicamos el tensor para tenerlo en dos dimensiones [256,2]. 
                    mx = out.abs().max().item() #Cogemos el valor absoluto más grande del frame procesado para comprobar si hay clipping.
                    if mx > 1:
                        print("Clipping!!")
                    out.clamp_(-1, 1) #para que ningún valor del tensor sea mayor que 1 o -1. Para solucionar el clipping!
                    out = out.cpu().numpy() #convertimos el tensor en un numpy array (256,2)
                    underflow = stream_out.write(out) #underflow =  False or True
                    if overflow or underflow:
                        if current_time >= last_error_time + cooldown_time:
                            last_error_time = current_time
                            print(f"Not processing audio fast enough should be less than {stride_ms:.1f}ms).")
                            
                    # Vaciamos el buffer
                    buffer = torch.empty((1,0),dtype=torch.float32, device=mps_device)

                # Si el buffer no está lleno, sigue cargando el buffer
                else:
                    continue

        except KeyboardInterrupt:
            print("Stopping")
            break
    stream_out.stop()
    stream_in.stop()


if __name__ == "__main__":
    main()
