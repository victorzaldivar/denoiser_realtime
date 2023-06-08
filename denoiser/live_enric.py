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
import time
import causal_improved_sudormrf_v3 


def get_parser():
    parser = argparse.ArgumentParser(
        "sudo-rm-rf.live",
        description="Performs live speech enhancement, reading audio from "
                    "the default mic (or interface specified by --in) and "
                    "writing the enhanced version to the output device "
                    "(or the interface specified by --out)."
        )
    parser.add_argument(
        "-i", "--in", dest="in_",
        help="name or index of input interface.")
    parser.add_argument(
        "-o", "--out", default="Soundflower (2ch)",
        help="name or index of output interface.")
    #add_model_flags(parser)
    parser.add_argument(
        "--no_compressor", action="store_false", dest="compressor",
        help="Deactivate compressor on output, might lead to clipping.")
    parser.add_argument(
        "--device", default="cpu", help='cpu, cuda or mps')
    #parser.add_argument(
    #    "--dry", type=float, default=0.04,
    #    help="Dry/wet knob, between 0 and 1. 0=maximum noise removal "
    #         "but it might cause distortions. Default is 0.04")
    parser.add_argument(
        "-t", "--num_threads", type=int, default=1,
        help="Number of threads. If you have DDR3 RAM, setting -t 1 can "
             "improve performance.")
    parser.add_argument(
        "-f", "--num_frames", type=int, default=1,
        help="Number of frames to process at once. Larger values increase "
             "the overall lag, but will improve speed.")
    parser.add_argument(
        '-b', '--bypass', action='store_true',
        help='bypass the model application')
    parser.add_argument(
        '-n', '--noise', action='store_true',
        help='use noise as input instead')
    return parser


def parse_audio_device(device):
    if device is None:
        return device
    try:
        return int(device)
    except ValueError:
        return device


def query_devices(device, kind):
    try:
        caps = sd.query_devices(device, kind=kind)
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


def main():
    fs = 16000
    length = 1600
    
    args = get_parser().parse_args()

    if args.num_threads:
        torch.set_num_threads(args.num_threads)

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
    
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load('e39_sudo_whamr_16k_enhnoisy_augment.pt', map_location=args.device))
    model = model.module.to(args.device)
    model.eval()

    print("Model loaded.")

    device_in = parse_audio_device(args.in_)
    caps = query_devices(device_in, "input")
    channels_in = min(caps['max_input_channels'], 2)
    stream_in = sd.InputStream(
        device=device_in,
        samplerate=fs,
        channels=channels_in)

    device_out = parse_audio_device(args.out)
    caps = query_devices(device_out, "output")
    channels_out = min(caps['max_output_channels'], 2)
    stream_out = sd.OutputStream(
        device=device_out,
        samplerate=fs,
        channels=channels_out)

    stream_in.start()
    stream_out.start()
    #first = True
    #current_time = 0
    #last_log_time = 0
    #last_error_time = 0
    #cooldown_time = 2
    #log_delta = 10
    #sr_ms = fs / 1000
    #stride_ms = stride / sr_ms
    #print(f"Ready to process audio, total lag: {streamer.total_length / sr_ms:.1f}ms.")

    while True:
        try:
            # idea 1: fer servir 4 més gran que el frame i fer batching, a veure si torch paral·lelitza? -> 
            # NO FUNCIONA, HAURIEM DE DONAR SORTIDA A MIG PROCÉS DEL BATCH

            # idea 2: intentar-ho dins callback amb els mètodes de la llibreria a veure què
            frame, overflow = stream_in.read(length)
            frame = torch.from_numpy(frame).mean(dim=1).to(args.device)
            frame = torch.stack(torch.split(frame, len(frame) // 4))
            if args.noise: 
                frame = torch.randn_like(frame)
            if not args.bypass:
                tic = time.time()
                ini_nrg = torch.sum(frame ** 2)
                frame = (frame - torch.mean(frame)) / torch.std(frame) #OPTIONAL NORMALIZATION 
                with torch.no_grad():
                    #out = model(frame.unsqueeze(0).unsqueeze(0)).detach().squeeze()
                    out = model(frame.unsqueeze(1)).detach().squeeze()
                    out = out.reshape(out.shape[0]*out.shape[1])
                out /= torch.sqrt(torch.sum(out ** 2) / ini_nrg) #energy constraint
                tac = time.time()
                #print((length/fs)/1000-((tac - tic)/1000)) #spare time!
            else:
                out = frame
            out = out.unsqueeze(1).repeat(1, 2) # upmix to stereo


            if not out.numel():
                continue
            if args.compressor:
                out = 0.99 * torch.tanh(out)
            #out = out[:, None].repeat(1, channels_out)
            mx = out.abs().max().item()
            if mx > 1:
                print("Clipping!!")
            out.clamp_(-1, 1)
            out = out.cpu().numpy()
            underflow = stream_out.write(out)
            #if overflow or underflow:
            #    if current_time >= last_error_time + cooldown_time:
            #        last_error_time = current_time
            #        tpf = 1000 * streamer.time_per_frame
            #        print(f"Not processing audio fast enough, time per frame is {tpf:.1f}ms "
            #              f"(should be less than {stride_ms:.1f}ms).")
        except KeyboardInterrupt:
            print("Stopping")
            break
    stream_out.stop()
    stream_in.stop()


if __name__ == "__main__":
    main()