import sounddevice as sd
import numpy as np
import causal_improved_sudormrf_v3
import torch
import argparse

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



# Define the audio processing function
def process_audio(indata, outdata, frames, time, status):
    #print(frames)
    print(frames)
    print(time)
    print(status)
    # Perform some audio processing
    # Here, we'll simply invert the audio signal
    mixture = torch.from_numpy(indata[:,0]).squeeze()
    if args.noise:
        mixture = torch.randn_like(mixture) * 0.5
    if not args.bypass:
        ini_nrg = torch.sum(mixture ** 2)
        mixture = (mixture - torch.mean(mixture)) / torch.std(mixture)
        with torch.no_grad():
            mixture = model(mixture.unsqueeze(0).unsqueeze(0)).detach().squeeze()
        outdata[:] = (mixture / torch.sqrt(torch.sum(mixture ** 2) / ini_nrg)).unsqueeze(1)
    else:    
        outdata[:] = indata

args = get_parser().parse_args()

# Select input and output devices (using device indices)
input_device_idx = int(args.in_)
output_device_idx = int(args.out)

# Set the sample rate and buffer size
#sample_rate = 16000
#buffer_size = 1600

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

# Start the audio stream
with sd.Stream(device=(input_device_idx, output_device_idx),
               channels=1,
               samplerate=16000,
               callback=process_audio) as stream :
    print("\nAudio stream started. Press Ctrl+C to stop.")
    try:
        while True:
            #print(stream.latency)
            pass
    except KeyboardInterrupt:
        print("Stopping")
# Clean up resources
sd.stop()
