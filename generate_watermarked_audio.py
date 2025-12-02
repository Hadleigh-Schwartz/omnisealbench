from typing import Any, Dict, Union
import numpy as np
import torch
import torchaudio
import soundfile
import glob

import sys
sys.path.append("../")
from utils import symbols_to_bits, bits_to_symbols
from neural_decoding.utils.utils import load_config

from omnisealbench.configs import *
from omnisealbench.models import AudioWatermarkDetector, AudioWatermarkGenerator, Watermark
from omnisealbench.watermarkgen import run_watermarkgen


def build_generator(
    model_key: str, model_config: Dict[str, Any], device: str
) -> Union[AudioWatermarkGenerator, Watermark]:
    if model_key == "custom_audio":
        generator: AudioWatermarkGenerator = model_config[model_key]["generator"]
    elif model_key == "custom_image":
        # generator: Watermark = model_config[model_key]["generator"]
        raise NotImplementedError
    else:
        if "builder_func" in model_config:
            builder_func = model_config["builder_func"]
            generator: Watermark = builder_func(
                device=device,
                nbits=model_config["nbits"],
                **model_config["additional_arguments"],
            )
        else:
            generator_config = model_config["generator"]
            build_generator = generator_config["builder_func"]
            additional_arguments = generator_config["additional_arguments"]

            generator: AudioWatermarkGenerator = build_generator(
                model=model_key, device=device, **additional_arguments
            )

    return generator


def watermarked_audio_samples(stegaphone_samples_path, 
                              watermark_config_path,
                              model_keys = ["timbre", "audioseal"]):
    
    # Load watermarking config (need payload delays for symbol to bits conversion)
    cfg = load_config(watermark_config_path)
    payload_delays = cfg['payload_delays']

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load default configs
    (
        watermarking_config,
        audio_attacks_config,
        image_attacks_config,
        registered_attacks,
        models,
        datasets,
    ) = load_configs(None) 
    target_sr = 16000 # all models use 16kHz

    # Build registered models configs
    registered_models_configs = build_registered_models_configs(
            "audio", models, None
    )



 
    sample_folders = glob.glob(stegaphone_samples_path + "/sample*")
    sample_folders.sort()
    for model_key in model_keys:

        model_config = registered_models_configs[model_key]
        message_sz = model_config["nbits"]


        # Build generator
        generator = build_generator(model_key, model_config, device)

        # Build detector
        if "builder_func" in model_config:
            detector: Watermark = generator
        else:
            detector_config = model_config["detector"]
            build_detector = detector_config["builder_func"]
            additional_arguments = detector_config["additional_arguments"]
            detector: AudioWatermarkDetector = build_detector(
                device=device, **additional_arguments
            )

        for sample_folder in sample_folders:
            print(f"Processing sample folder: {sample_folder} with model: {model_key}")
    
            # Load sample data
            symbols = np.load(f"{sample_folder}/encoded_symbols.npy")
            input_audio, sr = soundfile.read(f"{sample_folder}/original.wav") # load a mono audio file
            assert len(input_audio.shape) == 1, "Input audio must be mono"

            # Convert to tensor, reshape, and resample if needed
            waveform = torch.tensor(input_audio).unsqueeze(0)  # c=1 t
            waveform = waveform.float()
            waveform = waveform.unsqueeze(0)  # b=1 c=1 t
            if target_sr != sr:
                waveform = torchaudio.transforms.Resample(
                    orig_freq=sr,
                    new_freq=target_sr,
                )(waveform)

            # Convert symbols to bits of secret message to embed
            secret_message = symbols_to_bits(symbols, len(payload_delays))[:message_sz] # only can use first N bits
            secret_message = torch.tensor(secret_message, dtype=torch.int32).unsqueeze(0)  # b=1 message_sz

            # Generate watermarked audio
            watermarked_audio = generator.generate_watermark_audio(
                tensor=waveform,
                sample_rate=target_sr,
                secret_message=secret_message,
            )

            # #  Sanity check: Detect watermark
            # detect_prob, msg_decoded = detector.detect_watermark_audio(watermarked_audio, target_sr)
            # secret_message_np = secret_message.cpu().numpy().flatten()
            # msg_decoded_np = msg_decoded.cpu().numpy().flatten()
            # msg_decoded_binarized = (msg_decoded_np > 0.5).astype(np.int32)
            # print("Secret message: ", secret_message_np)
            # print("Decoded message: ", msg_decoded_binarized)
            # ber = np.sum(secret_message_np != msg_decoded_binarized) / message_sz
            # print(f"Bit Error Rate (BER): {ber:.2f}")
            # print(bits_to_symbols(secret_message_np.tolist(), len(payload_delays)))
            # print(symbols)

            # Save watermarked audio
            watermarked_audio_np = watermarked_audio.squeeze().cpu().numpy()
            soundfile.write(f"{sample_folder}/{model_key}.wav", watermarked_audio_np, target_sr)
            print(f"Saved watermarked audio to {sample_folder}/{model_key}.wav")

watermarked_audio_samples("../offline_data_samples", "../neural_decoding/configs/watermark_config1_rec.yaml")