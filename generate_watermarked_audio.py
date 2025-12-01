import json
import multiprocessing
import random
from datetime import datetime
from functools import singledispatch
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torchaudio
import soundfile


from omnisealbench.configs import *
# from omnisealbench.omnisealbench.detection import run_audio_detection, run_image_detection
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


device = "cuda" if torch.cuda.is_available() else "cpu"
(
        watermarking_config,
        audio_attacks_config,
        image_attacks_config,
        registered_attacks,
        models,
        datasets,
) = load_configs(None) # load default configs 
registered_models_configs = build_registered_models_configs(
        "audio", models, None
)

##########
target_sr = 16000
max_length_in_seconds = 5
############

model_key = "audioseal"
model_config = registered_models_configs[model_key]

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

audio_file = "/Users/hadleigh/ears_reverb_p001/emo_pain_sentences.wav"
raw_audio, sr = soundfile.read(audio_file) # load a mono audio file
assert len(raw_audio.shape) == 1, "Input audio must be mono"

waveform = torch.tensor(raw_audio).unsqueeze(0)  # c=1 t
waveform = waveform.float()

message_sz = model_config["nbits"]
max_length = max_length_in_seconds * sr

# Truncate or pad the waveform to max_length
if waveform.shape[0] > max_length:
    waveform = waveform[:max_length]
elif waveform.shape[0] < max_length:
    padding = max_length - waveform.shape[0]
    waveform = torch.nn.functional.pad(waveform, (0, padding))

waveform = waveform.unsqueeze(0)  # b=1 c=1 t

# resample to 16k
if target_sr != sr:
    waveform = torchaudio.transforms.Resample(
        orig_freq=sr,
        new_freq=target_sr,
    )(waveform)

# generate secret message
secret_message = torch.randint(
    0, 2, (waveform.shape[0], message_sz), dtype=torch.int32
)  # b=1 16
watermarked_audio = generator.generate_watermark_audio(
    tensor=waveform,
    sample_rate=target_sr,
    secret_message=secret_message,
)

detect_prob, msg_decoded = detector.detect_watermark_audio(watermarked_audio, target_sr)
secret_message_np = secret_message.cpu().numpy().flatten()
msg_decoded_np = msg_decoded.cpu().numpy().flatten()
msg_decoded_binarized = (msg_decoded_np > 0.5).astype(np.int32)
print("Secret message: ", secret_message_np)
print("Decoded message: ", msg_decoded_binarized)
ber = np.sum(secret_message_np != msg_decoded_binarized) / message_sz
print(f"Bit Error Rate (BER): {ber:.2f}")