from typing import Any, Dict, Union
import numpy as np
import torch
import torchaudio
import soundfile
import glob
import librosa
from colorama import Fore, Style

import sys
sys.path.append("../")
from utils import symbols_to_bits, bits_to_symbols
from neural_decoding.utils.utils import load_config

from omnisealbench.configs import *
from omnisealbench.models import AudioWatermarkDetector, AudioWatermarkGenerator, Watermark
from omnisealbench.watermarkgen import run_watermarkgen


class WatermarkWrapper:
    """
    A class for watermarking audio samples using various watermarking models.
    """
    
    def __init__(self, watermark_config_path: str, model_keys: List[str] = ["timbre", "audioseal"]):
        """
        Initialize the watermark generator with specified models.
        
        Args:
            watermark_config_path (str): Path to watermarking configuration file.
            model_keys (list): List of model keys to use for watermarking.
        """
        self.watermark_config_path = watermark_config_path
        self.model_keys = model_keys
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.target_sr = 16000  # all models use 16kHz
        
        # Load watermarking config (need payload delays for symbol to bits conversion)
        cfg = load_config(watermark_config_path)
        self.payload_delays = cfg['payload_delays']
        
        # Load default configs
        (
            self.watermarking_config,
            self.audio_attacks_config,
            self.image_attacks_config,
            self.registered_attacks,
            self.models,
            self.datasets,
        ) = load_configs(None)
        
        # Build registered models configs
        self.registered_models_configs = build_registered_models_configs(
            "audio", self.models, None
        )
        
        # Initialize generators and detectors for each model
        self.generators = {}
        self.detectors = {}
        self.message_sizes = {}
        
        for model_key in self.model_keys:
            model_config = self.registered_models_configs[model_key]
            self.message_sizes[model_key] = model_config["nbits"]
            
            # Build generator
            self.generators[model_key] = self._build_generator(model_key, model_config)
            
            # Build detector
            if "builder_func" in model_config:
                self.detectors[model_key] = self.generators[model_key]
            else:
                detector_config = model_config["detector"]
                build_detector = detector_config["builder_func"]
                additional_arguments = detector_config["additional_arguments"]
                self.detectors[model_key] = build_detector(
                    device=self.device, **additional_arguments
                )
    
    
    def _build_generator(
        self, model_key: str, model_config: Dict[str, Any]
    ) -> Union[AudioWatermarkGenerator, Watermark]:
        """
        Build a generator for the specified model.
        
        Args:
            model_key (str): Key identifying the model.
            model_config (Dict[str, Any]): Configuration for the model.
            
        Returns:
            Generator instance (AudioWatermarkGenerator or Watermark).
        """
        if model_key == "custom_audio":
            generator: AudioWatermarkGenerator = model_config[model_key]["generator"]
        elif model_key == "custom_image":
            raise NotImplementedError
        else:
            if "builder_func" in model_config:
                builder_func = model_config["builder_func"]
                generator: Watermark = builder_func(
                    device=self.device,
                    nbits=model_config["nbits"],
                    **model_config["additional_arguments"],
                )
            else:
                generator_config = model_config["generator"]
                build_generator = generator_config["builder_func"]
                additional_arguments = generator_config["additional_arguments"]

                generator: AudioWatermarkGenerator = build_generator(
                    model=model_key, device=self.device, **additional_arguments
                )

        return generator
    

    def generate_watermarked_audio_samples(self, stegaphone_samples_path: str):
        """
        Watermark audio samples for a StegaPhone set using specified models and save.

        Args:
            stegaphone_samples_path (str): Path to StegaPhone samples directory. These folders should have
                been created using data_collection.generate_stegaphone_watermarked_audio_samples().
        
        Returns:
            None, saves watermarked audio files in the respective sample folders.
        """
        # Get sample folders to process
        sample_folders = glob.glob(stegaphone_samples_path + "/sample*")
        sample_folders.sort()

        for model_key in self.model_keys:
            generator = self.generators[model_key]
            message_sz = self.message_sizes[model_key]

            for sample_folder in sample_folders:
                print(Fore.MAGENTA + f"Processing sample folder: {sample_folder} with model: {model_key}" + Style.RESET_ALL)
        
                # Load sample data
                symbols = np.load(f"{sample_folder}/encoded_symbols.npy")
                input_audio, file_sr = soundfile.read(f"{sample_folder}/original.wav")
                assert len(input_audio.shape) == 1, "Input audio must be mono"

                # Convert to tensor, reshape, and resample if needed
                waveform = torch.tensor(input_audio).unsqueeze(0)  # c=1 t
                waveform = waveform.float()
                waveform = waveform.unsqueeze(0)  # b=1 c=1 t
                if self.target_sr != file_sr:
                    waveform = torchaudio.transforms.Resample(
                        orig_freq=file_sr,
                        new_freq=self.target_sr,
                    )(waveform)

                # Convert symbols to bits of secret message to embed
                secret_message = symbols_to_bits(symbols, len(self.payload_delays))[:message_sz]
                secret_message = torch.tensor(secret_message, dtype=torch.int32).unsqueeze(0)

                # Generate watermarked audio
                watermarked_audio = generator.generate_watermark_audio(
                    tensor=waveform,
                    sample_rate=self.target_sr,
                    secret_message=secret_message,
                )

                # Save watermarked audio
                watermarked_audio_np = watermarked_audio.squeeze().cpu().numpy()
                # resample back to original sr if needed
                if self.target_sr != file_sr:
                    watermarked_audio_np = librosa.resample(
                        watermarked_audio_np, orig_sr=self.target_sr, target_sr=file_sr
                    )
                soundfile.write(f"{sample_folder}/{model_key}.wav", watermarked_audio_np, file_sr)
    

    def decode_recordings(self, stegaphone_samples_path: str):
         # Get sample folders to process
        sample_folders = glob.glob(stegaphone_samples_path + "/sample*")
        sample_folders.sort()

        for model_key in self.model_keys:
            detector = self.detectors[model_key]
            message_sz = self.message_sizes[model_key]

            for sample_folder in sample_folders:
                print(Fore.MAGENTA + f"Processing sample folder: {sample_folder} with model: {model_key}" + Style.RESET_ALL)
        
                # Load sample data
                recordings = glob.glob(f"{sample_folder}/timbre_recording_*.wav")

                for recording_file in recordings:
                    print(f"Decoding recording: {recording_file}")
                    input_audio, file_sr = soundfile.read(recording_file)
                    assert len(input_audio.shape) == 1, "Input audio must be mono"
                    gt_symbols = np.load(f"{sample_folder}/encoded_symbols.npy")
                    msg_gt = symbols_to_bits(gt_symbols, len(self.payload_delays))[:message_sz]

                    # Convert to tensor, reshape, and resample if needed
                    waveform = torch.tensor(input_audio).unsqueeze(0)  # c=1 t
                    waveform = waveform.float()
                    waveform = waveform.unsqueeze(0)  # b=1 c=1 t
                    if self.target_sr != file_sr:
                        waveform = torchaudio.transforms.Resample(
                            orig_freq=file_sr,
                            new_freq=self.target_sr,
                        )(waveform)
                    
                    detect_prob, msg_decoded = detector.detect_watermark_audio(waveform, self.target_sr)
                    msg_decoded_np = msg_decoded.cpu().numpy().flatten()
                    msg_decoded_binarized = (msg_decoded_np > 0.5).astype(np.int32)
                    print("Secret message: ", msg_gt)
                    print("Decoded message: ", msg_decoded_binarized)
                    ber = np.sum(msg_gt != msg_decoded_binarized) / message_sz
                    print(f"Bit Error Rate (BER): {ber:.2f}")
                    print(bits_to_symbols(msg_decoded_binarized, len(self.payload_delays)))
                    print(gt_symbols)
                

# Initialize the watermark generator
watermark_gen = WatermarkWrapper(
    watermark_config_path="../neural_decoding/configs/watermark_config1_rec.yaml",
    model_keys=["timbre", "audioseal"]
)

# # Generate watermarked audio samples
# watermark_gen.generate_watermarked_audio_samples(
#     stegaphone_samples_path="../offline_data_samples"
# )

watermark_gen.decode_recordings(
    stegaphone_samples_path="../offline_data_samples"
)



