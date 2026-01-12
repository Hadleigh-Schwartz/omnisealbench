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
    
    def __init__(self, stegaphone_watermark_config_path: str, model_keys: List[str] = ["timbre", "audioseal"]):
        """
        Initialize the watermark generator with specified models.
        
        Args:
            watermark_config_path (str): Path to watermarking configuration file.
            model_keys (list): List of model keys to use for watermarking.
        """
        self.model_keys = model_keys
        self.device = "cpu" #"cuda" if torch.cuda.is_available() else "cpu"
        self.target_sr = 16000  # all models use 16kHz
        
        # Load watermarking config (need payload delays for symbol to bits conversion)
        stegaphone_cfg = load_config(stegaphone_watermark_config_path)
        self.payload_delays = stegaphone_cfg['payload_delays']
        
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
    

    def generate_watermarked_audio_samples(self, stegaphone_source_samples_folder,  group_output_folder):
        """
        Watermark audio samples for a StegaPhone set using specified models and save.

        Args:
            stegaphone_samples_folder (str): Path to StegaPhone samples directory. These folders should have
                been created using data_collection.generate_stegaphone_watermarked_audio_samples(). It will contain
                a subfolder for each watermark_config used during data generation, and the original samples in each are the same,
                so we only need to process one of these subfolders and then simply copy the watermarked files to the other subfolders.
        
        Returns:
            None, saves watermarked audio files in the respective sample folders.
        """
  
        # Get sample folders from the first config folder to process
        sample_folders = glob.glob(f"{stegaphone_source_samples_folder}/sample*")
        sample_folders.sort()

        for model_key in self.model_keys:
            generator = self.generators[model_key]
            detector = self.detectors[model_key] # for sanity checking later
            message_sz = self.message_sizes[model_key]

            for sample_folder in sample_folders:
                ###### PREPARE FOLDERS AND LOAD AUDIO ######
                print(Fore.MAGENTA + f"Processing sample folder: {sample_folder} with model: {model_key}" + Style.RESET_ALL)
                sample_name = sample_folder.split("/")[-1]

                # Create companion folder for the output
                output_sample_folder = f"{group_output_folder}/{model_key}/{sample_name}"
                print(f"Output sample folder: {output_sample_folder}")
                os.makedirs(output_sample_folder, exist_ok=True)
                
                # Load original audio from StegaPhone sample folder to maintain consistency
                input_audio, file_sr = soundfile.read(f"{sample_folder}/original.wav")
                assert len(input_audio.shape) == 1, "Input audio must be mono"

                ##### WATERMARKING PROCESS #####
                # Convert to tensor, reshape, and resample if needed
                waveform = torch.tensor(input_audio).unsqueeze(0)  # c=1 t
                waveform = waveform.float()
                waveform = waveform.unsqueeze(0)  # b=1 c=1 t
                if self.target_sr != file_sr:
                    waveform = torchaudio.transforms.Resample(
                        orig_freq=file_sr,
                        new_freq=self.target_sr,
                    )(waveform)

                # Generate random secret message to embed
                secret_message = np.random.randint(0, 2, size=(message_sz,))
                secret_message = torch.tensor(secret_message, dtype=torch.int32).unsqueeze(0)

                # Generate watermarked audio
                watermarked_audio = generator.generate_watermark_audio(
                    tensor=waveform,
                    sample_rate=self.target_sr,
                    secret_message=secret_message,
                )

                # Sanity check: decode the watermarked audio right away to confirm message is correct
                _, msg_decoded = detector.detect_watermark_audio(watermarked_audio, self.target_sr)
                msg_decoded_np = msg_decoded.cpu().numpy().flatten()
                msg_decoded_binarized = (msg_decoded_np > 0.5).astype(np.int32)
                assert np.array_equal(msg_decoded_binarized, secret_message.squeeze().cpu().numpy()), \
                    "Decoded message does not match embedded message!"


                ##### SAVE THINGS #####
                # Convert the watermarked audio back to original sample rate if needed and save
                watermarked_audio = torchaudio.transforms.Resample(
                    orig_freq=self.target_sr,
                    new_freq=file_sr,
                )(watermarked_audio)
                watermarked_audio_np = watermarked_audio.squeeze().cpu().numpy()
                soundfile.write(f"{output_sample_folder}/encoded.wav", watermarked_audio_np, file_sr)
                
                # Save the OG audio there too
                soundfile.write(f"{output_sample_folder}/original.wav", input_audio, file_sr)

                # Save the bits that were embedded too 
                np.save(f"{output_sample_folder}/encoded_bits.npy", secret_message.squeeze().cpu().numpy())

                # Copy the raw_audio_path.txt file from the StegaPhone sample folder if it exists
                os.system(f"cp {sample_folder}/raw_audio_path.txt {output_sample_folder}/raw_audio_path.txt")


    def decode_recordings(self, samples_folder):
         # Get sample folders to process
       

        for model_key in self.model_keys:
            detector = self.detectors[model_key]
 
            sample_folders = glob.glob(f"{samples_folder}/{model_key}/sample*")
            sample_folders.sort()

            for sample_folder in sample_folders:
                print(Fore.MAGENTA + f"Processing sample folder: {sample_folder} with model: {model_key}" + Style.RESET_ALL)
        
                # Load sample data
                recordings = glob.glob(f"{sample_folder}/*recording*.wav") # TODO: update

                for recording_file in recordings:
                    print(f"Decoding recording: {recording_file}")
                    input_audio, file_sr = soundfile.read(recording_file)
                    msg_gt = np.load(f"{sample_folder}/encoded_bits.npy")

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
                    ber = np.sum(msg_gt != msg_decoded_binarized) / len(msg_gt)
                    print(f"Bit Error Rate (BER): {ber:.2f}")
                  

    def dynamic_message_validation(self, input_audio_path, model_key):
        """
        Confirm claim that if you embed a different message in consecutive chunks of audio,
        then you can't decode them correctly from the full audio unless you have the correct
        chunk alignment.
        """

        input_audio, file_sr = soundfile.read(input_audio_path)
        # resample to target_sr if needed
        if file_sr != self.target_sr:
            input_audio = librosa.resample(input_audio, orig_sr=file_sr, target_sr=self.target_sr)
            file_sr = self.target_sr

        generator = self.generators[model_key]
        detector = self.detectors[model_key]
        message_sz = self.message_sizes[model_key]

        # divide input audio into N chunks of equal size
        N = 2
        chunk_size = len(input_audio) // N

        ##### CASE 1: CONSTANT MESSGE ACROSS CHUNKS #####
        print(Fore.MAGENTA + "\n\nCASE 1: CONSTANT MESSAGE ACROSS CHUNKS\n\n" + Style.RESET_ALL)
        watermarked_chunks = []
 
        # Message to be embedded in each chunk:
        msg_gt = np.random.randint(0, 2, size=(message_sz,))
        secret_message = torch.tensor(msg_gt, dtype=torch.int32).unsqueeze(0)
        
        # Iterate through chunks, watermark each chunk with the same message
        # and collect the watermarked chunks into a single audio array
        for i in range(N):
            chunk = input_audio[i * chunk_size:(i + 1) * chunk_size]

            # # Convert to tensor, reshape, and resample if needed
            chunk_tensor = torch.tensor(chunk).unsqueeze(0)  # c=1 t
            chunk_tensor = chunk_tensor.float()
            chunk_tensor = chunk_tensor.unsqueeze(0)  # b=1 c=1 t

            # Generate watermarked audio
            watermarked_chunk = generator.generate_watermark_audio(
                tensor = chunk_tensor,
                sample_rate=self.target_sr,
                secret_message=secret_message,
            )

            _, chunk_msg_decoded = detector.detect_watermark_audio(watermarked_chunk, self.target_sr)
            chunk_msg_decoded_np = chunk_msg_decoded.cpu().numpy().flatten()
            chunk_msg_decoded_binarized = (chunk_msg_decoded_np > 0.5).astype(np.int32)
            print("Chunk GT message: ", msg_gt)
            print("Chunk Decoded message: ", chunk_msg_decoded_binarized)
            print("Chunk match: ", np.array_equal(chunk_msg_decoded_binarized, msg_gt))

            watermarked_chunk_np = watermarked_chunk.squeeze().cpu().numpy()
            watermarked_chunks.append(watermarked_chunk_np)

        complete_watermarked_audio_np = np.concatenate(watermarked_chunks, axis=0)
        complete_watermarked_audio = torch.tensor(complete_watermarked_audio_np).unsqueeze(0).unsqueeze(0)
        _, complete_msg_decoded = detector.detect_watermark_audio(complete_watermarked_audio, self.target_sr)
        complete_msg_decoded_np = complete_msg_decoded.cpu().numpy().flatten()
        complete_msg_decoded_binarized = (complete_msg_decoded_np > 0.5).astype(np.int32)
        print("Complete GT message: ", msg_gt)
        print("Complete Decoded message: ", complete_msg_decoded_binarized)

        
        # Select one random chunk of chunk_size within complete_watermarked_audio_np and decode it
        # Notice that this random chunk may not be aligned with the original chunks, but the message is 
        # still correctly decoded since all chunks had the same message
        random_chunk_start = np.random.randint(0, len(complete_watermarked_audio_np) - chunk_size)
        random_chunk = complete_watermarked_audio_np[random_chunk_start:random_chunk_start + chunk_size]
        random_chunk_tensor = torch.tensor(random_chunk).unsqueeze(0).unsqueeze(0)
        detect_prob, random_chunk_msg_decoded = detector.detect_watermark_audio(random_chunk_tensor, self.target_sr)
        random_chunk_msg_decoded_np = random_chunk_msg_decoded.cpu().numpy().flatten()
        random_chunk_msg_decoded_binarized = (random_chunk_msg_decoded_np > 0.5).astype(np.int32)
        print(f"Random Chunk from {random_chunk_start} to {random_chunk_start + chunk_size}: Decoded message: ", random_chunk_msg_decoded_binarized)

        # print whether or not each decoded thing is the same as the gt
        print("Complete decoded matches GT: ", np.array_equal(complete_msg_decoded_binarized, msg_gt))
        print("Random chunk decoded matches GT: ", np.array_equal(random_chunk_msg_decoded_binarized, msg_gt))


        ##### CASE 2: DIFFERENT MESSAGE ACROSS CHUNKS #####
        print(Fore.MAGENTA + "\n\nCASE 2: DIFFERENT MESSAGE ACROSS CHUNKS\n\n" + Style.RESET_ALL)
        watermarked_chunks = []
        msg_gt = [] # will be built up over chunks
       
        
        # Iterate through chunks, watermark each chunk with the same message
        # and collect the watermarked chunks into a single audio array
        for i in range(N):
            chunk = input_audio[i * chunk_size:(i + 1) * chunk_size]

            # Unique message to be embedded in this chunk:
            msg_gt_chunk = np.random.randint(0, 2, size=(message_sz,))
            secret_message = torch.tensor(msg_gt_chunk, dtype=torch.int32).unsqueeze(0)
            msg_gt.append(msg_gt_chunk)

            # Convert to tensor, reshape, and resample if needed
            chunk_tensor = torch.tensor(chunk).unsqueeze(0)  # c=1 t
            chunk_tensor = chunk_tensor.float()
            chunk_tensor = chunk_tensor.unsqueeze(0)  # b=1 c=1 t
            
            # Generate watermarked audio
            watermarked_chunk = generator.generate_watermark_audio(
                tensor = chunk_tensor,
                sample_rate=self.target_sr,
                secret_message=secret_message,
            )

            _, chunk_msg_decoded = detector.detect_watermark_audio(watermarked_chunk, self.target_sr)
            chunk_msg_decoded_np = chunk_msg_decoded.cpu().numpy().flatten()
            chunk_msg_decoded_binarized = (chunk_msg_decoded_np > 0.5).astype(np.int32)
            print("Chunk GT message: ", msg_gt_chunk)
            print("Chunk Decoded message: ", chunk_msg_decoded_binarized)
            print("Chunk match: ", np.array_equal(chunk_msg_decoded_binarized, msg_gt_chunk))

            watermarked_chunk_np = watermarked_chunk.squeeze().cpu().numpy()
            watermarked_chunks.append(watermarked_chunk_np)
    
        msg_gt = np.concatenate(msg_gt, axis=0)

        complete_watermarked_audio_np = np.concatenate(watermarked_chunks, axis=0)
        complete_watermarked_audio = torch.tensor(complete_watermarked_audio_np).unsqueeze(0).unsqueeze(0)
        _, complete_msg_decoded = detector.detect_watermark_audio(complete_watermarked_audio, self.target_sr)
        complete_msg_decoded_np = complete_msg_decoded.cpu().numpy().flatten()
        complete_msg_decoded_binarized = (complete_msg_decoded_np > 0.5).astype(np.int32)
        print("Complete GT message: ", msg_gt)
        print("Complete Decoded message: ", complete_msg_decoded_binarized)

        
        # Select one random chunk of chunk_size within complete_watermarked_audio_np and decode it
        # Notice that this random chunk may not be aligned with the original chunks, but the message is 
        # still correctly decoded since all chunks had the same message
        random_chunk_start = np.random.randint(0, len(complete_watermarked_audio_np) - chunk_size)
        random_chunk = complete_watermarked_audio_np[random_chunk_start:random_chunk_start + chunk_size]
        random_chunk_tensor = torch.tensor(random_chunk).unsqueeze(0).unsqueeze(0)
        _, random_chunk_msg_decoded = detector.detect_watermark_audio(random_chunk_tensor, self.target_sr)
        random_chunk_msg_decoded_np = random_chunk_msg_decoded.cpu().numpy().flatten()
        random_chunk_msg_decoded_binarized = (random_chunk_msg_decoded_np > 0.5).astype(np.int32)
        print(f"Random Chunk from {random_chunk_start} to {random_chunk_start + chunk_size}: Decoded message: ", random_chunk_msg_decoded_binarized)

        # print whether or not each decoded thing is the same as the gt
        print("Complete decoded matches GT: ", np.array_equal(complete_msg_decoded_binarized, msg_gt))
        print("Random chunk decoded matches GT: ", np.array_equal(random_chunk_msg_decoded_binarized, msg_gt))


                

# Initialize the watermark generator
watermark_gen = WatermarkWrapper(
    stegaphone_watermark_config_path="../neural_decoding/configs/watermark_config_pream_16bps.yaml", # needed to get payload delays for symbol to bits conversion
    model_keys=["timbre", "audioseal"]
)

# watermark_gen.dynamic_message_validation(
#     input_audio_path="/media/storage/hadleigh/stegaphone_data/final/p100/watermark_config_pream_16bps/sample_1/original.wav",
#     model_key="audioseal"
# )

# Generate watermarked audio samples
# TODO:  watermark "musdb_val", "musdb_test"
groups = ["p100", "p101", "p102", "p103", "p104", "p105", "p106", "p107"] # from the ears dataset, make sure to include these but not in training
for group in groups:
    watermark_gen.generate_watermarked_audio_samples(
        stegaphone_source_samples_folder=f"/media/storage/hadleigh/stegaphone_data/final/{group}/watermark_config_pream_16bps/", # source of original audio and symbols to use
        group_output_folder=f"/media/storage/hadleigh/stegaphone_data/final/{group}/" # a folder for audioseal and timbre folders to be created ni
    )

# watermark_gen.decode_recordings(
#     samples_folder = f"/media/storage/hadleigh/stegaphone_data/final/{group}"
# )




