import pyaudio
import numpy as np
import torch
import keyboard
from tqdm import tqdm
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
from safetensors.torch import load_file


RATE = 16000
CHUNK = 1024
RECORD_SECONDS = 5

LABELS = ['siren', 'gunshot', 'explosion', 'casual']
label2id = {label: i for i, label in enumerate(LABELS)}
AST_MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"
AST_MODEL_WEIGHT = '...'
NUM_LABELS = 4


if __name__ == "__main__":
    feature_extractor = AutoFeatureExtractor.from_pretrained(AST_MODEL_NAME)
    params_dict = load_file(AST_MODEL_WEIGHT)
    model = AutoModelForAudioClassification.from_pretrained(
        AST_MODEL_NAME, num_labels=NUM_LABELS, ignore_mismatched_sizes=True
    )
    model.load_state_dict(params_dict)
    model.eval()
    audio_interface = pyaudio.PyAudio()
    stream = audio_interface.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )
    print("Recording started. Press 'q' to exit.\n")
    while True:
        if keyboard.is_pressed("q"):
            print("Exit triggered.")
            break
        frames = []
        total_chunks = int(RATE / CHUNK * RECORD_SECONDS)
        for _ in tqdm(range(total_chunks), desc="Recording", ncols=100):
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(np.frombuffer(data, dtype=np.int16))
            # Allow early exit
            if keyboard.is_pressed("q"):
                print("Exit triggered.")
                break
        audio_data = np.concatenate(frames).astype(np.float32)
        inputs = feature_extractor(
            audio_data, sampling_rate=RATE,
            return_tensors="pt", padding=True
        )
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_id = torch.argmax(logits, dim=1).item()
        predicted_label = LABELS[predicted_id]
        print(f"\nPredicted class → {predicted_label} ({predicted_id})\n")
