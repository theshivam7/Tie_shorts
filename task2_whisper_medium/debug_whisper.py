import numpy as np, whisper, tempfile, wave
from datasets import load_dataset
ds = load_dataset('raianand/TIE_shorts', split='test', cache_dir='/home/users/ntu/ccdsshiv/hf_cache')
sample = ds[0]
audio = np.array(sample['audio']['array'], dtype=np.float32)
sr = sample['audio']['sampling_rate']
print(f'Sample rate: {sr}, Shape: {audio.shape}, Min: {audio.min():.4f}, Max: {audio.max():.4f}')
with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
    tmp_path = tmp.name
    audio_int16 = (audio * 32767).astype(np.int16)
    with wave.open(tmp_path, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(audio_int16.tobytes())
print(f'Wrote WAV to: {tmp_path}')
model = whisper.load_model('medium', device='cuda')
result = model.transcribe(tmp_path, language='en')
print(f'Result: {result["text"]}')
