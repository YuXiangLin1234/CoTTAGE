import os

import librosa
import soundfile as sf

TARGET_SR = 16000


def duration_seconds(path):
    try:
        info = sf.info(path)
        return info.frames / float(info.samplerate)
    except Exception:
        return 0.0


def ensure_16k(src_path, dest_path):
    if os.path.exists(dest_path):
        return dest_path
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    wav, _ = librosa.load(src_path, sr=TARGET_SR, mono=True)
    tmp_path = f"{dest_path}.{os.getpid()}.tmp.wav"
    sf.write(tmp_path, wav, TARGET_SR)
    os.replace(tmp_path, dest_path)
    return dest_path
