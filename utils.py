import os
import random

import numpy as np


def ensure_dir_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def normalize(vec):
    return vec / np.max(np.abs(vec), axis=0)


def draw_wave(*wave):
    from matplotlib import pyplot as plt
    plt.figure()
    for w in wave:
        plt.plot(w, linewidth=.2)
    plt.show()


def play(array, fs=16000):
    import sounddevice as sd
    print("Playing audio...")
    sd.play(array, fs, blocking=True)
    print("Stop playing.\n")


def rms(vec):
    """
    get the root mean square of a vector
    :param vec: a numpy 1darray
    :return: the RMS
    """
    return np.sqrt(np.mean(vec * vec))


def synthesize(base, overlay, snr):
    """
    Synthesize 2 waveforms with the given SNR

    :param base: the base sound wave represented as numpy 1darray
    :param overlay: the overlay sound wave, assume the length is far larger than the base wave
    :param snr: the signal-noise ratio in dB, range between [-10, 20]
    :return: the synthesized sound wave with the same shape as `base`
    """
    assert -5 <= snr < 50
    noise_pre_scale = 1 - snr / 50
    if snr > 0:
        overlay = overlay * noise_pre_scale

    len_speech = base.shape[0]
    len_noise = overlay.shape[0]
    assert len_noise > len_speech

    start_point_noise = random.randint(0, len_noise - len_speech)
    overlay = overlay[start_point_noise: start_point_noise + len_speech]

    rms_overlay = rms(overlay)
    rms_base = rms(base)

    db_overlay = 20 * np.log10(rms_overlay + 1e-8)
    db_base = 20 * np.log10(rms_base)

    snr_origin = db_base - db_overlay
    db_adjust = snr - snr_origin
    scale_adjust = np.power(10, db_adjust / 20)

    output = overlay + base * scale_adjust
    return output
