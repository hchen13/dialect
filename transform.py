import os

import numpy as np

from loader import get_small_speeches, assemble_matrices, play, draw_wave, SPEECH_ROOT, read_wav, NOISE_ROOT, \
    synthesize, normalize
from prototypes import wavenet


def load_model(weights_file, input_length):
    print("Creating model and loading weights...")
    model = wavenet(input_length)
    model.load_weights(weights_file)
    print("Model creation complete!\n")
    return model


def denoise(wave):
    from oct2py import octave
    output = octave.feval('logmmse', np.trim_zeros(wave), 16000)
    return np.float32(np.squeeze(output))


if __name__ == '__main__':
    input_length = 4000

    noise_file = 'city.wav'
    noise = read_wav(os.path.join(NOISE_ROOT, noise_file))
    speech = read_wav(os.path.join(SPEECH_ROOT, 'cd2md/input', '10.wav'))
    # speech = read_wav('test.wav')[8000:]
    speech = normalize(speech)
    mix = synthesize(speech, noise, 20)
    sample = normalize(mix)

    # sample, _ = get_small_speeches()[0]
    sample = denoise(sample)
    play(sample)

    feed, _ = assemble_matrices(sample, sample, input_length, shift=input_length)

    # model = load_model('models/final_weights.h5', input_length)
    model = load_model('models/wednesday.h5', input_length)

    predictions = model.predict(np.expand_dims(feed, axis=2), batch_size=8)
    output = predictions.flatten()
    denoised = denoise(output)
    draw_wave(sample, output, denoised)
    print("Playing the transformed audio")
    play(normalize(output))
    print("Playing the denoised transform")
    play(normalize(denoised))
