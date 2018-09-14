from loader import generate_train_data
from prototypes import wavenet
from utils import ensure_dir_exists

if __name__ == '__main__':
    input_length = 4000
    epochs = 500
    ensure_dir_exists('models/')

    model = wavenet(input_length)

    for e in range(epochs):
        train_batches = generate_train_data(input_length, 1000)
        print("Epoch {}/{}:".format(e + 1, epochs))
        for i, (x, y) in enumerate(train_batches):
            model.fit(x, y, batch_size=4, epochs=1, verbose=2)

        if (e + 1) % 50 == 0:
            print("Saving intermediate model weights...")
            model.save_weights('models/tmp.h5')

    print("\nTraining complete!\nSaving model...")
    model.save_weights('models/final_weights.h5')
    print("Model saved, terminate.")
