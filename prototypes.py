from keras import Input, Model
from keras.layers import Conv1D, multiply, add, Activation, Flatten, Dense
from keras.losses import mean_squared_error, mean_absolute_error
from keras.optimizers import Adam


def l1_l2_loss(w1, w2):
    def f(y_true, y_pred):
        loss = mean_absolute_error(y_true, y_pred) * w1 + \
               mean_squared_error(y_true, y_pred) * w2
        return loss
    return f


def wavenet_block(n_atrous_filters, atrous_filter_size, dilation_rate):
    def f(input_tensor):
        residual = input_tensor
        tanh_out = Conv1D(
            n_atrous_filters,
            atrous_filter_size,
            dilation_rate=dilation_rate,
            padding='same',
            activation='tanh'
        )(input_tensor)
        sigmoid_out = Conv1D(
            n_atrous_filters,
            atrous_filter_size,
            dilation_rate=dilation_rate,
            padding='same',
            activation='sigmoid'
        )(input_tensor)
        merged = multiply([tanh_out, sigmoid_out])
        skip_out = Conv1D(128, 1, activation='relu', padding='same')(merged)
        out = add([skip_out, residual])
        return out, skip_out
    return f


def wavenet(input_shape, dilation_blocks=30):
    dialect = Input(shape=(input_shape, 1), name='dialect')
    out, skip_out = wavenet_block(128, 3, 1)(dialect)
    skip_connections = [skip_out]
    for i in range(dilation_blocks):
        out, skip_out = wavenet_block(128, 3, 2 ** ((i + 1) % 10))(out)
        skip_connections.append(skip_out)
    net = add(skip_connections)
    net = Activation('relu')(net)
    net = Conv1D(2048, 1, activation='relu', padding='same')(net)
    net = Activation('relu')(net)
    net = Conv1D(256, 1)(net)
    mandarin = Conv1D(1, 1, padding='same', activation='tanh', name='mandarin')(net)

    model = Model(inputs=dialect, outputs=mandarin)
    optimizer = Adam(lr=.0001, decay=0.0, epsilon=1e-8)
    model.compile(loss=l1_l2_loss(0, 10), optimizer=optimizer)
    return model
