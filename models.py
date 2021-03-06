from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Activation, Input, Embedding
from keras.optimizers import RMSprop
from keras.models import load_model
from keras.layers.merge import Average
from keras.models import Model
import os

def M1_Embedding_128_256_relu(seq_length, num_inputs):
    model = Sequential()
    model.add(Embedding(input_shape=(seq_length-1,), input_dim=num_inputs, output_dim=512))
    model.add(LSTM(128, return_sequences=True))
    model.add(LSTM(256, activation='relu'))
    model.add(Dense(num_inputs))
    model.add(Activation('softmax'))

    optimizer = RMSprop(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])
    return model

def M2_Embedding_32_64_relu(seq_length, num_inputs):
    model = Sequential()
    model.add(Embedding(input_shape=(seq_length-1,), input_dim=num_inputs, output_dim=512))
    model.add(LSTM(32, return_sequences=True))
    model.add(LSTM(64, activation='relu'))
    model.add(Dense(num_inputs))
    model.add(Activation('softmax'))

    optimizer = RMSprop(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])
    return model

def M3_Embedding_32_64_64_64_relu(seq_length, num_inputs):
    model = Sequential()
    model.add(Embedding(input_shape=(seq_length-1,), input_dim=num_inputs, output_dim=512))
    model.add(LSTM(32, return_sequences=True))
    model.add(LSTM(64, activation='relu', return_sequences=True))
    model.add(LSTM(64, activation='relu', return_sequences=True))
    model.add(LSTM(64, activation='relu'))
    model.add(Dense(num_inputs))
    model.add(Activation('softmax'))

    optimizer = RMSprop(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])
    return model


def M4_Embedding_256_512_relu(seq_length, num_inputs):
    model = Sequential()
    model.add(Embedding(input_shape=(seq_length-1,), input_dim=num_inputs, output_dim=512))
    model.add(LSTM(256, return_sequences=True))
    model.add(LSTM(512, activation='relu'))
    model.add(Dense(num_inputs))
    model.add(Activation('softmax'))

    optimizer = RMSprop(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])
    return model

def M5_128(seq_length, num_inputs):
    model = Sequential()
    model.add(LSTM(128, input_shape=(seq_length-1, num_inputs)))
    model.add(Dense(num_inputs))
    model.add(Activation('softmax'))

    optimizer = RMSprop(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])
    return model

def M6_128_256_relu(seq_length, num_inputs):
    model = Sequential()
    model.add(LSTM(128, input_shape=(seq_length-1, num_inputs), return_sequences=True))
    model.add(LSTM(256, activation='relu'))
    model.add(Dense(num_inputs))
    model.add(Activation('softmax'))

    optimizer = RMSprop(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])
    return model

def M7_128_256_relu_dropout(seq_length, num_inputs):
    model = Sequential()
    model.add(LSTM(128, input_shape=(seq_length-1, num_inputs), return_sequences=True))
    model.add(LSTM(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_inputs))
    model.add(Activation('softmax'))

    optimizer = RMSprop(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])
    return model

def M8_M5_average_merge(models_folder):
    truncated_models_outputs = []
    models = []
    for model_file in os.listdir(models_folder):
        # Load single model
        model_path = os.path.join(models_folder, model_file)
        current_model = load_model(model_path)
        models.append(current_model)

        # Get rid of softmax activation layer so we get raw percentages
        current_model.pop()
        truncated_models_outputs.append(current_model.layers[-1].output)

    new_models = []
    input_layer = Input(shape=(SEQUENCE_LENGTH-1, len(labels)))
    for model in models:
        model = model(input_layer)
        new_models.append(model)
    
    # Merge all truncated models and add softmax at the end
    output = Average()([model for model in new_models])
    output = Activation("softmax", name="predictions")(output)

    model = Model(input_layer, output)
    optimizer = RMSprop(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])
    return model    



MODELS = {"M1": M1_Embedding_128_256_relu,
          "M2": M2_Embedding_32_64_relu,
          "M3": M3_Embedding_32_64_64_64_relu,
          "M4": M4_Embedding_256_512_relu,
          "M5": M5_128,
          "M6": M6_128_256_relu,
          "M7": M7_128_256_relu_dropout,
          "M8": M8_M5_average_merge}


def build_model(model_name, *args):
    model = MODELS[model_name](*args)
    return model