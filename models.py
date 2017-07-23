from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Activation, Input, Embedding
from keras.optimizers import RMSprop


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


MODELS = {"M1": M1_Embedding_128_256_relu,
		  "M2": M2_Embedding_32_64_relu,
		  "M3": M3_Embedding_32_64_64_64_relu,
		  "M4": M4_Embedding_256_512_relu,
		  "M5": M5_128,
		  "M6": M6_128_256_relu,
		  "M7": M7_128_256_relu_dropout}


def build_model(model_name, seq_length, num_inputs):
	model = MODELS[model_name](seq_length, num_inputs)
	return model