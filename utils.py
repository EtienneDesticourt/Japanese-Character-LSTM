from keras.callbacks import ModelCheckpoint


def get_save_callback(model_id):
        model_name = "weights\\LSTM_%s_.{epoch:02d}-{acc:.3f}-{loss:.3f}-{val_acc:.3f}-{val_loss:.3f}.h5" % model_id
        return ModelCheckpoint(model_name, monitor='acc', verbose=1, save_best_only=False)

def encode(character, labels):
    if character in labels:
        return labels.index(character)
    return len(labels)-1