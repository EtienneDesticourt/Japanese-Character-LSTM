from keras.callbacks import ModelCheckpoint
import numpy as np

def get_save_callback(model_id):
        model_name = "weights\\LSTM_%s_.{epoch:02d}-{acc:.3f}-{loss:.3f}-{val_acc:.3f}-{val_loss:.3f}.h5" % model_id
        return ModelCheckpoint(model_name, monitor='acc', verbose=1, save_best_only=False)

def encode(character, labels):
    if character in labels:
        return labels.index(character)
    return len(labels)-1

def build_sequence_array(text, num_samples, labels, sequence_length, stride, verbose=True):
    if num_samples * stride >= len(text):
        raise ValueError("Text isn't long enough to accomodate %s samples and a stride of %s." % (num_samples, stride))

    num_labels = len(labels)
    x = np.zeros((num_samples, sequence_length - 1, num_labels), dtype=np.bool)
    y = np.zeros((num_samples, num_labels), dtype=np.bool)

    for sample_id, seq_start in enumerate(range(0, num_samples * stride, stride)):

        for i in range(sequence_length - 1):
            character = text[seq_start + i]
            label_index = encode(character, labels)
            x[sample_id, i, label_index] = 1

        last_character = text[seq_start + i + 1]
        label_index = encode(last_character, labels)
        y[sample_id, label_index] = 1

        if verbose and sample_id % 2000 == 0:
            print(sample_id)
            
    return x, y
