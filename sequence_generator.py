from utils import encode
import numpy as np

class SequenceGenerator(object):

    def __init__(self, text, sequence_length, stride, batch_size, labels):
        self.text = text
        self.sequence_length = sequence_length
        self.stride = stride
        self.batch_size = batch_size
        self.labels = labels
        self.num_labels = len(labels)
        self.pos_in_text = 0

    def __len__(self):
        return 2

    def __iter__(self):
        return self

    def __next__(self):
        return self._build_batch()

    def _build_batch(self):
        text_size_for_batch = self.batch_size * self.stride

        if self.pos_in_text + text_size_for_batch >= len(self.text):
            self.pos_in_text = 0

        x = np.zeros((self.batch_size, self.sequence_length - 1, self.num_labels), dtype=np.bool)
        y = np.zeros((self.batch_size, self.num_labels), dtype=np.bool)
        
        for sample_id, seq_start in enumerate(range(0, self.batch_size * self.stride - self.stride, self.stride)):
            for char_id in range(self.sequence_length - 1):
                character = self.text[self.pos_in_text + seq_start + char_id]
                label_index = encode(character, self.labels)
                x[sample_id, char_id, label_index] = 1

            last_character = self.text[self.pos_in_text + seq_start + char_id + 1]
            label_index = encode(last_character, self.labels)
            y[sample_id, label_index] = 1

        self.pos_in_text += text_size_for_batch

        return x, y