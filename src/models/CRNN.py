import tensorflow as tf
from tensorflow.keras import layers, Model

# Zdefiniowanie klasy dla modelu CRNN (Convolutional Recurrent Neural Network)
class CRNNModel:
    def __init__(self, vocab_size, img_height=32, img_width=128):
        """
        Inicjalizacja modelu CRNN z rozmiarem słownika i wymiarami obrazu.
        Argumenty:
            vocab_size (int): Liczba znaków w słowniku (bez blank!).
            img_height (int): Wysokość wejściowego obrazu.
            img_width (int): Szerokość wejściowego obrazu.
        """
        self.vocab_size = vocab_size
        self.img_height = img_height
        self.img_width = img_width
        
    def build_model(self):
        """
        Budowanie architektury CRNN.
        Zwraca:
            model (tf.keras.Model): Model CRNN.
        """
        # Warstwa wejściowa dla obrazów w skali szarości (1 kanał)
        inputs = layers.Input(shape=(self.img_height, self.img_width, 1))
        
        # Ekstrakcja cech za pomocą CNN
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 1))(x)
        x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 1))(x)
        
        # Przekształcenie dla wejścia RNN
        x = layers.Permute((2, 1, 3))(x)
        x = layers.Reshape((-1, x.shape[-2] * x.shape[-1]))(x)
        x = layers.Dense(64, activation='relu')(x)
        
        # Dwukierunkowe warstwy LSTM dla modelowania sekwencji
        x = layers.Bidirectional(layers.LSTM(256, return_sequences=True, dropout=0.25))(x)
        x = layers.Bidirectional(layers.LSTM(256, return_sequences=True, dropout=0.25))(x)
        
        # Warstwa wyjściowa z aktywacją softmax dla prawdopodobieństw znaków
        # Rozmiar wyjścia to vocab_size (liczba znaków) + 1 dla tokenu CTC blank
        outputs = layers.Dense(self.vocab_size + 1, activation='softmax')(x)
        
        # Tworzenie modelu
        model = Model(inputs, outputs, name='CRNN')
        return model

# Definicja funkcji straty CTC (Connectionist Temporal Classification)
def ctc_loss_func(blank_token_idx):
    """
    Obliczanie straty CTC dla modelu.
    Argumenty:
        y_true (tensor): Prawdziwe etykiety.
        y_pred (tensor): Przewidywane wyjścia z modelu.
    Zwraca:
        loss (tensor): Obliczona strata CTC.
    """
    def loss(y_true, y_pred):
        # Zamień -1 (padding) na blank_token_idx (blank na końcu słownika)
        y_true_no_pad = tf.where(y_true == -1, blank_token_idx, y_true)
        # Długości etykiet (bez paddingu)
        label_length = tf.math.reduce_sum(tf.cast(tf.not_equal(y_true, -1), tf.int32), axis=1)
        label_length = tf.expand_dims(label_length, axis=1)
        # Długości predykcji (wszystkie time steps)
        input_length = tf.fill([tf.shape(y_pred)[0]], tf.shape(y_pred)[1])
        input_length = tf.expand_dims(input_length, axis=1)
        return tf.keras.backend.ctc_batch_cost(y_true_no_pad, y_pred, input_length, label_length)
    return loss