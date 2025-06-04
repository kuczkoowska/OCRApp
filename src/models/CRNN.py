import tensorflow as tf
from tensorflow.keras import layers, Model

# Zdefiniowanie klasy dla modelu CRNN (Convolutional Recurrent Neural Network)
class CRNNModel:
    def __init__(self, vocab_size, img_height=32, img_width=128):
        """
        Inicjalizacja modelu CRNN z rozmiarem słownika i wymiarami obrazu.
        Argumenty:
            vocab_size (int): Liczba znaków w słowniku.
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
        # Pierwsza warstwa konwolucyjna z 64 filtrami, jądrem 3x3, aktywacją ReLU i paddingiem 'same'
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
        # Warstwa max pooling zmniejszająca wymiary przestrzenne o połowę
        x = layers.MaxPooling2D((2, 2))(x)
        
        # Druga warstwa konwolucyjna z 128 filtrami, jądrem 3x3, aktywacją ReLU i paddingiem 'same'
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        # Warstwa max pooling dalej zmniejszająca wymiary przestrzenne
        x = layers.MaxPooling2D((2, 2))(x)
        
        # Trzecia warstwa konwolucyjna z 256 filtrami, jądrem 3x3, aktywacją ReLU i paddingiem 'same'
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        # Normalizacja wsadowa stabilizująca trening i poprawiająca zbieżność
        x = layers.BatchNormalization()(x)
        
        # Czwarta warstwa konwolucyjna z 256 filtrami, jądrem 3x3, aktywacją ReLU i paddingiem 'same'
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        # Warstwa max pooling ze stride (2, 1) zmniejszająca wysokość, ale pozostawiająca szerokość bez zmian
        x = layers.MaxPooling2D((2, 1))(x)
        
        # Piąta warstwa konwolucyjna z 512 filtrami, jądrem 3x3, aktywacją ReLU i paddingiem 'same'
        x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
        # Normalizacja wsadowa stabilizująca trening
        x = layers.BatchNormalization()(x)
        
        # Szósta warstwa konwolucyjna z 512 filtrami, jądrem 3x3, aktywacją ReLU i paddingiem 'same'
        x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
        # Warstwa max pooling ze stride (2, 1) dalej zmniejszająca wysokość, pozostawiając szerokość bez zmian
        x = layers.MaxPooling2D((2, 1))(x)
        
        # Przekształcenie dla wejścia RNN
        # Permutacja wymiarów na (batch, szerokość, wysokość, kanały) dla kompatybilności
        x = layers.Permute((2, 1, 3))(x)

        # Przekształcenie dla przetwarzania przez RNN
        x = layers.Reshape((-1, x.shape[-2] * x.shape[-1]))(x)

        # Warstwa Dense zmniejszająca wymiarowość cech dla wejścia RNN
        x = layers.Dense(64, activation='relu')(x)
        
        # Dwukierunkowe warstwy LSTM dla modelowania sekwencji
        # Pierwsza dwukierunkowa warstwa LSTM z 256 jednostkami i dropoutem dla regularizacji
        x = layers.Bidirectional(layers.LSTM(256, return_sequences=True, dropout=0.25))(x)
        # Druga dwukierunkowa warstwa LSTM z 256 jednostkami i dropoutem dla regularizacji
        x = layers.Bidirectional(layers.LSTM(256, return_sequences=True, dropout=0.25))(x)
        
        # Warstwa wyjściowa z aktywacją softmax dla prawdopodobieństw znaków
        # Rozmiar wyjścia to vocab_size (liczba znaków) + 1 dla tokenu CTC blank
        outputs = layers.Dense(self.vocab_size + 1, activation='softmax')(x)
        
        # Tworzenie modelu
        model = Model(inputs, outputs, name='CRNN')
        return model

# Definicja funkcji straty CTC (Connectionist Temporal Classification)
def ctc_loss_func(y_true, y_pred):
    """
    Obliczanie straty CTC dla modelu.
    Argumenty:
        y_true (tensor): Prawdziwe etykiety.
        y_pred (tensor): Przewidywane wyjścia z modelu.
    Zwraca:
        loss (tensor): Obliczona strata CTC.
    """
    # Pobranie rozmiaru batcha
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    # Pobranie długości sekwencji wejściowej (szerokość mapy cech)
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    # Pobranie długości sekwencji etykiet
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

    # Tworzenie tensorów dla długości wejść i etykiet
    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

    # Obliczanie straty CTC za pomocą funkcji backendu TensorFlow
    loss = tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return loss