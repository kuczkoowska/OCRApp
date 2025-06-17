import os
import tensorflow as tf
from PIL import Image
import numpy as np

# Klasa do ładowania i przygotowania danych MJSynth
class MJSynthDataLoader:
    def __init__(self, data_dir, img_height=32, img_width=128):
        """
        Inicjalizuje loader danych z katalogiem zawierającym zbiór danych i wymiarami obrazów.
        Args:
            data_dir (str): Ścieżka do katalogu ze zbiorem danych.
            img_height (int): Wysokość obrazów po zmianie rozmiaru.
            img_width (int): Szerokość obrazów po zmianie rozmiaru.
        """
        self.data_dir = data_dir
        self.img_height = img_height
        self.img_width = img_width

        # Słownik znaków - tylko litery łacińskie, blank token na końcu
        self.vocab = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        self.blank_token_idx = len(self.vocab)

        # Mapowanie znaków na indeksy i odwrotnie
        self.char_to_idx = {char: idx for idx, char in enumerate(self.vocab)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab) + 1  # +1 na blank

        self.pad_token_idx = -1  # padding do batchowania
    
    def decode_label(self, indices):
        # Zamienia sekwencję indeksów na tekst, ignoruje padding i blank
        return ''.join([self.idx_to_char[idx] for idx in indices if idx in self.idx_to_char and idx != self.pad_token_idx])

    def decode_predictions(self, predictions):
        """
        Dekoduje predykcje modelu CTC do tekstu.
        """
        decoded_texts = []
        
        for pred in predictions:
            # Wybierz najbardziej prawdopodobny znak w każdym kroku czasowym
            decoded_indices = tf.argmax(pred, axis=-1).numpy()
            
            # Usuń powtórzenia i blank tokeny
            decoded_sequence = []
            prev_idx = -1
            
            for idx in decoded_indices:
                if idx != prev_idx and idx != self.blank_token_idx:
                    decoded_sequence.append(idx)
                prev_idx = idx
            
            # Zamień indeksy na znaki
            decoded_text = ''.join([
                self.idx_to_char.get(idx, '') 
                for idx in decoded_sequence 
                if idx in self.idx_to_char
            ])
            
            decoded_texts.append(decoded_text)
        
        return decoded_texts

    def create_crnn_tf_dataset(self, samples, batch_size=32):
        # Funkcja do preprocessingu pojedynczego przykładu
        def preprocess_crnn(image_path, label):
            image = tf.io.read_file(image_path)
            image = tf.image.decode_image(image, channels=1)
            image.set_shape([None, None, 1])
            image = tf.image.resize(image, [self.img_height, self.img_width])
            image = tf.cast(image, tf.float32) / 255.0

            label_indices = tf.py_function(
                self.encode_label, [label], tf.int32
            )
            label_indices.set_shape([None])
            return image, label_indices

        # Przygotuj listy ścieżek i etykiet
        image_paths = [sample[0] for sample in samples]
        labels = [sample[1] for sample in samples]
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
        dataset = dataset.map(preprocess_crnn, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.padded_batch(
            batch_size,
            padded_shapes=([self.img_height, self.img_width, 1], [None]),
            padding_values=(0.0, self.pad_token_idx)
        )
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    def load_annotation_file(self, annotation_path):
        """
        Ładuje próbki z pliku adnotacji, wyciągając słowa z nazw plików.
        """
        samples = []

        if not os.path.exists(annotation_path):
            print(f"Ostrzeżenie: Nie znaleziono pliku z adnotacjami: {annotation_path}")
            return samples

        with open(annotation_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # Rozbij linię na ścieżkę i etykietę
                parts = line.split(' ')
                image_path = parts[0]

                # Pobierz nazwę pliku
                filename = os.path.basename(image_path)

                # Wyciągnij słowo z nazwy pliku - między pierwszym a drugim '_'
                if '_' in filename:
                    word = filename.split('_')[1]
                else:
                    word = os.path.splitext(filename)[0]

                if not image_path.startswith(self.data_dir):
                    image_path = os.path.join(self.data_dir, "mnt/ramdisk/max/90kDICT32px", image_path)

                samples.append((image_path, word))

        print(f"Załadowano {len(samples)} próbek z {annotation_path}")
        return samples

    def load_data(self, split='train', limit: int | None = None):
        """
        Ładuje dane dla określonego podziału (np. train, validation, test).
        """
        # Tworzy ścieżkę do pliku z adnotacjami dla danego podziału.
        annotation_file = f"mnt/ramdisk/max/90kDICT32px/annotation_{split}.txt"
        annotation_path = os.path.join(self.data_dir, annotation_file)

        print(f"Ładowanie zbioru {split} z {annotation_path}")

        # Ładuje próbki z pliku z adnotacjami.
        samples = self.load_annotation_file(annotation_path)

        # Filtruje próbki, aby uwzględnić tylko te z poprawnymi ścieżkami do obrazów i etykietami.
        valid_samples = []
        for i, (image_path, label) in enumerate(samples):
            if limit is not None and i >= limit:
                print(f"Osiągnięto limit {limit} próbek dla {split}.")
                break

            if os.path.exists(image_path) and self.is_valid_label(label):
                valid_samples.append((image_path, label))

        print(f"Załadowano {len(valid_samples)} próbek dla {split}")
        return valid_samples

    def is_valid_label(self, label):
        """
        Sprawdza, czy etykieta zawiera tylko dozwolone znaki ze słownika.
        """
        return all(c in self.vocab for c in label)

    def encode_label(self, label):
        """
        Koduje etykietę tekstową na sekwencję indeksów na podstawie słownika.
        """
        if isinstance(label, tf.Tensor):
            label = label.numpy().decode("utf-8")
        return np.array([self.char_to_idx[c] for c in label if c in self.char_to_idx], dtype=np.int32)

    def preprocess_image(self, image_path):
        try:
            # Otwiera obraz i konwertuje go na skalę szarości.
            image = Image.open(image_path).convert('L')

            # Zmienia rozmiar obrazu do określonych wymiarów.
            image = image.resize((self.img_width, self.img_height))

            # Normalizuje wartości pikseli do zakresu [0, 1].
            image = np.array(image, dtype=np.float32) / 255.0

            # Dodaje wymiar kanału do obrazu.
            image = np.expand_dims(image, axis=-1)
            return image
        except Exception as e:
            print(f"Błąd podczas ładowania obrazu {image_path}: {e}")
            return None