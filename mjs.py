import os
import tensorflow as tf
from PIL import Image
import numpy as np


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

        # Słownik bez znaku '-' (padding), blank token na końcu
        self.vocab = "0123456789abcdefghijklmnopqrstuvwxyz"
        self.blank_token_idx = len(self.vocab)  # blank = ostatni indeks

        self.char_to_idx = {char: idx for idx, char in enumerate(self.vocab)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab) + 1  # +1 na blank

        self.pad_token_idx = -1  # padding do batchowania
    
    def encode_label(self, label):
        return [self.char_to_idx[c] for c in label.lower() if c in self.char_to_idx]

    def decode_label(self, indices):
        # Ignoruj blank i padding
        return ''.join([self.idx_to_char[idx] for idx in indices if idx in self.idx_to_char])

    def decode_predictions(self, predictions):
        decoded = tf.keras.backend.ctc_decode(
            predictions,
            input_length=tf.fill([tf.shape(predictions)[0]], tf.shape(predictions)[1]),
            greedy=True
        )[0][0]
        texts = []
        for sequence in decoded:
            # Ignoruj blank (ostatni indeks) i padding (-1)
            text = ''.join([self.idx_to_char[int(idx)] for idx in sequence if 0 <= idx < self.vocab_size - 1])
            texts.append(text)
        return texts

    def create_crnn_tf_dataset(self, samples, batch_size=32):
        def generator():
            for image_path, label in samples:
                image = self.preprocess_image(image_path)
                if image is not None:
                    encoded_label = self.encode_label(label)
                    yield image, encoded_label

        output_signature = (
            tf.TensorSpec(shape=(self.img_height, self.img_width, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32)
        )

        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=output_signature
        )
        dataset = dataset.padded_batch(
            batch_size,
            padded_shapes=([self.img_height, self.img_width, 1], [None]),
            padding_values=(0.0, self.pad_token_idx)  # padding -1 dla CRNN
        )
        return dataset

    def create_vit_tf_dataset(self, samples, batch_size=32, sequence_length=12):
        def generator():
            for image_path, label in samples:
                image = self.preprocess_image(image_path)
                if image is not None:
                    encoded_label = self.encode_label(label)
                    yield image, encoded_label

        output_signature = (
            tf.TensorSpec(shape=(self.img_height, self.img_width, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32)
        )

        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=output_signature
        )
        dataset = dataset.padded_batch(
            batch_size,
            padded_shapes=([self.img_height, self.img_width, 1], [sequence_length]),
            padding_values=(0.0, 0)  # padding 0 dla ViT
        )
        return dataset
    
            
    def load_annotation_file(self, annotation_path):
        """
        Ładuje plik z adnotacjami i zwraca listę krotek (ścieżka_obrazu, etykieta).
        Args:
            annotation_path (str): Ścieżka do pliku z adnotacjami.
        Returns:
            list: Lista krotek zawierających ścieżki do obrazów i odpowiadające im etykiety.
        """
        samples = []

        # Sprawdza, czy plik z adnotacjami istnieje.
        if not os.path.exists(annotation_path):
            print(f"Ostrzeżenie: Nie znaleziono pliku z adnotacjami: {annotation_path}")
            return samples

        # Otwiera i czyta plik z adnotacjami linia po linii.
        with open(annotation_path, 'r') as f:
            for line in f:
                line = line.strip()  # Usuwa białe znaki z początku i końca linii.
                if not line:
                    continue  # Pomija puste linie.

                # Obsługuje różne formaty adnotacji.
                if ' ' in line:
                    # Format: ścieżka_obrazu etykieta
                    parts = line.split(' ', 1)
                    if len(parts) == 2:
                        image_path, label = parts
                        # Upewnia się, że ścieżka do obrazu jest względna względem katalogu zbioru danych.
                        if not image_path.startswith(self.data_dir):
                            image_path = os.path.join(self.data_dir, "mnt/ramdisk/max/90kDICT32px", image_path)
                        samples.append((image_path, label.lower()))
                else:
                    # Obsługuje przypadki, gdy etykieta jest osadzona w nazwie pliku.
                    # Przykład: word_1_image_name.jpg
                    filename = os.path.basename(line)
                    if '_' in filename:
                        parts = filename.split('_')
                        if len(parts) >= 2:
                            label = parts[1]
                            image_path = os.path.join(self.data_dir, line)
                            samples.append((image_path, "mnt/ramdisk/max/90kDICT32px", label.lower()))

        print(f"Załadowano {len(samples)} próbek z {annotation_path}")
        return samples

    def load_data(self, split='train', limit: int | None = None):
        """
        Ładuje dane dla określonego podziału (np. train, validation, test).
        Args:
            split (str): Podział zbioru danych do załadowania (np. 'train', 'val', 'test').
        Returns:
            list: Lista poprawnych krotek (ścieżka_obrazu, etykieta).
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
        Args:
            label (str): Etykieta do sprawdzenia.
        Returns:
            bool: True, jeśli etykieta jest poprawna, False w przeciwnym razie.
        """
        return all(c in self.vocab for c in label.lower())

    def encode_label(self, label):
        """
        Koduje etykietę tekstową na sekwencję indeksów na podstawie słownika.
        Args:
            label (str): Etykieta do zakodowania.
        Returns:
            list: Lista indeksów reprezentujących etykietę.
        """
        return [self.char_to_idx[c] for c in label.lower() if c in self.char_to_idx]

    def decode_label(self, indices):
        """
        Dekoduje sekwencję indeksów z powrotem na etykietę tekstową.
        Args:
            indices (list): Lista indeksów do zdekodowania.
        Returns:
            str: Zdekodowana etykieta tekstowa.
        """
        return ''.join([self.idx_to_char[idx] for idx in indices if idx in self.idx_to_char])

    def decode_predictions(self, predictions):
        """
        Dekoduje przewidywania z modelu CTC na etykiety tekstowe.
        Args:
            predictions (tf.Tensor): Tensor zawierający przewidywania modelu.
        Returns:
            list: Lista zdekodowanych etykiet tekstowych.
        """
        # Używa funkcji dekodowania CTC z TensorFlow.
        decoded = tf.keras.backend.ctc_decode(
            predictions,
            input_length=tf.fill([tf.shape(predictions)[0]], tf.shape(predictions)[1])
        )[0][0]

        texts = []
        for sequence in decoded:
            # Konwertuje indeksy na znaki, ignorując token pusty (indeks 0).
            text = ''.join([self.idx_to_char[int(idx)] for idx in sequence if idx > 0])
            texts.append(text)
        return texts

    def preprocess_image(self, image_path):
        """
        Ładuje i przetwarza obraz dla modelu.
        Args:
            image_path (str): Ścieżka do pliku obrazu.
        Returns:
            np.ndarray: Przetworzony obraz jako tablica NumPy lub None w przypadku błędu.
        """
        try:
            # Otwiera obraz i konwertuje go na skalę szarości.
            image = Image.open(image_path).convert('L')

            # Zmienia rozmiar obrazu do określonych wymiarów.
            image = image.resize((self.img_width, self.img_height))

            # Normalizuje wartości pikseli do zakresu [0, 1].
            image = np.array(image, dtype=np.float32) / 255.0

            # Dodaje wymiar kanału do obrazu (wymagane dla modeli TensorFlow).
            image = np.expand_dims(image, axis=-1)
            return image
        except Exception as e:
            print(f"Błąd podczas ładowania obrazu {image_path}: {e}")
            return None

    # def create_tf_dataset(self, samples, batch_size=32):
    #     """
    #     Tworzy zbiór danych TensorFlow z listy próbek.
    #     Args:
    #         samples (list): Lista krotek (ścieżka_obrazu, etykieta).
    #         batch_size (int): Liczba próbek na batch.
    #     Returns:
    #         tf.data.Dataset: Zbiór danych TensorFlow gotowy do treningu lub ewaluacji.
    #     """

    #     def generator():
    #         """
    #         Funkcja generatora do zwracania przetworzonych obrazów i zakodowanych etykiet.
    #         """
    #         for image_path, label in samples:
    #             image = self.preprocess_image(image_path)
    #             if image is not None:
    #                 encoded_label = self.encode_label(label)
    #                 yield image, encoded_label

    #     # Definiuje sygnaturę wyjściową dla zbioru danych (kształty i typy obrazów i etykiet).
    #     output_signature = (
    #         tf.TensorSpec(shape=(self.img_height, self.img_width, 1), dtype=tf.float32),
    #         tf.TensorSpec(shape=(None,), dtype=tf.int32)
    #     )

    #     # Tworzy zbiór danych TensorFlow z generatora.
    #     dataset = tf.data.Dataset.from_generator(
    #         generator,
    #         output_signature=output_signature
    #     )

    #     # Wypełnia sekwencje (etykiety) i grupuje zbiór danych w batchach.
    #     dataset = dataset.padded_batch(
    #         batch_size,
    #         padded_shapes=([self.img_height, self.img_width, 1], [None]),
    #         padding_values=(0.0, self.pad_token_idx)  # Używa 0.0 do wypełniania obrazów i -1 do wypełniania etykiet.
    #     )

    #     return dataset