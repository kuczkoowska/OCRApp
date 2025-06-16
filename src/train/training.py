import sys
import os
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import json
import tensorflow as tf
from pathlib import Path
from src.train.mjs import MJSynthDataLoader
from src.models.CRNN import CRNNModel, ctc_loss_func
from src.models.vit import ViTOCRModel
import matplotlib.pyplot as plt
import math
import datetime

# Klasa trenera dla modeli OCR
class OCRTrainer:
    def __init__(self, data_loader, model_type='crnn'):
        """
        Inicjalizuje OCRTrainer z loaderem danych i typem modelu.
        Args:
            data_loader: Instancja loadera danych.
            model_type: Typ modelu do trenowania ('crnn' lub 'vit').
        """
        self.data_loader = data_loader
        self.model_type = model_type
        self.history = None  # Do przechowywania historii treningu

    def build_model(self):
        """Buduje określony model na podstawie typu modelu."""
        if self.model_type == 'crnn':
            # Budowanie modelu CRNN
            model_builder = CRNNModel(self.data_loader.vocab_size)
            self.model = model_builder.build_model()

            # Kompilacja modelu z funkcją straty CTC
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss=ctc_loss_func(self.data_loader.blank_token_idx),
                run_eagerly=True
            )

        elif self.model_type == 'vit':
            # Budowanie modelu ViT
            model_builder = ViTOCRModel(self.data_loader.vocab_size)
            self.model = model_builder.build_model()

            # Kompilacja modelu z funkcją straty sparse categorical crossentropy
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )

    def train(self, train_dataset, val_dataset, train_samples, val_samples, batch_size, epochs=10, save_path='models/'):
        """
        Trenuje model na podanych zbiorach danych.
        Args:
            train_dataset: Zbiór danych treningowych.
            val_dataset: Zbiór danych walidacyjnych.
            train_samples: Liczba próbek treningowych.
            val_samples: Liczba próbek walidacyjnych.
            batch_size: Rozmiar batcha.
            epochs: Liczba epok treningu.
            save_path: Katalog do zapisu wytrenowanego modelu.
        """
        # Tworzenie katalogu do zapisu modeli
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        model_dir = f'{save_path}/{self.model_type}_{timestamp}'
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model configuration
        with open(f'{model_dir}/config.json', 'w') as f:
            json.dump({
                'model_type': self.model_type,
                'batch_size': batch_size,
                'epochs': epochs,
                'train_samples': train_samples,
                'val_samples': val_samples,
                'timestamp': timestamp
            }, f, indent=2)

        # Obliczanie liczby kroków na epokę z obsługą częściowych batchy
        steps_per_epoch = math.ceil(train_samples / batch_size)
        validation_steps = math.ceil(val_samples / batch_size)
        
        # Definiowanie callbacków do treningu
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=f'{model_dir}/{self.model_type}_best.keras',
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir=f'{model_dir}/logs',
                histogram_freq=1
            )
        ]

        # Trenowanie modelu i zapis historii treningu
        print(f"Rozpoczęcie treningu modelu {self.model_type}...")
        print(f"Liczba próbek treningowych: {train_samples}")
        print(f"Liczba próbek walidacyjnych: {val_samples}")
        print(f"Liczba kroków na epokę: {steps_per_epoch}")
        print(f"Liczba kroków walidacyjnych: {validation_steps}")
        
        self.history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1
        )

        # Save final model
        self.model.save(f'{model_dir}/{self.model_type}_final.keras')
        
        return self.history

    def plot_training_history(self, save_path='results/'):
        """
        Rysuje historię treningu (strata i dokładność).
        Args:
            save_path: Katalog do zapisu wykresu.
        """
        if self.history is None:
            print("Brak dostępnej historii treningu")
            return

        # Tworzenie katalogu do zapisu wyników
        os.makedirs(save_path, exist_ok=True)

        # Tworzenie wykresów dla straty i dokładności
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Wykres straty treningowej i walidacyjnej
        axes[0].plot(self.history.history['loss'], label='Strata treningowa')
        axes[0].plot(self.history.history['val_loss'], label='Strata walidacyjna')
        axes[0].set_title('Strata modelu')
        axes[0].set_xlabel('Epoka')
        axes[0].set_ylabel('Strata')
        axes[0].legend()

        # Wykres dokładności treningowej i walidacyjnej (jeśli dostępne)
        if 'accuracy' in self.history.history:
            axes[1].plot(self.history.history['accuracy'], label='Dokładność treningowa')
            axes[1].plot(self.history.history['val_accuracy'], label='Dokładność walidacyjna')
            axes[1].set_title('Dokładność modelu')
            axes[1].set_xlabel('Epoka')
            axes[1].set_ylabel('Dokładność')
            axes[1].legend()
        else:
            # Jeśli dokładność nie jest dostępna (np. dla straty CTC), wyświetl komunikat
            axes[1].text(0.5, 0.5, 'Brak dostępnych metryk dokładności\n(strata CTC)',
                         ha='center', va='center', transform=axes[1].transAxes)

        plt.tight_layout()
        plt.savefig(f'{save_path}/{self.model_type}_training_history.png')
        plt.show()


def main():
    # Konfiguracja procesu treningu
    config = {
        'mjsynth_path': '/home/jagoda/studia/inteloblicz/OCRAppModels/mjsynth',
        'batch_size': 8,
        'epochs': 10,
        'img_height': 32,
        'img_width': 128,
        'data_amount_limit': 500
    }

    print("Inicjalizacja loadera danych MJSynth...")
    # Inicjalizacja loadera danych
    data_loader = MJSynthDataLoader(
        config['mjsynth_path'],
        img_height=config['img_height'],
        img_width=config['img_width']
    )

    # Ładowanie zbiorów danych treningowych, walidacyjnych i testowych
    print("Ładowanie danych treningowych...")
    train_samples = data_loader.load_data('train', config['data_amount_limit'])
    samples = data_loader.load_data('test', limit=5)
    for path, label in samples:
        print(f"Ścieżka: {path}")
        print(f"Etykieta: {label}")
    # Add this debug section after loading data:
    print("\nTesting label encoding:")
    for path, label in samples[:3]:
        encoded = data_loader.encode_label(label)
        decoded = data_loader.decode_label(encoded)
        print(f"Original: '{label}' → Encoded: {encoded} → Decoded: '{decoded}'")
    print("Ładowanie danych walidacyjnych...")
    val_samples = data_loader.load_data('val', config['data_amount_limit'])

    print("Ładowanie danych testowych...")
    test_samples = data_loader.load_data('test', config['data_amount_limit'])

    # Lista modeli do trenowania
    models_to_train = ['crnn']
    results = {}  # Słownik do przechowywania wyników dla każdego modelu

    for model_type in models_to_train:
        print(f"\n{'=' * 50}")
        print(f"Trenowanie modelu {model_type.upper()}")
        print(f"{'=' * 50}")

        # Przygotowanie datasetów zależnie od modelu
        if model_type == 'vit':
            train_dataset = data_loader.create_vit_tf_dataset(
                train_samples, batch_size=config['batch_size'], sequence_length=12, for_vit=True
            ).repeat()
            val_dataset = data_loader.create_vit_tf_dataset(
                val_samples, batch_size=config['batch_size'], sequence_length=12, for_vit=True
            ).repeat()
            test_dataset = data_loader.create_vit_tf_dataset(
                test_samples, batch_size=config['batch_size'], sequence_length=12, for_vit=True
            )
        else:  # CRNN
            train_dataset = data_loader.create_crnn_tf_dataset(
                train_samples, batch_size=config['batch_size']
            ).repeat()
            print("\n[DEBUG] Checking a batch of images and labels from train_dataset:")
            for images, labels in train_dataset.take(1):
                print("Images shape:", images.shape)
                print("Labels shape:", labels.shape)
                print("First label indices:", labels[0].numpy())
                print("First label decoded:", data_loader.decode_label(labels[0].numpy()))
            val_dataset = data_loader.create_crnn_tf_dataset(
                val_samples, batch_size=config['batch_size']
            ).repeat()  # Added .repeat() for consistency
            test_dataset = data_loader.create_crnn_tf_dataset(
                test_samples, batch_size=config['batch_size']
            )

        # Inicjalizacja trenera dla bieżącego typu modelu
        trainer = OCRTrainer(data_loader, model_type=model_type)
        trainer.build_model()
        print("\n[DEBUG] Model weights mean (first 3 layers):", [w.mean() for w in trainer.model.get_weights()[:3]])

        # Wyświetlenie podsumowania modelu
        print(f"Podsumowanie modelu dla {model_type}:")
        trainer.model.summary()

        # Trenowanie modelu
        history = trainer.train(
            train_dataset,
            val_dataset,
            train_samples=len(train_samples),
            val_samples=len(val_samples),
            batch_size=config['batch_size'],
            epochs=config['epochs'],
            save_path='models'
        )
        
        for batch_images, batch_labels in test_dataset.take(2):  # <-- pobierz 2 batch'e
            preds = trainer.model.predict(batch_images)
            decoded = data_loader.decode_predictions(preds)
            for j in range(min(2, len(batch_labels))):  # <-- wyświetl 2 próbki z batcha
                print(f"True label   : {data_loader.decode_label(batch_labels[j].numpy())}")
                print(f"Predicted    : {decoded[j]}")

                input_len = np.ones(preds.shape[0]) * preds.shape[1]
                decoded_ctc, _ = tf.keras.backend.ctc_decode(preds, input_length=input_len, greedy=True)

                decoded_indices = decoded_ctc[0][j].numpy()
                decoded_text = ''.join([data_loader.idx_to_char.get(idx, '') for idx in decoded_indices if idx != data_loader.blank_token_idx])
                print(f"[CTC_DECODE] Predicted: '{decoded_text}'")
                print("\n[DEBUG] Surowe predykcje (logity/probabilities) dla tej próbki:")
                print(preds[j])
                print("-" * 40)

        print("Training loss history:", history.history['loss'])
        print("Validation loss history:", history.history['val_loss'])
        # Rysowanie historii treningu
        trainer.plot_training_history(save_path=f'results/{model_type}_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}')

        # Ocena modelu na zbiorze testowym
        print(f"Ewaluacja modelu {model_type.upper()}...")
        test_steps = math.ceil(len(test_samples) / config['batch_size'])
        try:
            test_loss = trainer.model.evaluate(test_dataset, steps=test_steps, verbose=1)
            print(f"Strata testowa: {test_loss}")
        except Exception as e:
            print(f"Błąd podczas ewaluacji modelu: {e}")
            test_loss = None

        # Wyświetlenie surowych predykcji na testowym batchu
        try:
            for batch_images, _ in test_dataset.take(1):
                predictions = trainer.model.predict(batch_images)
                print("Surowe predykcje (pierwsza próbka):", predictions[0])
        except Exception as e:
            print(f"Błąd podczas generowania predykcji: {e}")

        # Przechowywanie wyników dla bieżącego modelu
        results[model_type] = {
            'final_train_loss': history.history['loss'][-1],
            'final_val_loss': history.history['val_loss'][-1],
            'test_loss': test_loss if isinstance(test_loss, float) else (test_loss[0] if isinstance(test_loss, list) else None)
        }

        # Dodanie metryk dokładności, jeśli dostępne
        if 'accuracy' in history.history:
            results[model_type]['final_train_acc'] = history.history['accuracy'][-1]
            results[model_type]['final_val_acc'] = history.history['val_accuracy'][-1]
            if isinstance(test_loss, list) and len(test_loss) > 1:
                results[model_type]['test_acc'] = test_loss[1]

    # Zapis wyników do pliku JSON z timestampem
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    results_file = f'{results_dir}/mjsynth_comparison_results_{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Wyniki zapisane w pliku: {results_file}")

    # Wyświetlenie końcowego porównania modeli
    print(f"\n{'=' * 50}")
    print("KOŃCOWE PORÓWNANIE")
    print(f"{'=' * 50}")

    for model_type, metrics in results.items():
        print(f"\n{model_type.upper()}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")


if __name__ == "__main__":
    main()