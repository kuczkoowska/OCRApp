import os
import json
import tensorflow as tf
from pathlib import Path
from mjs import MJSynthDataLoader
from src.models.CRNN import CRNNModel, ctc_loss_func
from src.models.vit import ViTOCRModel
import matplotlib.pyplot as plt


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

    def train(self, train_dataset, val_dataset, train_samples, val_samples, batch_size, epochs=50, save_path='models/'):
        """
        Trenuje model na podanych zbiorach danych.
        Args:
            train_dataset: Zbiór danych treningowych.
            val_dataset: Zbiór danych walidacyjnych.
            epochs: Liczba epok treningu.
            save_path: Katalog do zapisu wytrenowanego modelu.
        """
        # Tworzenie katalogu do zapisu modeli
        os.makedirs(save_path, exist_ok=True)

        # Definiowanie callbacków do treningu
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,  # Zatrzymanie treningu, jeśli brak poprawy przez 10 epok
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,  # Zmniejszenie współczynnika uczenia o połowę
                patience=5,  # Po 5 epokach bez poprawy
                min_lr=1e-7  # Minimalny współczynnik uczenia
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=f'{save_path}/{self.model_type}_best.keras',
                monitor='val_loss',
                save_best_only=True,  # Zapis tylko najlepszego modelu
                save_weights_only=False
            )
        ]

        # Trenowanie modelu i zapis historii treningu
        self.history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            steps_per_epoch=train_samples // batch_size,
            validation_steps=val_samples // batch_size,
            callbacks=callbacks,
            verbose=1
        )

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
        'batch_size': 8,  # Rozmiar batcha dla treningu
        'epochs': 10,  # Liczba epok
        'img_height': 32,  # Wysokość obrazu do zmiany rozmiaru
        'img_width': 128,  # Szerokość obrazu do zmiany rozmiaru
        'data_amount_limit': 2000  # Maksymalna liczba próbek do załadowania
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

    print("Ładowanie danych walidacyjnych...")
    val_samples = data_loader.load_data('val', config['data_amount_limit'])

    print("Ładowanie danych testowych...")
    test_samples = data_loader.load_data('test', config['data_amount_limit'])

    # Ograniczenie rozmiaru zbioru danych, jeśli określono w konfiguracji
    if config['data_amount_limit']:
        train_samples = train_samples[:config['data_amount_limit']]
        val_samples = val_samples[:config['data_amount_limit'] // 5]
        test_samples = test_samples[:config['data_amount_limit'] // 10]

    # Lista modeli do trenowania
    models_to_train = ['crnn','vit']
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
            ).repeat()
        else:  # CRNN
            train_dataset = data_loader.create_crnn_tf_dataset(
                train_samples, batch_size=config['batch_size']
            ).repeat()
            val_dataset = data_loader.create_crnn_tf_dataset(
                val_samples, batch_size=config['batch_size']
            ).repeat()
            test_dataset = data_loader.create_crnn_tf_dataset(
                test_samples, batch_size=config['batch_size']
            ).repeat()



        # Inicjalizacja trenera dla bieżącego typu modelu
        trainer = OCRTrainer(data_loader, model_type=model_type)
        trainer.build_model()

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

        # Rysowanie historii treningu
        trainer.plot_training_history()

        # Ocena modelu na zbiorze testowym
        print(f"Ewaluacja modelu {model_type.upper()}...")
        test_steps = len(test_samples) // config['batch_size']
        test_loss = trainer.model.evaluate(test_dataset, steps=test_steps, verbose=1)
        print(f"Strata testowa: {test_loss}")

        # Przechowywanie wyników dla bieżącego modelu
        results[model_type] = {
            'final_train_loss': history.history['loss'][-1],
            'final_val_loss': history.history['val_loss'][-1],
            'test_loss': test_loss if isinstance(test_loss, float) else test_loss[0]
        }

        # Dodanie metryk dokładności, jeśli dostępne
        if 'accuracy' in history.history:
            results[model_type]['final_train_acc'] = history.history['accuracy'][-1]
            results[model_type]['final_val_acc'] = history.history['val_accuracy'][-1]
            if isinstance(test_loss, list) and len(test_loss) > 1:
                results[model_type]['test_acc'] = test_loss[1]

    # Zapis wyników do pliku JSON
    os.makedirs('results', exist_ok=True)
    with open('results/mjsynth_comparison_results.json', 'w') as f:
        json.dump(results, f, indent=2)

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