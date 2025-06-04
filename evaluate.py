import tensorflow as tf
import numpy as np
import editdistance
from mjs import MJSynthDataLoader

# Klasa do oceny modeli OCR przy użyciu zbioru danych MJSynth
class MJSynthEvaluator:
    def __init__(self, data_loader):
        # Inicjalizacja z instancją ładowarki danych
        self.data_loader = data_loader
        
    def calculate_character_accuracy(self, y_true, y_pred):
        """
        Obliczanie dokładności na poziomie znaków.
        Porównuje każdy znak w prawdziwym i przewidywanym tekście.
        """
        correct_chars = 0  # Liczba poprawnie przewidzianych znaków
        total_chars = 0    # Całkowita liczba znaków w prawdziwym tekście
        
        for true_text, pred_text in zip(y_true, y_pred):
            # Porównanie znaków w prawdziwym i przewidywanym tekście
            for i in range(min(len(true_text), len(pred_text))):
                if true_text[i] == pred_text[i]:
                    correct_chars += 1
            # Użycie długości dłuższego tekstu jako całkowitej liczby znaków
            total_chars += max(len(true_text), len(pred_text))
        
        # Zwrócenie dokładności jako stosunku poprawnych do całkowitych znaków
        return correct_chars / total_chars if total_chars > 0 else 0
    
    def calculate_word_accuracy(self, y_true, y_pred):
        """
        Obliczanie dokładności na poziomie słów.
        Porównuje całe słowa w prawdziwym i przewidywanym tekście.
        """
        # Liczenie, ile słów jest dokładnie takich samych
        correct_words = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
        # Zwrócenie dokładności jako stosunku poprawnych słów do całkowitej liczby słów
        return correct_words / len(y_true) if len(y_true) > 0 else 0
    
    def calculate_edit_distance(self, y_true, y_pred):
        """
        Obliczanie średniej odległości edycyjnej.
        Odległość edycyjna mierzy, ile zmian jest potrzebnych, aby przekształcić jeden ciąg w drugi.
        """
        # Obliczanie odległości edycyjnej dla każdej pary prawdziwego i przewidywanego tekstu
        distances = [editdistance.eval(true, pred) for true, pred in zip(y_true, y_pred)]
        # Zwrócenie średniej odległości edycyjnej
        return np.mean(distances)
    
    def evaluate_model(self, model, test_dataset, model_type='crnn', max_batches=None):
        """
        Przeprowadzenie kompleksowej oceny modelu.
        Oblicza dokładność na poziomie znaków, słów oraz odległość edycyjną.
        """
        y_true = []  # Lista do przechowywania prawdziwych etykiet tekstowych
        y_pred = []  # Lista do przechowywania przewidywanych etykiet tekstowych
        
        batch_count = 0  # Licznik przetworzonych partii
        for batch_images, batch_labels in test_dataset:
            # Zatrzymanie przetwarzania, jeśli osiągnięto max_batches
            if max_batches and batch_count >= max_batches:
                break
                
            # Pobranie przewidywań z modelu
            predictions = model.predict(batch_images, verbose=0)
            
            if model_type == 'crnn':
                # Dekodowanie przewidywań dla modeli opartych na CTC
                pred_texts = self.data_loader.decode_predictions(predictions)
            else:
                # Dekodowanie przewidywań dla modeli nieopartych na CTC
                pred_indices = tf.argmax(predictions, axis=-1)  # Pobranie najbardziej prawdopodobnych indeksów znaków
                pred_texts = []
                for sequence in pred_indices:
                    # Konwersja indeksów na znaki i tworzenie przewidywanego tekstu
                    text = ''.join([self.data_loader.num_to_char[int(idx)] 
                                  for idx in sequence if idx > 0])
                    pred_texts.append(text)
            
            # Dekodowanie prawdziwych etykiet na tekst
            for label_sequence in batch_labels:
                true_text = ''.join([self.data_loader.num_to_char[int(idx)] 
                                   for idx in label_sequence if idx > 0])
                y_true.append(true_text)
            
            # Dodanie przewidywanych tekstów do listy
            y_pred.extend(pred_texts)
            batch_count += 1
            
            # Wyświetlanie postępu co 10 partii
            if batch_count % 10 == 0:
                print(f"Przetworzono {batch_count} partii...")
        
        # Obliczanie metryk oceny
        results = {
            'character_accuracy': self.calculate_character_accuracy(y_true, y_pred),
            'word_accuracy': self.calculate_word_accuracy(y_true, y_pred),
            'edit_distance': self.calculate_edit_distance(y_true, y_pred)
        }
        
        # Wyświetlanie przykładowych przewidywań do ręcznej inspekcji
        print("\nPrzykładowe przewidywania:")
        for i in range(min(10, len(y_true))):
            print(f"Prawdziwe: '{y_true[i]}' | Przewidywane: '{y_pred[i]}'")
        
        # Zwrócenie wyników oceny oraz prawdziwych/przewidywanych tekstów
        return results, y_true, y_pred

# Funkcja do oceny zapisanych modeli
def evaluate_saved_models():
    """
    Ładowanie i ocena zapisanych modeli OCR (CRNN i ViT).
    Oblicza dokładność na poziomie znaków, słów oraz odległość edycyjną.
    """
    # Ładowanie ładowarki danych MJSynth
    data_loader = MJSynthDataLoader('/home/jagoda/studia/inteloblicz/OCRApp/mjsynth')
    
    # Ładowanie podziału zbioru danych testowych
    test_images, test_labels = data_loader.load_dataset_split('test')
    # Tworzenie zbioru danych TensorFlow do oceny
    test_dataset = data_loader.create_tf_dataset(
        test_images[:1000], test_labels[:1000],  # Ograniczenie do 1000 próbek dla szybszej oceny
        batch_size=32, shuffle=False
    )
    
    # Tworzenie instancji klasy oceniającej
    evaluator = MJSynthEvaluator(data_loader)
    
    # Ocena zarówno modeli CRNN, jak i ViT
    for model_type in ['crnn', 'vit']:
        model_path = f'models/{model_type}_best.h5'  # Ścieżka do zapisanego modelu
        
        try:
            print(f"\nOcena modelu {model_type.upper()}...")
            # Ładowanie zapisanego modelu
            model = tf.keras.models.load_model(model_path, compile=False)
            
            # Ocena modelu i pobranie wyników
            results, y_true, y_pred = evaluator.evaluate_model(
                model, test_dataset, model_type=model_type, max_batches=20
            )
            
            # Wyświetlanie wyników oceny
            print(f"\nWyniki {model_type.upper()}:")
            for metric, value in results.items():
                print(f"  {metric}: {value:.4f}")
                
        except Exception as e:
            # Obsługa błędów podczas oceny modelu
            print(f"Błąd podczas oceny {model_type}: {e}")

# Punkt wejścia dla skryptu
if __name__ == "__main__":
    evaluate_saved_models()