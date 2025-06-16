import tensorflow as tf
import numpy as np
import editdistance
from mjs import MJSynthDataLoader

# Klasa do oceny modeli OCR przy użyciu zbioru danych MJSynth
class MJSynthEvaluator:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        
    def calculate_character_accuracy(self, y_true, y_pred):
        """
        Obliczanie dokładności na poziomie znaków.
        Porównuje każdy znak w prawdziwym i przewidywanym tekście.
        """
        correct_chars = 0
        total_chars = 0
        
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
                pred_indices = tf.argmax(predictions, axis=-1)
                pred_texts = []
                for sequence in pred_indices:
                    text = ''.join([self.data_loader.idx_to_char[int(idx)] 
                                    for idx in sequence if 0 <= int(idx) < self.data_loader.vocab_size - 1])
                    pred_texts.append(text)
            
            # Dekodowanie prawdziwych etykiet na tekst
            for label_sequence in batch_labels:
                true_text = ''.join([
                    self.data_loader.idx_to_char.get(int(idx), '') 
                    for idx in label_sequence.numpy()
                    if int(idx) in self.data_loader.idx_to_char and int(idx) != self.data_loader.pad_token_idx #dodalam and mozliwe ze jest zle
                ])
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
        
        # Fix the prediction display section:
        print(f"Przykładowe przewidywania dla {model_type.upper()}:")
        for batch_images, batch_labels in test_dataset.take(1):
            predictions = model.predict(batch_images, verbose=0)  # Use 'model' not 'trainer'
            
            if model_type == 'crnn':
                # Decode CTC predictions properly
                decoded_predictions = self.data_loader.decode_predictions(predictions)
                
                # Get true labels for comparison
                true_labels = []
                for label_sequence in batch_labels:
                    # Convert indices back to text
                    true_text = ''.join([
                        self.data_loader.idx_to_char.get(int(idx), '') 
                        for idx in label_sequence.numpy() 
                        if int(idx) in self.data_loader.idx_to_char
                    ])
                    true_labels.append(true_text)
                
                # Display first few examples
                for i in range(min(3, len(decoded_predictions))):
                    print(f"Prawdziwe: '{true_labels[i]}' | Przewidywane: '{decoded_predictions[i]}'")
            else:
                print("Surowe predykcje (pierwsza próbka):", predictions[0])
        
        # Return the required values
        return results, y_true, y_pred

# Funkcja do oceny zapisanych modeli
def evaluate_saved_models():
    """
    Ładowanie i ocena zapisanych modeli OCR (CRNN i ViT).
    Oblicza dokładność na poziomie znaków, słów oraz odległość edycyjną.
    """
    # Ładowanie ładowarki danych MJSynth
    data_loader = MJSynthDataLoader('/home/jagoda/studia/inteloblicz/OCRAppModels/mjsynth')
    
    # Ładowanie podziału zbioru danych testowych
    test_samples = data_loader.load_data('test', limit=2)
    test_dataset = data_loader.create_crnn_tf_dataset(
        test_samples, batch_size=2
    )
    
    # Tworzenie instancji klasy oceniającej
    evaluator = MJSynthEvaluator(data_loader)
    
    # Ocena zarówno modeli CRNN, jak i ViT
    for model_type in ['crnn']:
        model_path = f'../results/{model_type}_best.keras'
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