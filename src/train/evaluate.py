import tensorflow as tf
import numpy as np
import editdistance
from mjs import MJSynthDataLoader

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
            for i in range(min(len(true_text), len(pred_text))):
                if true_text[i] == pred_text[i]:
                    correct_chars += 1
            total_chars += max(len(true_text), len(pred_text))
        
        return correct_chars / total_chars if total_chars > 0 else 0
    
    def calculate_word_accuracy(self, y_true, y_pred):
        """
        Obliczanie dokładności na poziomie słów.
        Porównuje całe słowa w prawdziwym i przewidywanym tekście.
        """
        correct_words = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
        return correct_words / len(y_true) if len(y_true) > 0 else 0
    
    def calculate_edit_distance(self, y_true, y_pred):
        """
        Obliczanie średniej odległości edycyjnej.
        Odległość edycyjna mierzy, ile zmian jest potrzebnych, aby przekształcić jeden ciąg w drugi.
        """
        distances = [editdistance.eval(true, pred) for true, pred in zip(y_true, y_pred)]
        return np.mean(distances)
    
    def evaluate_model(self, model, test_dataset, model_type='crnn', max_batches=None):
        """
        Przeprowadzenie kompleksowej oceny modelu.
        Oblicza dokładność na poziomie znaków, słów oraz odległość edycyjną.
        """
        y_true = []  # Lista do przechowywania prawdziwych etykiet tekstowych
        y_pred = []  # Lista do przechowywania przewidywanych etykiet tekstowych
        
        batch_count = 0
        for batch_images, batch_labels in test_dataset:
            if max_batches and batch_count >= max_batches:
                break
                
            predictions = model.predict(batch_images, verbose=0)
            
            if model_type == 'crnn':
                pred_texts = self.data_loader.decode_predictions(predictions)
        
            
            # Dekodowanie prawdziwych etykiet na tekst
            for label_sequence in batch_labels:
                # Zamiana indeksów na znaki, ignorowanie paddingu
                true_text = ''.join([
                    self.data_loader.idx_to_char.get(int(idx), '') 
                    for idx in label_sequence.numpy()
                    if int(idx) in self.data_loader.idx_to_char and int(idx) != self.data_loader.pad_token_idx
                ])
                y_true.append(true_text)
            
            # Dodanie przewidywanych tekstów do listy
            y_pred.extend(pred_texts)
            batch_count += 1
            
            # Wyświetlanie postępu co 10 partii
            if batch_count % 10 == 0:
                print(f"Przetworzono {batch_count} partii...")
        
        results = {
            'character_accuracy': self.calculate_character_accuracy(y_true, y_pred),
            'word_accuracy': self.calculate_word_accuracy(y_true, y_pred),
            'edit_distance': self.calculate_edit_distance(y_true, y_pred)
        }
        
        print(f"Przykładowe przewidywania dla {model_type.upper()}:")
        for batch_images, batch_labels in test_dataset.take(1):
            predictions = model.predict(batch_images, verbose=0)
            
            if model_type == 'crnn':
                decoded_predictions = self.data_loader.decode_predictions(predictions)
                
                # Dekodowanie prawdziwych etykiet
                true_labels = []
                for label_sequence in batch_labels:
                    true_text = ''.join([
                        self.data_loader.idx_to_char.get(int(idx), '') 
                        for idx in label_sequence.numpy() 
                        if int(idx) in self.data_loader.idx_to_char and int(idx) != self.data_loader.pad_token_idx
                    ])
                    true_labels.append(true_text)
                
                # Wyświetlenie kilku przykładów
                for i in range(min(3, len(decoded_predictions))):
                    print(f"Prawdziwe: '{true_labels[i]}' | Przewidywane: '{decoded_predictions[i]}'")
            else:
                print("Surowe predykcje (pierwsza próbka):", predictions[0])
        
        return results, y_true, y_pred

# Funkcja do oceny zapisanych modeli
def evaluate_saved_models():
    """
    Ładowanie i ocena zapisanych modeli OCR (CRNN i ViT).
    Oblicza dokładność na poziomie znaków, słów oraz odległość edycyjną.
    """
    data_loader = MJSynthDataLoader('/home/jagoda/studia/inteloblicz/OCRAppModels/mjsynth')
    
    test_samples = data_loader.load_data('test', limit=2)
    test_dataset = data_loader.create_crnn_tf_dataset(
        test_samples, batch_size=2
    )
    
    evaluator = MJSynthEvaluator(data_loader)
    
    for model_type in ['crnn']:
        model_path = f'../results/{model_type}_best.keras'
        try:
            print(f"\nOcena modelu {model_type.upper()}...")
            model = tf.keras.models.load_model(model_path, compile=False)
            
            results, y_true, y_pred = evaluator.evaluate_model(
                model, test_dataset, model_type=model_type, max_batches=20
            )
            
            print(f"\nWyniki {model_type.upper()}:")
            for metric, value in results.items():
                print(f"  {metric}: {value:.4f}")
                
        except Exception as e:
            print(f"Błąd podczas oceny {model_type}: {e}")

if __name__ == "__main__":
    evaluate_saved_models()