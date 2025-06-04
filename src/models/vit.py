import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

class ViTOCRModel:
    def __init__(self, vocab_size, img_height=32, img_width=128, patch_size=8, 
                 embed_dim=256, num_heads=8, num_layers=6):
        """
        Inicjalizacja modelu Vision Transformer (ViT) dla OCR.

        Argumenty:
            vocab_size (int): Rozmiar słownika dla rozpoznawania tekstu.
            img_height (int): Wysokość obrazu wejściowego.
            img_width (int): Szerokość obrazu wejściowego.
            patch_size (int): Rozmiar każdego wycinka obrazu.
            embed_dim (int): Wymiarowość osadzeń wycinków.
            num_heads (int): Liczba głów uwagi w transformatorze.
            num_layers (int): Liczba warstw enkodera transformatora.
        """
        self.vocab_size = vocab_size
        self.img_height = img_height
        self.img_width = img_width
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # Oblicz całkowitą liczbę wycinków w obrazie
        self.num_patches = (img_height // patch_size) * (img_width // patch_size)
        
    def build_model(self):
        """
        Budowa modelu Vision Transformer (ViT) dla OCR.

        Zwraca:
            model (tf.keras.Model): Zbudowany model ViT.
        """
        # Zdefiniuj warstwę wejściową dla obrazów w skali szarości
        inputs = layers.Input(shape=(self.img_height, self.img_width, 1))
        
        # Wyodrębnij wycinki z obrazu wejściowego
        patches = self.create_patches(inputs)
        
        # Osadź wycinki w przestrzeni o wyższej wymiarowości
        patch_embeddings = layers.Dense(self.embed_dim)(patches)
        
        # Dodaj osadzenia pozycyjne, aby zachować informacje przestrzenne
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        position_embeddings = layers.Embedding(
            input_dim=self.num_patches, 
            output_dim=self.embed_dim
        )(positions)
        
        # Połącz osadzenia wycinków z osadzeniami pozycyjnymi
        encoded_patches = patch_embeddings + position_embeddings
        
        # Przepuść zakodowane wycinki przez wiele warstw enkodera transformatora
        for _ in range(self.num_layers):
            encoded_patches = self.transformer_block(encoded_patches)
        
        # Znormalizuj wyjście transformatora
        x = layers.LayerNormalization()(encoded_patches)
        
        # Zastosuj globalne uśrednianie, aby zmniejszyć wymiarowość
        x = layers.GlobalAveragePooling1D()(x)
        
        # Dodaj warstwę gęstą z aktywacją ReLU do ekstrakcji cech
        x = layers.Dense(512, activation='relu')(x)
        
        # Dodaj dropout dla regularizacji
        x = layers.Dropout(0.1)(x)
        
        # Zdefiniuj długość sekwencji dla rozpoznawania tekstu
        sequence_length = self.img_width // 4  # Dostosuj w zależności od potrzeb
        
        # Mapuj cechy na rozmiar słownika dla każdego kroku sekwencji
        x = layers.Dense(sequence_length * self.vocab_size)(x)
        
        # Przekształć wyjście, aby dopasować długość sekwencji i rozmiar słownika
        outputs = layers.Reshape((sequence_length, self.vocab_size))(x)
        
        # Zastosuj aktywację softmax, aby uzyskać prawdopodobieństwa dla każdej litery
        outputs = layers.Activation('softmax')(outputs)
        
        # Utwórz ostateczny model
        model = Model(inputs, outputs, name='ViT_OCR')
        return model
    
    def create_patches(self, images):
        """
        Wyodrębnij wycinki z obrazów wejściowych.

        Argumenty:
            images (tf.Tensor): Batch obrazów wejściowych.

        Zwraca:
            patches (tf.Tensor): Wyodrębnione wycinki przekształcone w tensor 2D.
        """
        # Użyj funkcji extract_patches TensorFlow, aby podzielić obraz na wycinki
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],  # Rozmiar wycinka
            strides=[1, self.patch_size, self.patch_size, 1],  # Rozmiar kroku
            rates=[1, 1, 1, 1],  # Współczynnik dylatacji
            padding="VALID",  # Bez wypełnienia
        )
        
        # Przekształć wycinki w tensor 2D do dalszego przetwarzania
        batch_size = tf.shape(patches)[0]
        patches = tf.reshape(patches, [batch_size, self.num_patches, -1])
        return patches
    
    def transformer_block(self, encoded_patches):
        """
        Zdefiniuj pojedynczy blok enkodera transformatora.

        Argumenty:
            encoded_patches (tf.Tensor): Tensor wejściowy zakodowanych wycinków.

        Zwraca:
            outputs (tf.Tensor): Tensor wyjściowy po zastosowaniu bloku transformatora.
        """
        # Zastosuj wielogłową uwagę własną do zakodowanych wycinków
        attention = layers.MultiHeadAttention(
            num_heads=self.num_heads,  # Liczba głów uwagi
            key_dim=self.embed_dim  # Wymiarowość mechanizmu uwagi
        )(encoded_patches, encoded_patches)
        
        # Dodaj połączenie rezydualne i znormalizuj wyjście
        x1 = layers.Add()([attention, encoded_patches])
        x1 = layers.LayerNormalization()(x1)
        
        # Zastosuj sieć feed-forward z dwiema warstwami gęstymi
        x2 = layers.Dense(self.embed_dim * 2, activation='relu')(x1)  # Rozszerz wymiarowość
        x2 = layers.Dense(self.embed_dim)(x2)  # Projekcja z powrotem do oryginalnej wymiarowości
        
        # Dodaj kolejne połączenie rezydualne i znormalizuj wyjście
        outputs = layers.Add()([x2, x1])
        outputs = layers.LayerNormalization()(outputs)
        
        return outputs