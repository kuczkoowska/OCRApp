import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Layer
import numpy as np

class PatchExtract(Layer):
    def __init__(self, patch_size, num_patches, img_channels=1, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.img_channels = img_channels
        self.patch_dim = patch_size * patch_size * img_channels  # <-- liczba cech w patchu

    def call(self, images):
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        batch_size = tf.shape(patches)[0]
        # Użyj jawnie patch_dim
        patches = tf.reshape(patches, [batch_size, self.num_patches, self.patch_dim])
        return patches
    
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
        # Warstwa wejściowa dla obrazów w skali szarości
        inputs = layers.Input(shape=(self.img_height, self.img_width, 1))
        
        # wycinki z obrazu wejściowego
        patches = PatchExtract(self.patch_size, self.num_patches, img_channels=1)(inputs)
        
        patch_embeddings = layers.Dense(self.embed_dim)(patches)
        
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        position_embeddings = layers.Embedding(
            input_dim=self.num_patches, 
            output_dim=self.embed_dim
        )(positions)
        position_embeddings = tf.expand_dims(position_embeddings, axis=0) 
        encoded_patches = patch_embeddings + position_embeddings
                
        # zakodowane wycinki są przepuszczane przez wiele warstw transformatora
        for _ in range(self.num_layers):
            encoded_patches = self.transformer_block(encoded_patches)
        
        # normalizacja wyjścia transformatora
        x = layers.LayerNormalization()(encoded_patches)
        
        x = layers.GlobalAveragePooling1D()(x)
        
        # Warstwa z aktywacją ReLU do ekstrakcji cech
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.1)(x)
        
        # długość sekwencji dla rozpoznawania tekstu
        sequence_length = 12
        
        # Mapowanie cech na rozmiar słownika dla każdego kroku
        x = layers.Dense(sequence_length * self.vocab_size)(x)
        
        # Przekształcanie wyjścia
        outputs = layers.Reshape((sequence_length, self.vocab_size))(x)
        
        # uzyskiwanie prawdopodobieństwa dla każdej litery
        outputs = layers.Activation('softmax')(outputs)
        
        model = Model(inputs, outputs, name='ViT_OCR')
        return model
    
    def create_patches(self, images):
        # dzielenie obrazu na wycinki
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],  # Rozmiar wycinka
            strides=[1, self.patch_size, self.patch_size, 1],  # Rozmiar kroku
            rates=[1, 1, 1, 1],  # Współczynnik dylatacji
            padding="VALID",  # Bez wypełnienia
        )
        
        batch_size = tf.shape(patches)[0]
        patches = tf.reshape(patches, [batch_size, self.num_patches, -1])
        return patches
    
    def transformer_block(self, encoded_patches):
        attention = layers.MultiHeadAttention(
            num_heads=self.num_heads,  # Liczba głów uwagi
            key_dim=self.embed_dim  # Wymiarowość mechanizmu uwagi
        )(encoded_patches, encoded_patches)
        
        # normalizacja
        x1 = layers.Add()([attention, encoded_patches])
        x1 = layers.LayerNormalization()(x1)
        
        x2 = layers.Dense(self.embed_dim * 2, activation='relu')(x1) 
        x2 = layers.Dense(self.embed_dim)(x2)
        
        # kolejne połączenie rezydualne i normalizacja wyjścia
        outputs = layers.Add()([x2, x1])
        outputs = layers.LayerNormalization()(outputs)
        
        return outputs