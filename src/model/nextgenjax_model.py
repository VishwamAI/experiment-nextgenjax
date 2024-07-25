import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Conv2D, MaxPooling2D, Flatten, Concatenate
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

class NextGenJaxModel(Model):
    def __init__(self, vocab_size, max_sequence_length=100, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.tokenizer = Tokenizer(num_words=self.vocab_size, filters='')
        self.tokenizer.fit_on_texts([' '.join([str(i) for i in range(vocab_size)])])

        # Input layers
        text_input = Input(shape=(self.max_sequence_length, self.vocab_size))
        voice_input = Input(shape=(128, 128, 1))  # Specify a fixed shape for voice input

        # Text processing branch
        x_text = LSTM(units=64, return_sequences=True)(text_input)
        x_text = LSTM(units=64)(x_text)
        x_text = Flatten()(x_text)  # Add Flatten layer

        # Voice processing branch
        x_voice = Conv2D(32, kernel_size=(3, 3), activation='relu')(voice_input)
        x_voice = MaxPooling2D(pool_size=(2, 2))(x_voice)
        x_voice = Flatten()(x_voice)

        # Combine branches
        combined = Concatenate()([x_text, x_voice])
        combined = Flatten()(combined)  # Ensure fully-defined shape

        # Common output layer
        output = Dense(units=self.vocab_size, activation='softmax', input_shape=(None,))(combined)

        self.model = Model(inputs=[text_input, voice_input], outputs=output)

    def call(self, inputs):
        return self.model(inputs)

    def text_to_text(self, input_text):
        sequences = self.tokenizer.texts_to_sequences([input_text])
        padded_sequences = pad_sequences(sequences, maxlen=self.max_sequence_length, padding='post', truncating='post')
        one_hot_sequences = to_categorical(padded_sequences, num_classes=self.vocab_size)
        # Create a dummy voice input (all zeros) since the model expects both inputs
        dummy_voice_input = tf.zeros((1, 128, 128, 1))  # Updated shape
        predictions = self.model.predict([one_hot_sequences, dummy_voice_input])
        output_indices = predictions.argmax(axis=-1)
        output_text = self.tokenizer.sequences_to_texts(output_indices)[0]
        return output_text

    def voice_to_text(self, voice_input):
        # Ensure voice_input has the correct shape (1, 128, 128, 1)
        voice_input = tf.ensure_shape(voice_input, (1, 128, 128, 1))
        dummy_text_input = tf.zeros((1, self.max_sequence_length, self.vocab_size))
        predictions = self.model.predict([dummy_text_input, voice_input])
        output_indices = predictions.argmax(axis=-1)
        output_text = self.tokenizer.sequences_to_texts(output_indices)[0]
        return output_text

# Instantiate the model with a defined vocabulary size
vocab_size = 10000  # This should be set based on the actual vocabulary size
nextgenjax_model = NextGenJaxModel(vocab_size=vocab_size)

# Save the updated model architecture
nextgenjax_model.model.save('nextgenjax_model.keras')
