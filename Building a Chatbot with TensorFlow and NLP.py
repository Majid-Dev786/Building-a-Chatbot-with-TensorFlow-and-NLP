# Import necessary libraries and modules for building our model
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# Here, I define my dataset. It consists of texts and their corresponding labels.
texts = [
    "I am happy",
    "I am sad",
    "You are awesome",
    "You are terrible",
    "They are amazing"
]
labels = ["positive", "negative", "positive", "negative", "positive"]

# I use the Tokenizer to convert text into sequences of integers where each integer 
# represents a unique word.
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)  # This builds the word index
vocab_size = len(tokenizer.word_index) + 1  # I calculate the size of the vocabulary for the Embedding layer

# Convert texts to sequences of integers
sequences = tokenizer.texts_to_sequences(texts)

# Padding sequences to ensure they all have the same length for the model input
max_sequence_length = max(len(seq) for seq in sequences)  # Finding the longest sequence
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)  # Padding

# Mapping labels to integers and then to one-hot encoding for the model output
label_mapping = {"positive": 0, "negative": 1}
encoded_labels = [label_mapping[label] for label in labels]
one_hot_labels = to_categorical(encoded_labels)  # One-hot encode the labels

# Defining the model architecture
embedding_dim = 16
model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length=max_sequence_length))  # Embedding layer for text input
model.add(LSTM(16))  # LSTM layer for sequence processing
model.add(Dense(2, activation='softmax'))  # Dense layer for classification

# Compiling the model with loss, optimizer, and metrics to track
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Training the model on the dataset
model.fit(padded_sequences, one_hot_labels, epochs=100)  # Training the model with padded sequences and one-hot encoded labels

# Preparing test data to evaluate the model
test_texts = [
    "I am happy",
    "You are terrible"
]
test_sequences = tokenizer.texts_to_sequences(test_texts)  # Convert test texts to sequences
padded_test_sequences = pad_sequences(test_sequences, maxlen=max_sequence_length)  # Padding the test sequences

# Making predictions with the model
predictions = model.predict(padded_test_sequences)  # Predicting labels for the test data
predicted_labels = [labels[np.argmax(pred)] for pred in predictions]  # Converting predictions to labels
print(predicted_labels)  # Printing the predicted labels



