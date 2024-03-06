import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# Define the input data
texts = [
    "I am happy",
    "I am sad",
    "You are awesome",
    "You are terrible",
    "They are amazing"
]
labels = ["positive", "negative", "positive", "negative", "positive"]

# Tokenize the input texts
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
vocab_size = len(tokenizer.word_index) + 1

# Convert texts to sequences
sequences = tokenizer.texts_to_sequences(texts)

# Pad sequences for equal length
max_sequence_length = max(len(seq) for seq in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)

# Convert labels to one-hot vectors
label_mapping = {"positive": 0, "negative": 1}
encoded_labels = [label_mapping[label] for label in labels]
one_hot_labels = to_categorical(encoded_labels)

# Create the model
embedding_dim = 16
model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length=max_sequence_length))
model.add(LSTM(16))
model.add(Dense(2, activation='softmax'))

# Compile and train the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(padded_sequences, one_hot_labels, epochs=100)

# Generate predictions
test_texts = [
    "I am happy",
    "You are terrible"
]
test_sequences = tokenizer.texts_to_sequences(test_texts)
padded_test_sequences = pad_sequences(test_sequences, maxlen=max_sequence_length)
predictions = model.predict(padded_test_sequences)
predicted_labels = [labels[np.argmax(pred)] for pred in predictions]
print(predicted_labels)
