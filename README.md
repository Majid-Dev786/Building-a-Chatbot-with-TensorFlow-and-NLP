# Building a Chatbot with TensorFlow and NLP

## Description
This project is dedicated to building an intelligent chatbot using TensorFlow and Natural Language Processing (NLP). 

The chatbot is designed to understand and respond to user inputs positively or negatively, showcasing the power of machine learning in processing and interpreting human language.

## Table of Contents 
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)

## Installation
To get this project up and running on your machine, follow these steps:

1. Clone the repository:
```bash
git clone https://github.com/Sorena-Dev/Building-a-Chatbot-with-TensorFlow-and-NLP.git
```

2. Navigate to the project directory:
```bash
cd Building-a-Chatbot-with-TensorFlow-and-NLP
```

3. Install the required dependencies:
```bash
pip install numpy tensorflow
```

## Usage
This chatbot can be integrated into various applications to enhance user interaction. For instance, it can be used in customer service to automatically respond to common queries or integrated into an interactive website for engaging visitors.

To run the chatbot, execute the `Building a Chatbot with TensorFlow and NLP.py` script. Here's a simple example of how to use it in a real-world scenario:

```python
# Example of using the chatbot
from Building_a_Chatbot_with_TensorFlow_and_NLP import chatbot_response

response = chatbot_response("How are you?")
print(response)
```

## Features
- **Natural Language Processing:** Utilizes NLP techniques for understanding human languages.
- **TensorFlow Implementation:** Built with TensorFlow, enabling the model to learn and predict with high accuracy.
- **Versatile Applications:** Can be adapted for various use cases, including customer support, interactive websites, and more.

## Explanation :

Importing Libraries
Firstly, the necessary libraries are imported. numpy is used for handling arrays, and various components from tensorflow.keras are imported for building and training the neural network model.

Data Preparation
Input Data: The script defines a list of text samples (texts) and their corresponding labels (labels), which indicate the sentiment of each text sample. The goal is to classify these samples into positive or negative sentiments.

Tokenization: Using the Tokenizer from Keras, the text samples are tokenized, converting each text into a sequence of integers, where each integer represents a unique word in the dataset. The vocabulary size is determined by counting the unique words in all text samples.

Sequence Padding: Since neural networks require inputs of the same size, the sequences are padded to ensure they all have equal length, defined by the longest sequence in the dataset.

Label Encoding: The textual labels are converted into numerical form—more specifically, into one-hot encoded vectors. This process involves mapping each label to an integer (label_mapping) and then converting these integers into a binary class matrix (one_hot_labels).

Model Building
The model is constructed using the Sequential API from Keras. It includes:
An Embedding layer, which maps the vocabulary indices into a dense vector of fixed size (embedding_dim). This layer is essential for processing text in neural networks.
An LSTM layer, a type of recurrent neural network (RNN) layer that is good at capturing temporal dependencies and sequences in data, making it suitable for text.
A Dense layer with a softmax activation function that outputs the probabilities of each class (positive or negative sentiment in this case).

Compilation and Training
The model is compiled with the categorical_crossentropy loss function, suitable for multi-class classification problems, and the adam optimizer. The accuracy metric is used to evaluate the model's performance.
The model is trained using the prepared and padded sequences as input and the one-hot encoded labels as targets. The training process is set to run for 100 epochs.

Making Predictions
The script prepares new text samples (test_texts) for prediction by tokenizing and padding them in the same way as the training data.

The model predicts the sentiment of these new samples, and the script prints the predicted labels based on the highest probability output from the model.

This script is a basic example of how to use TensorFlow and Keras for a text classification task, showcasing the steps from data preparation to model prediction. It illustrates the use of RNNs (LSTM) for handling sequence data and provides a foundation for more complex natural language processing tasks.


.

