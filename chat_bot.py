import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
import numpy as np
import os


def get_response(user_input, model, tokenizer, max_length):
    tokenized_input = tokenizer.texts_to_sequences([user_input])  # Tokenize user input
    padded_input = tf.keras.preprocessing.sequence.pad_sequences(tokenized_input, maxlen=max_length,
                                                                 padding='post')  # Pad sequences
    predicted_index = tf.argmax(model.predict(padded_input), axis=-1).numpy()[0]  # Get index of predicted response

    if predicted_index < len(responses):
        return responses[predicted_index]
    else:
        return "I'm sorry, I didn't understand that."


# Sample responses
responses = [
    "Hello! How can I help you?",
    "I'm sorry, I didn't understand that.",
    "Could you please provide more details?",
    "That's interesting. Tell me more!",
    "I'm here to assist you.",
    "Let me think about that for a moment...",
    "Have you tried searching online for an answer?",
]

# File to save new responses
responses_file = "log/new_responses.txt"


# Function to create the neural network model
def create_model(input_shape, output_shape):
    model = Sequential([
        Embedding(input_shape, 64),  # Embedding layer for word embeddings
        GlobalAveragePooling1D(),  # Global Average Pooling layer for dimensionality reduction
        Dense(64, activation='relu'),  # Dense hidden layer with ReLU activation
        Dense(output_shape, activation='softmax')  # Output layer with softmax activation
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


# Function to train the neural network model
def train_model(model, inputs, outputs, epochs=10):
    model.fit(inputs, outputs, epochs=epochs, verbose=0)


# Function to generate response using the trained model
def get_response(user_input, model, tokenizer, max_length):
    tokenized_input = tokenizer.texts_to_sequences([user_input])  # Tokenize user input
    padded_input = tf.keras.preprocessing.sequence.pad_sequences(tokenized_input, maxlen=max_length,
                                                                 padding='post')  # Pad sequences
    predicted_index = tf.argmax(model.predict(padded_input), axis=-1).numpy()[0]  # Get index of predicted response
    return responses[predicted_index]  # Return the predicted response


# Function to save new responses to a file
def save_responses(new_responses):
    with open(responses_file, "a") as file:
        for response in new_responses:
            file.write(response + "\n")


def main():
    print("Welcome to AdvancedAI!")

    # Check if responses file exists, if not create one
    if not os.path.exists(responses_file):
        open(responses_file, 'w').close()

    # Load existing responses from file
    with open(responses_file, "r") as file:
        existing_responses = file.readlines()
    existing_responses = [response.strip() for response in existing_responses]

    # Combine existing responses with sample responses
    all_responses = responses + existing_responses

    tokenizer = tf.keras.preprocessing.text.Tokenizer()  # Initialize tokenizer
    tokenizer.fit_on_texts(all_responses)  # Fit tokenizer on all response data
    vocab_size = len(tokenizer.word_index) + 1  # Calculate vocabulary size

    model = create_model(input_shape=vocab_size, output_shape=vocab_size)  # Create neural network model

    # Prepare inputs and outputs for training
    max_length = max([len(tokenizer.texts_to_sequences([response])[0]) for response in all_responses])
    input_sequences = []
    target_sequences = []
    for response in all_responses:
        tokenized_response = tokenizer.texts_to_sequences([response])[0]
        for i in range(1, len(tokenized_response)):
            input_sequences.append(tokenized_response[:i])
            target_sequences.append(tokenized_response[i])
    inputs = tf.keras.preprocessing.sequence.pad_sequences(input_sequences, maxlen=max_length - 1, padding='post')
    outputs = np.array(target_sequences)

    train_model(model, inputs, outputs)  # Train the model

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break

        response = get_response(user_input, model, tokenizer, max_length)  # Get response from the model
        print("AdvancedAI:", response)

        # Ask user if the response was helpful and save it if it's a new response
        feedback = input("Was this response helpful? (yes/no): ")
        if feedback.lower() == 'no':
            new_response = input("Please provide the correct response: ")
            if new_response not in all_responses:
                save_responses([new_response])
                all_responses.append(new_response)


if __name__ == "__main__":
    main()

