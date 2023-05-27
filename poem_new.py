import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM, Dense, Embedding

# Load the poem dataset
with open('poems_1.txt', 'r', encoding='utf-8') as file:
    corpus = file.read()

# Preprocess the dataset
corpus = corpus.replace('\n', '')  # Remove line breaks
#print(corpus)
sentences = corpus.split(',')  # Split sentences by comma
#print(sentences)
# Create character-level mappings
chars = sorted(list(set(corpus)))
char_to_num = {char: num for num, char in enumerate(chars)}
num_to_char = {num: char for num, char in enumerate(chars)}
vocab_size = len(chars)

# Generate input-output pairs for training
input_seqs = []
output_seqs = []
seq_length = 40  # Length of input sequence
for i in range(0, len(corpus) - seq_length):
    input_seq = corpus[i:i + seq_length]
    output_seq = corpus[i + seq_length]
    input_seqs.append([char_to_num[char] for char in input_seq])
    output_seqs.append(char_to_num[output_seq])

# Convert input-output pairs to numpy arrays
X = np.reshape(input_seqs, (len(input_seqs), seq_length, 1))
X = X / float(vocab_size)  # Normalize input
y = tf.keras.utils.to_categorical(output_seqs, num_classes=vocab_size)  # Perform one-hot encoding

# Build the LSTM model
model = Sequential()
model.add(Embedding(vocab_size, 100, input_length=seq_length))
model.add(LSTM(256, return_sequences=True))
model.add(LSTM(256, return_sequences=True))
model.add(LSTM(256))
model.add(Dense(vocab_size, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Train the model
model.fit(X, y, batch_size=128, epochs=10)


# Generate text using the trained model
start_index = np.random.randint(0, len(input_seqs) - 1)
input_seq = input_seqs[start_index]
generated_text = ''.join([num_to_char[num] for num in input_seq])

# Generate at least 4 characters
num_outputs = 6  # 每次生成的字符数目
while len(generated_text) < num_outputs:
    x = np.reshape(input_seq, (1, len(input_seq), 1))
    x = x / float(vocab_size)
    predictions = model.predict(x, verbose=0)
    # 从预测结果中随机选择一个字符
    index = np.random.choice(range(vocab_size), p=predictions.flatten())
    result = num_to_char[index]
    generated_text += result
    input_seq.append(index)
    input_seq = input_seq[1:]

print(generated_text)


