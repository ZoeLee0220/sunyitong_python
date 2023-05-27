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

# 生成5句诗句
num_poems = 14


chars = sorted(list(set(corpus)))
char_to_num = {char: num for num, char in enumerate(chars)}
num_to_char = {num: char for num, char in enumerate(chars)}
vocab_size = len(chars)

# Build the LSTM model
model = Sequential()
model.add(Embedding(vocab_size, 100, input_length=len(corpus)))
model.add(LSTM(256, return_sequences=True))
model.add(LSTM(256))
model.add(Dense(vocab_size, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

for _ in range(num_poems):
    # 随机选择一个输入句子
    input_sentence = np.random.choice(sentences)
    generated_text = input_sentence

    while len(generated_text) <4:  # 生成的诗句至少10个字符
        # 将输入句子转换为模型可接受的输入格式
        input_seq = [char_to_num[char] for char in generated_text]
        x = np.reshape(input_seq, (1, len(input_seq), 1))
        x = x / float(vocab_size)

        # 使用模型生成下一个字符的概率分布
        predictions = model.predict(x, verbose=0)
        index = np.random.choice(range(vocab_size), p=predictions.flatten())
        result = num_to_char[index]
        generated_text += result

    print(generated_text)
