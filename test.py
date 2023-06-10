import torch
import torch.nn as nn
import torch.optim as optim

class RhymingModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RhymingModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        _, (hidden, _) = self.lstm(input)
        output = self.linear(hidden[-1])
        return output

def word_to_tensor(word):
    tensor = torch.zeros(len(word), 1, len(phonetic_alphabet))
    for i, char in enumerate(word):
        phonetic_index = phonetic_alphabet.index(char)
        tensor[i][0][phonetic_index] = 1
    return tensor

def get_rhyme(model, word, n_suggestions=5):
    model.eval()
    with torch.no_grad():
        input_tensor = word_to_tensor(word)
        output = model(input_tensor)
        _, top_indices = output.topk(n_suggestions)
        rhyme_list = []
        for index in top_indices.squeeze():
            rhyme_word = word_list[index.item()]
            rhyme_list.append(rhyme_word)
        return rhyme_list

word_list = []

with open('word_list.systype', 'r') as file:
    for line in file:
        word = line.strip()
        word_list.append(word)
phonetic_alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
input_size = len(phonetic_alphabet)
hidden_size = 128
output_size = len(word_list)

model = RhymingModel(input_size, hidden_size, output_size)

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    for word_index in range(len(word_list)):
        word = word_list[word_index]
        target_index = (word_index + 1) % len(word_list)

        input_tensor = word_to_tensor(word)
        target_tensor = torch.tensor([target_index])

        output = model(input_tensor)
        loss = loss_function(output, target_tensor)
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch: {epoch+1}, Loss: {loss.item()}')

test_word = 'jesus'
rhyme_suggestions = get_rhyme(model, test_word, n_suggestions=3)
print(f'Rhyme suggestions for "{test_word}": {rhyme_suggestions}')
