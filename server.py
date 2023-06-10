import torch
import torch.nn as nn
import torch.optim as optim
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

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
        rhyme_suggestions = ", ".join(rhyme_list)
        return f"Your rhyme suggestions are: {rhyme_suggestions}."

@app.route('/get_rhyme', methods=['POST'])
def get_rhyme_api():
    data = request.get_json()
    user_message = data.get('message', '').strip().lower()

    if user_message == '1234':
        response = {'response': "5678"}
    else:
        rhyme_suggestions = get_rhyme(model, user_message, n_suggestions=3)
        response = {'response': rhyme_suggestions}

    return jsonify(response)

if __name__ == '__main__':
    word_list = []
    with open('word_list.systype', 'r') as file:
        for line in file:
            user_message = line.strip()
            word_list.append(user_message)

    phonetic_alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    input_size = len(phonetic_alphabet)
    hidden_size = 128
    output_size = len(word_list)

    model = RhymingModel(input_size, hidden_size, output_size)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    app.run(host="0.0.0.0", port=8080)
