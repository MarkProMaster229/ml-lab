# пример: токенизированное предложение
sentences = [
    "кошка да собака мои верные друзья",
    "Штирлиц погладил кошку, кошка сдохла. Странно, — подумал Штирлиц, поплевав на утюг.",
    "Штирлица в 4 утра разбудил крик петуха. Он выглянул в окно и увидел - что петух навернулся с самоката."
]
vocab = {}
for sent in sentences:
    for word in sent.split():
        if word not in vocab:
            vocab[word] = len(vocab)
inv_vocab = {idx: word for word, idx in vocab.items()}

tokenized_example = [10, 11, 15, 16, 8, 0, 13, 7, 14, 12, 6, 9]  # допустим, это токены

# восстанавливаем слова
sentence_back = " ".join(inv_vocab[token] for token in tokenized_example)
print(sentence_back)
