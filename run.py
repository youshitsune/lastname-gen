import torch
import torch.nn.functional as F

W = torch.load("weights.pt")

with open("last-names.txt", "r") as f:
    words = list(f.read().splitlines())

chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s, i in stoi.items()}

user_input = int(input("How much last names do you want to generate: "))

for i in range(user_input):
    out = []
    ix = 0
    while True:
        x_encode = F.one_hot(torch.tensor([ix]), num_classes=27).float()
        logits = x_encode @ W
        cnt = logits.exp()
        prbs = cnt / cnt.sum(1, keepdims=True)
        ix = torch.multinomial(cnt, num_samples=1, replacement=True).item()
        if ix == 0:
            break
        else:
            out.append(itos[ix])
    print("".join(out))

