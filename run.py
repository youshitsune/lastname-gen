import torch
import torch.nn.functional as F

W = torch.load("weights.pt")

user_input = int(input("How much last names do you want to generate: "))

for i in range(user_input):
    out = []
    ix = 0
    while True:
        x_encode = F.hot_one(torch.tensor([ix]), num_classes=27).float()
        logits = x_encode @ W
        cnt = logits.exp()
        prbs = cnt / cnt.sum(1, keepdims=True)
        ix = torch.multinomial(p, num_samples=1, replacement=True).item()
        if ix == 0:
            break
        else:
            out.append(itos[ix])
    print("".join(out))

