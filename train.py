import torch
import torch.nn.functional as F

with open("last-names.txt", "r") as f:
    words = list(f.read().splitlines())

chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}

xs, ys = [], []
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        xs.append(ix1)
        ys.append(ix2)

xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = xs.nelement()
W = torch.randn((27, 27), requires_grad = True)

for k in range(500):
    x_encode = F.one_hot(xs, num_classes=27).float()
    logits = x_encode @ W
    cnt = logits.exp()
    prbs = cnt / cnt.sum(1, keepdims=True)
    loss = -prbs[torch.arrange(num), ys].log().mean()

    W.grad = None
    loss.backward()

    W.data += -10 * W.grad

torch.save(W, "weights.pt")
