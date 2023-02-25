import streamlit as st
import torch
import torch.nn.functional as F

def run(n):
    C = torch.load("c.pt")
    W1 = torch.load("w1.pt")
    b1 = torch.load("b1.pt")
    W2 = torch.load("w2.pt")
    b2 = torch.load("b2.pt")
    block_size = 3

    with open("last-names.txt", "r") as f:
        words = list(f.read().splitlines())

    chars = sorted(list(set(''.join(words))))
    stoi = {s:i+1 for i, s in enumerate(chars)}
    stoi['.'] = 0
    itos = {i:s for s, i in stoi.items()}
    
    gen = []
    for _ in range(n):
        out = []
        context = [0] * block_size
        while True:
            emb = C[torch.tensor([context])]
            h = torch.tanh(emb.view(1, -1) @ W1 + b1)
            logits = h @ W2 + b2
            probs = F.softmax(logits, dim=1)
            ix = torch.multinomial(probs, num_samples=1).item()
            context = context[1:] + [ix]
            if ix == 0:
                break
            else:
                out.append(ix)
        gen.append("".join(itos[i] for i in out))

    return gen

st.write("# Last name generator")
st.write("Source code: https://github.com/youshitsune/lastname-gen")
n = st.slider('How much last names do you want to generate', min_value=1, max_value=10, value=1, step=1)
if n and st.button("Generate"):
    out = run(n)
    st.write("Generated last names are:")
    for i in out:
        st.write(f"{i}")


