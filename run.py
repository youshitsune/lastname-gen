import streamlit as st
import torch
import torch.nn.functional as F

def run(n):
    W = torch.load("weights.pt")

    with open("last-names.txt", "r") as f:
        words = list(f.read().splitlines())

    chars = sorted(list(set(''.join(words))))
    stoi = {s:i+1 for i, s in enumerate(chars)}
    stoi['.'] = 0
    itos = {i:s for s, i in stoi.items()}
    
    gen = []
    for i in range(n):
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
        gen.append("".join(out))

    return gen

st.write("# Last name generator")
st.write("Source code: https://github.com/youshitsune/lastname-gen")
n = st.number_input('How much last names do you want to generate', min_value=1, max_value=10, value=1, step=1)
if n and st.button("Generate"):
    out = run(n)
    st.write("Generated last names are:")
    for i in out:
        st.write(f"{i}")


