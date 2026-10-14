import torch
import torch.nn as nn
from torch.nn import functional as F

block_size = 128
n_embd = 256
n_head = 4
n_layer = 6
dropout = 0.2

weights_file = 'best_model.pt'
max_new_tokens = 500
temperature = 0.8
top_k = 40

device = 'cuda' if torch.cuda.is_available() else 'cpu'

with open('HonkaiStarRail.txt', 'r', encoding='cp1252') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars)}
itos = { i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])


class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        return self.lm_head(x)


model = BigramLanguageModel(vocab_size).to(device)
model.load_state_dict(torch.load(weights_file, map_location=device))
model.eval()
print("loaded", weights_file, "on", device)


def curlify(s):
    out = []
    prev = ' '
    for c in s:
        if c == '"':
            c = '\u201c' if prev.isspace() or prev in '(\u2014-' else '\u201d'
        elif c == "'":
            c = '\u2019'
        out.append(c)
        prev = c
    return ''.join(out)


@torch.no_grad()
def write(story, n_new):
    ids = [stoi[c] for c in story if c in stoi]
    if not ids:
        ids = [stoi.get('\n', 0)]
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    new_text = ''
    for _ in range(n_new):
        logits = model(idx[:, -block_size:])[:, -1, :] / temperature
        if top_k:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')
        probs = F.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, nxt), dim=1)
        ch = itos[nxt.item()]
        new_text += ch
        print(ch, end='', flush=True)
    print()
    return new_text


print("Type some text and the model continues it. Your text and its output build up as one story.")
print("Commands: /len N   /temp X   /topk N   /reset   /quit")
print("Empty line = just keep writing.\n")

story = ''
while True:
    try:
        line = input('> ')
    except (EOFError, KeyboardInterrupt):
        print()
        break

    if line.startswith('/'):
        parts = line.split()
        cmd = parts[0]
        try:
            if cmd == '/quit':
                break
            elif cmd == '/reset':
                story = ''
                print("story cleared")
            elif cmd == '/len':
                max_new_tokens = int(parts[1])
                print("length:", max_new_tokens)
            elif cmd == '/temp':
                temperature = float(parts[1])
                print("temperature:", temperature)
            elif cmd == '/topk':
                top_k = int(parts[1])
                print("top_k:", top_k)
            else:
                print("unknown command")
        except (IndexError, ValueError):
            print("bad value")
        continue

    if line:
        line = curlify(line)
        dropped = sorted(set(c for c in line if c not in stoi))
        if dropped:
            print("skipped characters the model doesn't know:", dropped)
            line = ''.join(c for c in line if c in stoi)
        if story and not story[-1].isspace():
            story += ' '
        story += line

    story += write(story, max_new_tokens)