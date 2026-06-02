from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import tiktoken
import time

device = 'cpu'
if torch.cuda.is_available():
	device = 'cuda'
elif torch.mps.is_available():
	device = 'mps';

device = 'cpu'

print(f'Using device {device}')

class DataLoaderLite:
	def __init__(self, B, T):
		self.B = B
		self.T = T
		with open('tinyshakespeare.txt') as f:
			text = f.read()
		enc=tiktoken.get_encoding('gpt2')
		tokens = enc.encode(text)
		self.tokens = torch.tensor(tokens)
		print(f'Loaded {len(self.tokens)} tokens')
		print(f'1 epoch = {len(self.tokens) // (B*T)} batches')
		self.current_position = 0

	def next_batch(self):
		B, T = self.B, self.T
		buffer = self.tokens[self.current_position:self.current_position+B*T+1]
		x = buffer[:-1].view(B, T).to(device)
		y = buffer[1:].view(B, T).to(device)
		self.current_position += B*T
		if self.current_position + (B*T+1)>len(self.tokens):
			self.current_position=0
		return x, y

	
class MLP(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.c_fc = nn.Linear(config.n_embed, 4*config.n_embed)
		# no reason to use this these days, just use the exact version
		self.gelu = nn.GELU(approximate="tanh") # it'd be good to read the paper on why gelu is better than relu (or use swiglu)
		self.c_proj = nn.Linear(4*config.n_embed, config.n_embed)

	def forward(self, x):
		x = self.c_fc(x)
		x = self.gelu(x)
		x = self.c_proj(x)
		return x

class CausalSelfAttention(nn.Module):
    mask: torch.Tensor

    def __init__(self, config):
        super().__init__()
        assert config.n_embed % config.n_head == 0
        # key, query, and value projections for all in one head
        self.c_attn = nn.Linear(config.n_embed, 3*config.n_embed)
        # output projection
        self.c_proj = nn.Linear(config.n_embed, config.n_embed)
        self.n_head = config.n_head
        self.n_embed = config.n_embed
        # mask for making this a decoder (1, 1, context_size, context_size)
        self.register_buffer('mask', torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch, sequence length, n_embed

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embed, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, n_head, T, head_size)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, n_head, T, head_size)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, n_head, T, head_size)

        # attention: materializes the TxT matrix for each batch and n_heads
        # att = (q @ k.transpose(-2, -1)) * (1.0/math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        # att = F.softmax(att, dim=-1) # make all rows sum to 1

        # y = att @ v # (B, n_head, T, T) @ (B, n_head, T, heads) -> (B, n_head, T, heads)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y



@dataclass
class GPTConfig:
	block_size: int = 1024 # context length
	vocab_size: int = 50257
	n_layer: int = 12
	n_head: int = 12
	n_embed: int = 768

class Block(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.config=config
		self.ln_1 = nn.LayerNorm(config.n_embed)
		self.attn = CausalSelfAttention(config)
		self.ln_2 = nn.LayerNorm(config.n_embed)
		self.mlp = MLP(config)

	def forward(self, x):
		x = x + self.attn(self.ln_1(x))
		x = x + self.mlp(self.ln_2(x))
		return x
		


class GPT(nn.Module):

	def __init__(self, config: GPTConfig):
		super().__init__()
		self.config = config

		# these come from the GPT2 params
		self.wte = nn.Embedding(config.vocab_size, config.n_embed)
		self.wpe = nn.Embedding(config.block_size, config.n_embed)
		self.h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
		self.ln_f = nn.LayerNorm(config.n_embed) # final layer norm

		self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False) # final classifier

		# weight sharing scheme -- words with the same encoding meaning should have similar prediction outputs at the end
		self.wte.weight = self.lm_head.weight
		for name, module in self.named_modules():
			self._init_weights(name, module)

	def forward(self, idx, targets=None):
		# idx is (B, T)
		B, T = idx.size()
		assert T <= self.config.block_size, f"cannot forward a sequence of length {T}, block size is {self.config.block_size}"

		# embed the tokens and positions
		pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
		pos_emb = self.wpe(pos) # (T, n_embed)
		tok_emb = self.wte(idx) # (B, T, n_embed)
		x = tok_emb + pos_emb

		# main blocks of the transformer
		for block in self.h:
			x = block(x)
		
		# final layer norm and the classifier
		x = self.ln_f(x)
		logits = self.lm_head(x) # (B, T, vocab_size)
		loss = None
		if targets is not None:
			loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
		return logits, loss
	
	# TODO: check if this actually does anything at all
	def _init_weights(self, name, module):
		if isinstance(module, nn.Linear):
			std = 0.02
			if name.endswith('c_proj'):
				std *= (2*self.config.n_layer) ** -0.5 # 1/sqrt(nLayers)
			torch.nn.init.normal_(module.weight, mean=0.0, std=std)
			if module.bias is not None:
				torch.nn.init.zeros_(module.bias)
		elif isinstance(module, nn.Embedding):
			torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


def sample_random_model():
	max_length = 50
	num_return_sequences = 5

	enc = tiktoken.get_encoding('gpt2')
	tokens = enc.encode('Hello, I am a language model')
	print(f'{tokens=}')
	tokens = torch.tensor(tokens, dtype=torch.long)
	tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
	x = tokens.to(device)
	print(f'Starting {x.shape=}')

	model = GPT(GPTConfig()).to(device)
	model.eval()

	print(f'{x.shape=}')

	while x.size(1) < max_length:
		with torch.no_grad():
			logits = model(x) # (B, T, vocab_size)
			logits = logits[:, -1, :] # (B, vocab_size)
			probs = F.softmax(logits, dim=-1)
			topk_probs, topk_indecies = torch.topk(probs, 50, dim=-1)
			ix=torch.multinomial(topk_probs, 1)
			xcol = torch.gather(topk_indecies, -1, ix)
			x = torch.cat((x, xcol), dim=1)

	for i in range(x.shape[0]):
		tokens = x[i, :max_length].tolist()
		decoded = enc.decode(tokens)
		print(">", decoded)


# sample_random_model()
with open('tinyshakespeare.txt', 'r') as f:
	tinyshakespeare = f.read()

train_loader = DataLoaderLite(B=4, T=1024)
torch.set_float32_matmul_precision('high')


model = GPT(GPTConfig())
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, foreach=False, fused=False)

for i in range(50):
	# these are already on the gpu
	t0 = time.time()
	x, y = train_loader.next_batch()

	optimizer.zero_grad()
	with torch.autocast(device_type='mps', dtype=torch.float16):
		logits,loss = model(x, y)
	loss.backward()
	optimizer.step()
	if (device == 'mps'):
		torch.mps.synchronize()
	if (device == 'cuda'):
		torch.cuda.synchronize()
	dt = time.time()-t0
	print(f'step {i} loss: {loss.item()} dt={dt:.2f}')
 
