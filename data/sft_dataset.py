import torch
from torch.utils.data import Dataset
import tiktoken

class SFTDataset(Dataset):
    """
    SFT Dataset with loss masking.
    Masks the prompt tokens so the model only learns to predict the response.
    """
    def __init__(self, data, enc, max_seq=512):
        self.max_seq  = max_seq
        self.examples = []
        skipped = 0
        for item in data:
            tokens, mask = self._format(item, enc)
            if tokens is None:
                skipped += 1
                continue
            self.examples.append((tokens, mask))
        print(f"SFTDataset: {len(self.examples):,} examples ({skipped:,} skipped — too long)")

    def _format(self, item, enc):
        # build prompt (user turn)
        prompt   = f"<|user|>\n{item['instruction']}"
        if item.get("input", "").strip():
            prompt += f"\n{item['input']}"
        prompt  += "\n<|assistant|>\n"

        # build response
        response = item["output"] + "<|endoftext|>"

        # encode separately for masking
        prompt_toks   = enc.encode(prompt,   allowed_special="all")
        response_toks = enc.encode(response, allowed_special="all")
        tokens        = prompt_toks + response_toks

        if len(tokens) > self.max_seq:
            return None, None

        # loss mask: 0=ignore prompt, 1=train on response
        mask = [0] * len(prompt_toks) + [1] * len(response_toks)
        return tokens, mask

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        tokens, mask = self.examples[idx]
        tokens = torch.tensor(tokens, dtype=torch.long)
        mask   = torch.tensor(mask,   dtype=torch.long)
        
        # shift by 1 for next-token prediction
        x = tokens[:-1]
        y = tokens[1:]
        m = mask[1:]
        return x, y, m

def collate_fn(batch):
    """Pad variable-length examples to longest in batch."""
    xs, ys, ms = zip(*batch)
    maxlen = max(x.size(0) for x in xs)
    xp = torch.zeros(len(xs), maxlen, dtype=torch.long)
    yp = torch.full((len(ys), maxlen), -1, dtype=torch.long)  # -1 = ignore_index
    mp = torch.zeros(len(ms), maxlen, dtype=torch.long)
    for i, (x, y, m) in enumerate(zip(xs, ys, ms)):
        T = x.size(0)
        xp[i,:T] = x
        yp[i,:T] = y
        mp[i,:T] = m
    return xp, yp, mp
