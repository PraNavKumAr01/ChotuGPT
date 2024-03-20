import torch
from config import device
from config import decode
from config import GPTLanguageModel
import argparse

parser = argparse.ArgumentParser(description='gpt43M')
parser.add_argument('--model_path', type=str, default="", required=True)
parser.add_argument('--max_new_tokens', type=int, default=500, required=True)
args = parser.parse_args()

def generate():
    model = GPTLanguageModel()
    model.load_state_dict(torch.load(args.model_path))
    m = model.to(device)
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    text = decode(m.generate(context, max_new_tokens=500)[0].tolist())
    return text

if __name__ == '__main__':
    new_text = generate()
    print(new_text)
