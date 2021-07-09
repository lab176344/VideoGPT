import os
import argparse
import torch

from videogpt import ScenarioData, VideoGPT, load_videogpt
from videogpt.utils import save_video_grid


parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str, default='lightning_logs/version_19/checkpoints/epoch=93-step=19999.ckpt')
parser.add_argument('--n', type=int, default=50)
args = parser.parse_args()
n = args.n

if not os.path.exists(args.ckpt):
    gpt = load_videogpt(args.ckpt)
else:
    gpt = VideoGPT.load_from_checkpoint(args.ckpt)
gpt = gpt.cuda()
gpt.eval()
args = gpt.hparams['args']

args.batch_size = n
data = ScenarioData(args)
loader = data.test_dataloader()
batch = next(iter(loader))
batch = {k: v.cuda() for k, v in batch.items()}

samples = gpt.sample(n, batch)
save_video_grid(samples, 'samples.mp4')
