import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from videogpt import VQVAE, ScenarioData


def main():
    pl.seed_everything(1234)

    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = VQVAE.add_model_specific_args(parser)
    parser.add_argument('--data_path', type=str, default='D:\GitHub\VideoGPT\data\dataset')
    parser.add_argument('--sequence_length', type=int, default=4)
    parser.add_argument('--resolution', type=int, default=120)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=8)
    #parser.add_argument('--gpus', type=int, default=1)

    args = parser.parse_args()

    data = ScenarioData(args)
    # pre-make relevant cached files if necessary
    data.train_dataloader()
    data.test_dataloader()
    model = VQVAE(args)

    callbacks = []
    callbacks.append(ModelCheckpoint(monitor='val/recon_loss', mode='min'))

    kwargs = dict()
    if args.gpus > 1:
        kwargs = dict(distributed_backend='ddp', gpus=args.gpus)
    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks,
                                            max_steps=200000,**kwargs)

    trainer.fit(model, data)


if __name__ == '__main__':
    main()

