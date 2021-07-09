import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from videogpt import VideoGPT, ScenarioData


def main():
    pl.seed_everything(1234)

    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = VideoGPT.add_model_specific_args(parser)
    parser.add_argument('--data_path', type=str, default='/home/lambda-save/lakshman_experiments_don\'t_delete/VideoGPT/data/dataset')
    parser.add_argument('--resolution', type=int, default=120)
    parser.add_argument('--sequence_length', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--n_labelled', type=int, default=8)

    args = parser.parse_args()

    data = ScenarioData(args)
    # pre-make relevant cached files if necessary
    data.train_dataloader()
    data.test_dataloader()

    args.class_cond_dim = data.n_classes if args.class_cond else None
    model = VideoGPT(args)

    callbacks = []
    callbacks.append(ModelCheckpoint(monitor='val/loss', mode='min', save_top_k=-1))

    kwargs = dict()
    if args.gpus > 1:
        # find_unused_parameters = False to support gradient checkpointing
        kwargs = dict(distributed_backend='ddp', gpus=args.gpus,
                      plugins=[pl.plugins.DDPPlugin(find_unused_parameters=False)])
    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks,
                                            max_steps=args.max_steps, **kwargs)

    trainer.fit(model, data)


if __name__ == '__main__':
    main()

