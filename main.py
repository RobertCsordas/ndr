from typing import Optional
import framework
import tasks
import os
import torch
torch.backends.cudnn.benchmark = True


def register_args(parser: framework.helpers.ArgumentParser):
    tasks.register_args(parser)
    parser.add_argument("-batch_size", default=128)
    parser.add_argument("-lr", default=1e-3)
    parser.add_argument("-wd", default=0.0)
    parser.add_argument("-lr_warmup", default=0)
    parser.add_argument("-test_interval", default=1000)
    parser.add_argument("-state_size", default=128)
    parser.add_argument("-n_layers", default=2)
    parser.add_argument("-stop_after", default="None", parser=parser.int_or_none_parser)
    parser.add_argument("-task", default="tuple")
    parser.add_argument("-dropout", default=0.0)
    parser.add_argument("-grad_clip", default="1.0", parser=parser.float_or_none_parser)
    parser.add_argument("-embedding_size", default="16", parser=parser.int_or_none_parser)
    parser.add_argument("-encoder_decoder.n_think_steps", default=0)
    parser.add_argument("-transformer.n_heads", default=4)
    parser.add_argument("-transformer.use_paper_lr_schedule", default=False)
    parser.add_argument("-transformer.variant", default="standard")
    parser.add_argument("-transformer.ff_multiplier", default=2.0)
    parser.add_argument("-transformer.encoder_n_layers", default=3)
    parser.add_argument("-transformer.decoder_n_layers", default="3", parser=parser.int_or_none_parser)
    parser.add_argument("-transformer.tied_embedding", default=True)
    parser.add_argument("-transformer.attention_dropout", default=0.0)
    parser.add_argument("-test_batch_size", default="None", parser=parser.int_or_none_parser)
    parser.add_argument("-restore_pretrained", type=str)
    parser.add_argument("-test_pretrained", default=1)
    parser.add_argument("-train_baseline", default=False, help="Train the model on easy task and test on hard,"
                                                               "no masking")
    parser.add_argument("-lr_sched.steps", default="", parser=parser.int_list_parser)
    parser.add_argument("-lr_sched.gamma", default=0.1)
    parser.add_argument("-lr_sched.type", default="step", choice=["step", "noam"])
    parser.add_argument("-optimizer", default="adam", choice=["adam", "adamw", "sgd"])
    parser.add_argument("-adam.betas", default="0.9,0.999", parser=parser.float_list_parser)
    parser.add_argument("-adam.eps", default=1e-8)
    parser.add_argument("-amp", default=False)
    parser.add_argument("-tied_embedding", default=False)
    parser.add_argument("-label_smoothing", default=0.0)
    parser.add_argument("-max_length_per_batch", default="none", parser=parser.int_or_none_parser)
    parser.add_argument("-length_bucketed_sampling", default=False)
    parser.add_argument("-eos", default=True)
    parser.add_argument("-sos", default=True)
    parser.add_argument("-speedtest", default=False)

    parser.add_profile([
        parser.Profile("scan", {
            "task": "scan",
            "n_layers": 2,
            "state_size": 200,
            "lr": 1e-3,
            "grad_clip": "5",
            "stop_after": 15000,
            "step_per_mask": 15000,
            "batch_size": 256,
            "dropout": 0.5,
            "embedding_size": 16
        }),

   
        parser.Profile("trafo_scan", {
            "task": "trafo_scan",
            "state_size": 128,
            "transformer.n_heads": 8,
            "test_batch_size": 2048
        }, include="scan"),

       

        parser.Profile("listops_trafo", {
            "task": "listops_trafo",
            "state_size": 256,
            "transformer.n_heads": 8,
            "batch_size": 256,
            "lr": 1e-3,
            "grad_clip": 1,
        }),

    ])

def initialize(restore: Optional[str] = None):
    helper = framework.helpers.TrainingHelper(wandb_project_name="length_generalization",
                                              register_args=register_args, extra_dirs=["export", "model_weights"],
                                              log_async=True, restore=restore)


    task = tasks.get_task(helper.args.task)
    task = task(helper)
    return helper, task

def main():
    helper, task = initialize()

    if helper.args.restore_pretrained:
        assert not helper.args.train_baseline

        pretrained = os.path.expanduser(helper.args.restore_pretrained)
        if not helper.args.restore_pretrained.endswith(".pth"):
            pretrained = os.path.join(pretrained, str(helper.args.sweep_id_for_grid_search), "model.pth")

        assert os.path.isfile(pretrained), f"Failed to load pretrained weights. File {pretrained} not found."

        task.load_weights(pretrained)
        if helper.args.test_pretrained:
            helper.log({f"load_validation/{k}": v for k, v in task.validate().items()})
        print("Done. Skipping training...")
    else:
        if helper.args.train_baseline:
            task.set_baseline_mode()

        task.train()

        print("Training finished. Saving model...")
        task.save_weights()

    if helper.args.analysis.enable and not helper.args.train_baseline:
        task.post_train()

    task.finish()
    helper.finish()


if __name__ == "__main__":
    main()
