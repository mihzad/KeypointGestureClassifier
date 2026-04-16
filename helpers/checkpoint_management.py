from pathlib import Path
import mlflow
import torch
import matplotlib.pyplot as plt

from helpers.mlflow_utils import prepare_confusion_matrix
from helpers.common import printshare
def load_checkpoint(checkpoint_run_id: str | None, model, optimizer, scheduler, checkpoint_path: str | None, hyperparams_override):

    epoch = 0
    if checkpoint_run_id is None:
        printshare("No checkpoint to load.")
        return model, optimizer, scheduler, epoch

    local_path = mlflow.artifacts.download_artifacts(run_id=checkpoint_run_id, artifact_path=checkpoint_path)
    if local_path is None or checkpoint_path is None:
        printshare("Checkpoint path/artifact not found.")
        return model, optimizer, scheduler, epoch
    
    printshare("Loading pretrained model, optimizer & scheduler state dicts...")
    checkpoint = torch.load(local_path)

    if 'model' not in checkpoint:
        missing, unexpected = model.load_state_dict(checkpoint, strict=False)
        printshare("got no optimizer & scheduler state dicts. model state dict set up successfully.")

    else:
        missing, unexpected = model.load_state_dict(checkpoint['model'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer'])
        for g in optimizer.param_groups:
            g['weight_decay'] = hyperparams_override["optimizer_weight_decay"]

        #scheduler.load_state_dict(checkpoint["scheduler"])
        scheduler.last_epoch = checkpoint['epoch']
        epoch = checkpoint['epoch'] + 1

        printshare("model, optimizer, scheduler dicts - set up successfully.")


    printshare(f"[DEBUG] model missing statedict vals: {missing};")
    printshare(f"[DEBUG] model unexpected statedict vals: {unexpected}")

    return model, optimizer, scheduler, epoch


def save_checkoint_and_stats(model, optimizer, scheduler, stats,
                              checkpoints_dir: Path, stats_dir: Path):
    #======= metrics & confusion matrices ========
    mlflow.log_metrics({
            "train_loss": stats["train_loss"],
            "train_precision": stats["train_precision"],
            "train_recall": stats["train_recall"],
            "train_accuracy": stats["train_accuracy"]
            }, step=stats["epoch"])
        
    train_cm_fig = prepare_confusion_matrix(stats["train_targets"], stats["train_predictions"])
    mlflow.log_figure(train_cm_fig, f"plots/train/ep_{stats['epoch']}_confusion_matrix.png")
    plt.close()
    
    mlflow.log_metrics({
            "val_loss": stats["val_loss"],
            "val_precision": stats["val_precision"],
            "val_recall": stats["val_recall"],
            "val_accuracy": stats["val_accuracy"]
            }, step=stats["epoch"])
        
    val_cm_fig = prepare_confusion_matrix(stats["val_targets"], stats["val_predictions"])
    mlflow.log_figure(val_cm_fig, f"plots/val/ep_{stats['epoch']}_confusion_matrix.png")
    plt.close()


    model_info_str = f"ep_{stats['epoch']+1}_p_{round(stats['val_precision'], 1)}_r_{round(stats['val_recall'], 1)}_a_{round(stats['val_accuracy'], 1)}"

    torch.save({ # model
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': stats["epoch"],

    }, checkpoints_dir / f'{model_info_str}_model.pth')

    mlflow.log_artifact(checkpoints_dir / f'{model_info_str}_model.pth', artifact_path="checkpoints")


    torch.save(stats, #stats
        stats_dir / f'{model_info_str}_stats.pth')
    
    mlflow.log_artifact(stats_dir / f'{model_info_str}_stats.pth', artifact_path="checkpoints/stats")

    torch.save(stats, #stats_for_additional_scaler_statistics_file, 0-based epoch count
        stats_dir / f'ep_{stats["epoch"]+1}_scaling_stats.pth')