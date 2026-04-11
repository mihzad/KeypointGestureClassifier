import math
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW

from sklearn.metrics import precision_score, recall_score, accuracy_score
import mlflow
from torchinfo import summary

from helpers.common import curr_time, printshare

from model import create_model
from augmentations import create_visual_transforms, create_keypoint_transforms

from helpers.data_loading import KeypointsDataset, VideoFramesFolderDataset, create_dataloaders
from lr_scheduler import create_scheduler
from helpers.checkpoint_management import load_checkpoint, save_checkoint_and_stats
from tester import perform_testing

import shutil

CHECKPOINT_RUN_ID: str | None = None
IMG_SIZE = 224
RAND_STATE = 44
N_CLASSES = 33
N_WORKSERS = 4


CHECKPOINTS_DIR = Path("checkpoints") #temp local storage for checkpoints, will be cleared after training.
STATS_DIR = CHECKPOINTS_DIR / "stats" #temp local storage for stats, will be cleared after training.
DATA_DIR = Path("data")
DATA_IMAGES_DIR= Path("data_images")

# Set up MLflow tracking URI and global exception hook for MLflow run management
mlflow.set_tracking_uri("sqlite:///mlflow.db")

def perform_training(model,
                     training_set:  KeypointsDataset,
                     validation_set: KeypointsDataset,
                     params: dict,
                     pretrained: bool | Path = False):
    printshare("training preparation...")

    mlflow.set_experiment(params["experiment_name"])
    run_obj = mlflow.start_run(run_name=params["run_name"])
    
    try:
        printshare("====== MLflow ======")
        printshare(f"experiment name: {params['experiment_name']}")
        printshare(f"run name: {params['run_name']}")
        printshare(f"run id: {run_obj.info.run_id}")
        printshare("====================")


        mlflow.log_params(params)
        train_loader, val_loader = create_dataloaders(
            training_set=training_set,
            validation_set=validation_set,
            batch_size=params["batch_size"],
            num_workers=N_WORKSERS)

        #========= optimizer, criterion, scheduler =========

        criterion = nn.CrossEntropyLoss()
        optimizer = AdamW(
            params=filter(lambda p: p.requires_grad, model.parameters()),
            lr=params["lr"], weight_decay=params["w_decay_begin"])
        scheduler = create_scheduler(optimizer, params["scheduler_params"])

        mlflow.log_params({"optimizer": type(optimizer).__name__,
                            "scheduler": type(scheduler).__name__})


        #========== loading checkpoint ==========
        epoch_began_with = 0
        model, optimizer, scheduler, epoch_began_with = load_checkpoint(
            checkpoint_run_id=CHECKPOINT_RUN_ID,
            model=model, optimizer=optimizer, scheduler=scheduler,
            checkpoint_path=pretrained if isinstance(pretrained, str) else None,
            hyperparams_override={"optimizer_weight_decay": params["w_decay_begin"]}
        )
        curr_epoch = epoch_began_with

        #manual testing cycle
        #from scripts.dataset_visualizer import infinite_visualization
        #infinite_visualization(train_set)

        CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
        STATS_DIR.mkdir(parents=True, exist_ok=True)
        printshare("done.")

        #===========================================
        #============= TRAINING PHASE ==============
        #===========================================
        w_decay_mid_applied, w_decay_end_applied = False, False
        while curr_epoch < params["epochs"]:
            printshare(f"[{curr_time()}] epoch {curr_epoch + 1}/{params['epochs']} processing...")

            # applying weight decay adjustments according to schedule
            if curr_epoch > params["epochs"]*params["w_decay_mid_epochs_coef"]:
                if curr_epoch <= params["epochs"]*params["w_decay_end_epochs_coef"]:
                    if not w_decay_mid_applied:
                        for g in optimizer.param_groups:
                            g['weight_decay'] = params["w_decay_mid"]
                        w_decay_mid_applied = True
                else:
                    if not w_decay_end_applied:
                        for g in optimizer.param_groups:
                            g['weight_decay'] = params["w_decay_end"]
                        w_decay_end_applied = True



            # redirecting scaler statistics file for additional performance-based weighting in dataloader every N epochs (if applicable)
            if (curr_epoch-epoch_began_with) >= params["redirect_scaler_statistics_after_epoch"] and \
                  (curr_epoch-epoch_began_with) % params["redirect_scaler_statistics_every_n_epochs"] == 0:
                additional_scaler_statistics_file = STATS_DIR / f'ep_{(curr_epoch+1)-1}_scaling_stats.pth'
                printshare(f"redirecting scaler statistics to epoch {(curr_epoch+1)-1} stats file: {additional_scaler_statistics_file}")
                del train_loader
                train_loader, _ = create_dataloaders(
                    training_set=training_set,
                    validation_set=validation_set,
                    batch_size=params["batch_size"],
                    num_workers=N_WORKSERS,
                    additional_scaler_statistics_file=additional_scaler_statistics_file,
                    additional_scaler_eps=params["weighted_sampler_eps"],
                    train_only=True
                )

            #========== training phase itself ==========
            train_targets, train_predictions, train_loss = perform_training_epoch(
                model=model,
                train_loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
            )
            train_precision = round(100 * precision_score(y_true=train_targets, y_pred=train_predictions, average='macro'), 3)
            train_recall = round(100 * recall_score(y_true=train_targets, y_pred=train_predictions, average='macro'), 3)
            train_accuracy = round(100 * accuracy_score(y_true=train_targets, y_pred=train_predictions), 3)

            
            printshare(f"training done. precision: {train_precision}%; recall: {train_recall}%; accuracy: {train_accuracy}%")


            printshare(f"[{curr_time()}] processing validation phase...")
            val_targets, val_predictions, val_loss = perform_validation_epoch(
                model=model,
                val_loader=val_loader,
                criterion=criterion,
            )
            val_precision = round(100 * precision_score(y_true=val_targets, y_pred=val_predictions, average='macro'), 3)
            val_recall = round(100 * recall_score(y_true=val_targets, y_pred=val_predictions, average='macro'), 3)
            val_accuracy = round(100 * accuracy_score(y_true=val_targets, y_pred=val_predictions), 3)

            
            printshare(f"validation done. precision: {val_precision}%; recall: {val_recall}%; accuracy: {val_accuracy}%\n\n")

            
            stats = {
                "epoch": curr_epoch,

                "val_precision": val_precision,
                "val_recall": val_recall,
                "val_accuracy": val_accuracy,

                "val_targets": val_targets,
                "val_predictions": val_predictions,
                "val_loss": val_loss,


                "train_precision": train_precision,
                "train_recall": train_recall,
                "train_accuracy": train_accuracy,
                

                "train_targets": train_targets,
                "train_predictions": train_predictions,
                "train_loss": train_loss
                }
            
            save_checkoint_and_stats(model, optimizer, scheduler, stats,
                                    CHECKPOINTS_DIR, STATS_DIR)
            curr_epoch += 1

        #final logging
        mlflow.pytorch.log_model(pytorch_model=model, name=f"{type(model).__name__}_final")

        with open("summary.txt", "w", encoding="utf-8") as f:
            model_statistics = summary(model,
                input_size=(64, 16, 42), # (B,T,J*2)
                depth=10,
                col_names=["input_size", "output_size", "num_params"],
                row_settings=["var_names", "depth"])
            f.write(str(model_statistics))
        mlflow.log_artifact(local_path="summary.txt", artifact_path="model/parameter_summary.txt")

        mlflow.end_run()

        #clearing the caches
        os.remove("summary.txt")
        shutil.rmtree(CHECKPOINTS_DIR)

        printshare(f"[{curr_time()}] training successfully finished.")
        return model
    
    except Exception as e:
        printshare(f"An error occurred during training: {e}")
        mlflow.end_run(status="FAILED")
        raise e


def perform_training_epoch(model,
                           train_loader, criterion, optimizer, scheduler):
    targets = []
    predictions = []
    batch_losses = []
    model.train()

    optimizer.zero_grad()

    for inputs, labels in train_loader:
        inputs, labels = inputs.cuda(), labels.cuda()

        outputs = model(inputs)

        pred_vals, pred_classes = torch.max(outputs.data, 1)
        targets.extend(labels.detach().cpu().numpy())
        predictions.extend(pred_classes.detach().cpu().numpy())

        loss = criterion(outputs, labels)
        loss.backward()
        batch_losses.append(loss.item())

        optimizer.step()
        optimizer.zero_grad()

    scheduler.step()
    epoch_loss = sum(batch_losses) / len(batch_losses)
    return targets, predictions, epoch_loss


def perform_validation_epoch(model, val_loader, criterion):
    model.eval()
    with torch.no_grad():
        targets = []
        predictions = []
        batch_losses = []
        for inputs, labels in val_loader:
            inputs, labels = inputs.cuda(), labels.cuda()

            outputs = model(inputs)

            batch_losses.append(criterion(outputs, labels).item())

            pred_vals, pred_classes = torch.max(outputs.data, 1)

            targets.extend(labels.detach().cpu().numpy())
            predictions.extend(pred_classes.detach().cpu().numpy())

        epoch_loss = sum(batch_losses) / len(batch_losses)
        return targets, predictions, epoch_loss



if __name__ == '__main__':

    model_additional_config = {
        "TE_stem_channels": 96,
        "TE_inner_residuals_dim": 32,
        "TE_between_residuals_dim": 96,
        "features_per_encoder":96,
        "classifier_hidden_dim": 256,
        "train_postnorm_jitter": 0.0
        }
    model = create_model(N_CLASSES, model_additional_config)

    _, visual_nontrain_transform = create_visual_transforms(img_size=IMG_SIZE)


    training_setup={
        "experiment_name": "Keypoint Gesture Classifier",
        "run_name": f"{type(model).__name__}_[{curr_time()}]",

        "keypoint_augmentations_config": {
            "rotation_max_angle_deg": 15,
            "stretch_scale_x": (0.8, 1.2),
            "stretch_scale_y": (0.8, 1.2),
            "jitter_strength": 0.04
        },

        "model_additional_config": model_additional_config,

        "redirect_scaler_statistics_after_epoch": 15,
        "redirect_scaler_statistics_every_n_epochs": 2,

        "epochs": 60,
        "batch_size": 64,
        "lr": 1e-3,

        "w_decay_begin": 1e-4,
        "w_decay_mid_epochs_coef": 0.1, #epochs * coef = interval after which decay will reapply to mid value
        "w_decay_mid": 1e-3,
        "w_decay_end_epochs_coef": 0.6, #epochs * coef = interval after which decay will reapply to end value
        "w_decay_end": 1e-4,

        "weighted_sampler_eps": 0.25, #eps value for dataloader w.sampler calculation.
         # sets upper bound (~=1+ln(1/eps)) on relative frequencies of classes oversampled

        "scheduler_params":{
            "warmup_steps": 5,
            "T_0": 6,
            "T_mult": 1.2,
            "decay_factor": 0.9,
            "base_lr": 1e-3,
            "eta_min": 1e-8
        }
    }
    
    keypoint_train_transform, keypoint_nontrain_transform = create_keypoint_transforms(training_setup["keypoint_augmentations_config"])

    train_set = KeypointsDataset(DATA_DIR / "data_train.npz", keypoint_transform=keypoint_train_transform)
    val_set = KeypointsDataset(DATA_DIR / "data_validation.npz", keypoint_transform=keypoint_nontrain_transform)
    test_set = VideoFramesFolderDataset(DATA_IMAGES_DIR / 'test', per_img_transform=visual_nontrain_transform, production_ready=True)

    

    perform_training(model, train_set, val_set,
                     params=training_setup,
                     pretrained=False,
                     )

    #perform_testing(model, batch_size=8, testing_set=train_set,
    #                weights_file=CHECKPOINTS_DIR / f'ep_161_p_94.3_r_93.5_a_93.9_model.pth')



