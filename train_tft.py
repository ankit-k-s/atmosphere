import pandas as pd
import pickle
import os
import torch

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss, MultiLoss

PARAM_FILE = "data/processed/tft/tft_dataset_params.pkl"
DATA_FILE = "data/processed/tft/tft_ready_data_sanitized.parquet"
MODEL_DIR = "models/tft"
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------------------------------------------------
# H100 GPU PRODUCTION CONFIGURATION
# ---------------------------------------------------------
BATCH_SIZE = 1024         # Massive batch size to saturate 80GB VRAM
MAX_EPOCHS = 50           # Full training cycle
LEARNING_RATE = 0.03
NUM_WORKERS = 8           # High CPU thread count to keep the GPU fed
DATASET_FRACTION = 1.0    # 100% of the 13.4 million row dataset

def train_tft_model():
    print("INITIALIZING H100 GPU PRODUCTION TRAINING PROTOCOL...\n")
    
    print("Loading serialized tensors and parameters...")
    with open(PARAM_FILE, "rb") as f:
        dataset_parameters = pickle.load(f)
        
    df = pd.read_parquet(DATA_FILE)
    
    val_cutoff = df["time_idx"].max() - 2880
    train_df = df[df["time_idx"] <= val_cutoff]
    val_df = df[df["time_idx"] > val_cutoff - dataset_parameters["max_encoder_length"]]
    
    if DATASET_FRACTION < 1.0:
        print(f"Sub-sampling active. Reducing training data to {DATASET_FRACTION * 100}%...")
        unique_stations = train_df['station'].unique()
        sampled_train_list = []
        for station in unique_stations:
            station_data = train_df[train_df['station'] == station]
            cutoff_idx = int(len(station_data) * (1.0 - DATASET_FRACTION))
            sampled_train_list.append(station_data.iloc[cutoff_idx:])
        train_df = pd.concat(sampled_train_list, ignore_index=True)

    print("Constructing Multi-Threaded DataLoaders...")
    training_dataset = TimeSeriesDataSet.from_parameters(dataset_parameters, train_df)
    validation_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset, val_df, predict=True, stop_randomization=True
    )
    
    # Enable persistent workers to prevent data loading bottlenecks between epochs
    train_dataloader = training_dataset.to_dataloader(train=True, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, persistent_workers=True, pin_memory=True)
    val_dataloader = validation_dataset.to_dataloader(train=False, batch_size=BATCH_SIZE * 2, num_workers=NUM_WORKERS, persistent_workers=True, pin_memory=True)
    
    num_targets = len(training_dataset.target_names)
    multi_loss = MultiLoss([QuantileLoss()] * num_targets)
    
    print("\nInitializing Temporal Fusion Transformer Architecture...")
    tft = TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=LEARNING_RATE,
        hidden_size=64,             
        attention_head_size=4,      
        dropout=0.1,                
        hidden_continuous_size=32,  
        loss=multi_loss,
        log_interval=50,            
        reduce_on_plateau_patience=4
    )
    
    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=1e-4, patience=5, verbose=True, mode="min"
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=MODEL_DIR, monitor="val_loss", save_top_k=1, mode="min", filename="best-tft-model"
    )
    lr_logger = LearningRateMonitor()
    logger = TensorBoardLogger(save_dir=MODEL_DIR, name="tft_multipollutant")
    
    print("\nConfiguring Lightning GPU Trainer...")
    
    # Ensure torch matrix math takes advantage of the Hopper Tensor Cores
    torch.set_float32_matmul_precision("medium")
    
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed",  # BFloat16 Mixed Precision for H100 acceleration
        gradient_clip_val=0.1,   
        callbacks=[lr_logger, early_stop_callback, checkpoint_callback],
        logger=logger,
        fast_dev_run=False
    )
    
    print("\n[INITIATING NEURAL NETWORK TRAINING LOOP]")
    trainer.fit(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
    
    print("\n[TRAINING COMPLETE] Best model weights have been secured in the models/tft directory.")

if __name__ == "__main__":
    train_tft_model()