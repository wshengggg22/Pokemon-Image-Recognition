# config.py

# General settings
NUM_WORKERS = 2

# Training hyperparameters
BATCH_SIZE = 64         # can tune with 64, 16
LEARNING_RATE = 1e-4    # can tune with 1e-3, 5e-4, 1e-4
WEIGHT_DECAY = 0     # can tune with 0, 1e-5, 5e-5
DROPOUT_RATE = 0.1      # can tune with 0.1, 0.2, 0.3   

# Model architecture
NUM_CLASSES = 150
INPUT_SIZE = (224, 224)

