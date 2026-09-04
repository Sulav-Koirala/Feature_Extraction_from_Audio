from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CONFUSION_DIR = OUTPUTS_DIR / "confusion_matrices"

FEATURES_CSV = OUTPUTS_DIR / "features.csv"
METRICS_JSON = OUTPUTS_DIR / "metrics.json"
SPEED_MODEL_PATH = MODELS_DIR / "speed_model.pkl"
LOUDNESS_MODEL_PATH = MODELS_DIR / "loudness_model.pkl"

HF_DATASET_ID = "devrahulbanjara/ne-en-codeswitching-asr-technical-interview"
HF_PARQUET_FILES = [
    "data/train-00000-of-00001.parquet",       
    "data/validation-00000-of-00001.parquet", 
    "data/test-00000-of-00001.parquet",
]

SAMPLE_RATE = 16000        
N_MFCC = 13                
TRIM_TOP_DB = 30          
MIN_DURATION_SEC = 0.5    

LOWER_PERCENTILE = 100.0 / 3.0   
UPPER_PERCENTILE = 200.0 / 3.0   
TEST_SIZE = 0.20
SEED = 42

LOUDNESS_CLASSES = ["Soft", "Normal", "Loud"]
SPEED_CLASSES = ["Slow", "Normal", "Fast"]
