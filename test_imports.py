
print("1. Starting imports...", flush=True)
try:
    import pandas as pd
    print("2. pandas imported", flush=True)
    import numpy as np
    print("3. numpy imported", flush=True)
    from sklearn.ensemble import RandomForestClassifier
    print("4. sklearn imported", flush=True)
    from imblearn.over_sampling import SMOTE
    print("5. imblearn imported", flush=True)
    import xgboost
    print("6. xgboost imported", flush=True)
    import lightgbm
    print("7. lightgbm imported", flush=True)
    
    # Try importing local module
    print("8. Importing ml_training...", flush=True)
    from ml_training import StockPredictionML
    print("9. ml_training imported successfully", flush=True)

except Exception as e:
    print(f"CRASHED AT IMPORT: {e}", flush=True)
except ImportError as e:
    print(f"IMPORT ERROR: {e}", flush=True)
