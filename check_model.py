"""
check_model.py
~~~~~~~~~~~~~~
Run this to diagnose why the model fails to load.

Usage (from project root, in your conda myenv):
    python check_model.py
"""
import sys
import traceback
from pathlib import Path

print(f"Python: {sys.version}")
print(f"CWD:    {Path.cwd()}\n")

# ── Check scikit-learn version ────────────────────────────────────────────────
try:
    import sklearn
    print(f"scikit-learn:  {sklearn.__version__}")
except ImportError:
    print("scikit-learn:  NOT INSTALLED")

try:
    import joblib
    print(f"joblib:        {joblib.__version__}")
except ImportError:
    print("joblib:        NOT INSTALLED")

print()

# ── Try loading each candidate ─────────────────────────────────────────────────
candidates = [
    "complete_fraud_pipeline.joblib",
    "models/fraud_pipeline.joblib",
]

for path in candidates:
    p = Path(path)
    print(f"Checking: {p.resolve()}")
    print(f"  exists: {p.exists()}")
    if p.exists():
        print(f"  size:   {p.stat().st_size / 1024:.1f} KB")
        try:
            artifact = joblib.load(p)
            print(f"  type:   {type(artifact)}")
            if hasattr(artifact, 'predict'):
                print(f"  ✅ Model loaded OK — has .predict()")
            elif isinstance(artifact, dict):
                print(f"  ✅ Dict keys: {list(artifact.keys())}")
            else:
                print(f"  ⚠️  Unknown type: {type(artifact)}")
        except Exception as e:
            print(f"  ❌ FAILED: {e}")
            traceback.print_exc()
    print()

# ── Offer to retrain ──────────────────────────────────────────────────────────
print("=" * 60)
data_file = Path("creditcard.csv")
if data_file.exists():
    print(f"✅ creditcard.csv found ({data_file.stat().st_size / 1024 / 1024:.1f} MB)")
    print("   Run this to retrain with your current scikit-learn version:")
    print("   python -c \"")
    print("   import pandas as pd, joblib")
    print("   from sklearn.ensemble import RandomForestClassifier")
    print("   from sklearn.preprocessing import RobustScaler")
    print("   from sklearn.pipeline import Pipeline")
    print("   df = pd.read_csv('creditcard.csv').dropna().drop_duplicates()")
    print("   X = df.drop(['Class'], axis=1)")
    print("   y = df['Class']")
    print("   pipe = Pipeline([('scaler', RobustScaler()), ('clf', RandomForestClassifier(n_estimators=100, random_state=42))])")
    print("   pipe.fit(X, y)")
    print("   joblib.dump(pipe, 'complete_fraud_pipeline.joblib')")
    print("   print('Done!')\"")
else:
    print(f"❌ creditcard.csv not found at {data_file.resolve()}")
    print("   Download it from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
