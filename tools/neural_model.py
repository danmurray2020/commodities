"""Neural network models for commodity price prediction.

Implements:
1. 1D CNN — captures local temporal patterns
2. Simple comparison framework vs XGBoost
"""

import json
import sys
import subprocess
from pathlib import Path

import numpy as np

COFFEE_DIR = Path(__file__).parent.parent / "coffee"
COCOA_DIR = Path(__file__).parent.parent / "chocolate"


def train_and_compare(project_dir: Path, name: str, price_col: str) -> dict | None:
    """Train neural models and compare against XGBoost."""
    script = f"""
import json, sys, numpy as np, pandas as pd, joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

sys.path.insert(0, '.')
from features import add_price_features, merge_cot_data, merge_weather_data, merge_enso_data, build_target

# Load data
df = pd.read_csv('data/combined_features.csv', index_col=0, parse_dates=True)
df = add_price_features(df)
df = merge_cot_data(df)
df = merge_weather_data(df)
df = merge_enso_data(df)
df = build_target(df, price_col='{price_col}', horizon=63)
df = df.dropna()

# Load feature list from production model
models_dir = 'models'
for meta_file in ['v2_production_metadata.json', 'production_metadata.json']:
    try:
        with open(f'{{models_dir}}/{{meta_file}}') as f:
            meta = json.load(f)
        break
    except FileNotFoundError:
        continue

feature_cols = [f for f in meta['features'] if f in df.columns]
X_all = df[feature_cols].values
y_dir = df['target_direction'].values
y_ret = df['target_return'].values
n_features = len(feature_cols)

# --- 1D CNN Model ---
class CNN1D(nn.Module):
    def __init__(self, n_features, seq_len=21):
        super().__init__()
        self.conv1 = nn.Conv1d(n_features, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(32, 16)
        self.fc2 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch, seq_len, features) -> (batch, features, seq_len)
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv1(x))
        x = self.dropout(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc2(x))
        return x.squeeze(-1)

# --- Simple MLP for comparison ---
class MLP(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

# --- Walk-forward evaluation ---
SEQ_LEN = 21  # lookback window for CNN
min_train = 504
purge = 63
test_size = 63

n = len(X_all)
splits = []
for i in range(5):
    test_end = n - i * test_size
    test_start = test_end - test_size
    train_end = test_start - purge
    if train_end < min_train:
        break
    splits.append((train_end, test_start, test_end))
splits.reverse()

results = {{'xgb': [], 'cnn': [], 'mlp': []}}

for fold_i, (train_end, test_start, test_end) in enumerate(splits):
    # Prepare data
    X_train_raw = X_all[:train_end]
    y_train = y_dir[:train_end]
    X_test_raw = X_all[test_start:test_end]
    y_test = y_dir[test_start:test_end]

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    # --- XGBoost baseline ---
    clf_params = meta['classification'].get('params', {{}})
    xgb = XGBClassifier(**clf_params, eval_metric='logloss', early_stopping_rounds=30, random_state=42)
    xgb.fit(X_train_scaled, y_train, eval_set=[(X_test_scaled, y_test)], verbose=False)
    xgb_acc = float(accuracy_score(y_test, xgb.predict(X_test_scaled)))
    xgb_proba = xgb.predict_proba(X_test_scaled)[:, 1]
    results['xgb'].append(xgb_acc)

    # --- MLP ---
    device = torch.device('cpu')
    X_train_t = torch.FloatTensor(X_train_scaled).to(device)
    y_train_t = torch.FloatTensor(y_train.astype(float)).to(device)
    X_test_t = torch.FloatTensor(X_test_scaled).to(device)

    mlp = MLP(n_features).to(device)
    optimizer = torch.optim.Adam(mlp.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.BCELoss()

    dataset = TensorDataset(X_train_t, y_train_t)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    mlp.train()
    for epoch in range(100):
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            pred = mlp(batch_X)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()

    mlp.eval()
    with torch.no_grad():
        mlp_proba = mlp(X_test_t).numpy()
    mlp_pred = (mlp_proba > 0.5).astype(int)
    mlp_acc = float(accuracy_score(y_test, mlp_pred))
    results['mlp'].append(mlp_acc)

    # --- 1D CNN ---
    # Create sequences: each sample is a window of SEQ_LEN days
    def create_sequences(X, y, seq_len):
        seqs, labels = [], []
        for i in range(seq_len, len(X)):
            seqs.append(X[i-seq_len:i])
            labels.append(y[i])
        return np.array(seqs), np.array(labels)

    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train, SEQ_LEN)
    # For test, use the last SEQ_LEN days before each test point
    X_test_seqs = []
    all_scaled = np.vstack([X_train_scaled, X_test_scaled])
    offset = train_end  # where test starts in all_scaled
    for i in range(len(X_test_scaled)):
        idx = offset + i  # position in all_scaled (approximate)
        start = max(0, train_end - SEQ_LEN + i)
        end = train_end + i
        if end - start < SEQ_LEN:
            # Pad with first available
            pad = np.zeros((SEQ_LEN - (end - start), n_features))
            seq = np.vstack([pad, X_all[start:end]])
        else:
            seq = scaler.transform(X_all[end-SEQ_LEN:end])
        X_test_seqs.append(seq)
    X_test_seq = np.array(X_test_seqs)

    X_train_seq_t = torch.FloatTensor(X_train_seq).to(device)
    y_train_seq_t = torch.FloatTensor(y_train_seq.astype(float)).to(device)
    X_test_seq_t = torch.FloatTensor(X_test_seq).to(device)

    cnn = CNN1D(n_features, SEQ_LEN).to(device)
    optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001, weight_decay=1e-4)

    seq_dataset = TensorDataset(X_train_seq_t, y_train_seq_t)
    seq_loader = DataLoader(seq_dataset, batch_size=64, shuffle=True)

    cnn.train()
    for epoch in range(100):
        for batch_X, batch_y in seq_loader:
            optimizer.zero_grad()
            pred = cnn(batch_X)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()

    cnn.eval()
    with torch.no_grad():
        cnn_proba = cnn(X_test_seq_t).numpy()
    cnn_pred = (cnn_proba > 0.5).astype(int)
    cnn_acc = float(accuracy_score(y_test, cnn_pred))
    results['cnn'].append(cnn_acc)

    print(f'Fold {{fold_i}}: XGB={{xgb_acc:.2%}}, MLP={{mlp_acc:.2%}}, CNN={{cnn_acc:.2%}}', file=sys.stderr)

# --- Ensemble: average probabilities from all 3 ---
# (Can't ensemble across folds, but report if ensemble would help)

summary = {{}}
for model_name in ['xgb', 'mlp', 'cnn']:
    accs = results[model_name]
    summary[model_name] = {{
        'fold_accuracies': [round(a, 4) for a in accs],
        'avg_accuracy': round(float(np.mean(accs)), 4),
        'std_accuracy': round(float(np.std(accs)), 4),
    }}

print(json.dumps(summary))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, cwd=str(project_dir), timeout=1200,
    )
    stderr = result.stderr
    if result.returncode != 0:
        print(f"  ERROR: {stderr[-400:]}")
        return None

    # Print fold-level results from stderr
    for line in stderr.split("\n"):
        if line.startswith("Fold"):
            print(f"  {line}")

    try:
        return json.loads(result.stdout.strip().split("\n")[-1])
    except (json.JSONDecodeError, IndexError):
        print(f"  ERROR parsing output")
        return None


def main():
    print("=" * 60)
    print("NEURAL NETWORK vs XGBOOST COMPARISON")
    print("=" * 60)

    for project_dir, name, price_col in [
        (COFFEE_DIR, "Coffee", "coffee_close"),
        (COCOA_DIR, "Cocoa", "cocoa_close"),
    ]:
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")
        r = train_and_compare(project_dir, name, price_col)
        if r:
            print(f"\n  {'Model':<10} {'Avg Acc':>8} {'Std':>8} {'Folds':>40}")
            print(f"  {'-'*68}")
            for model_name in ['xgb', 'mlp', 'cnn']:
                m = r[model_name]
                folds = ", ".join([f"{a:.0%}" for a in m['fold_accuracies']])
                marker = " <-- BEST" if m['avg_accuracy'] == max(r[k]['avg_accuracy'] for k in r) else ""
                print(f"  {model_name.upper():<10} {m['avg_accuracy']:>7.1%} {m['std_accuracy']:>7.1%} {folds:>40}{marker}")

            # Recommendation
            best = max(r, key=lambda k: r[k]['avg_accuracy'])
            xgb_acc = r['xgb']['avg_accuracy']
            best_nn = max(r['mlp']['avg_accuracy'], r['cnn']['avg_accuracy'])
            if best_nn > xgb_acc + 0.02:
                print(f"\n  Neural net ({best.upper()}) outperforms XGBoost by {best_nn - xgb_acc:+.1%}")
                print(f"  Consider adding to ensemble")
            elif abs(best_nn - xgb_acc) <= 0.02:
                print(f"\n  Neural nets are comparable to XGBoost (within 2%)")
                print(f"  Could add to ensemble for diversity, but marginal gain")
            else:
                print(f"\n  XGBoost still wins by {xgb_acc - best_nn:+.1%}")
                print(f"  Stick with XGBoost — neural nets don't add value here")


if __name__ == "__main__":
    main()
