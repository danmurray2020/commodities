#!/usr/bin/env python3
"""One-time script to generate commodity files from crude_oil templates."""
import os
from pathlib import Path

ROOT = Path(__file__).parent.parent

commodities = [
    ('gold', 'Gold', 'GC=F', 'gold_close', 'gold'),
    ('silver', 'Silver', 'SI=F', 'silver_close', 'silver'),
    ('platinum', 'Platinum', 'PL=F', 'platinum_close', 'platinum'),
    ('corn', 'Corn', 'ZC=F', 'corn_close', 'corn'),
    ('oats', 'Oats', 'ZO=F', 'oats_close', 'oats'),
    ('live_cattle', 'Live Cattle', 'LE=F', 'cattle_close', 'live_cattle'),
    ('lean_hogs', 'Lean Hogs', 'HE=F', 'hogs_close', 'lean_hogs'),
    ('cotton', 'Cotton', 'CT=F', 'cotton_close', 'cotton'),
    ('lumber', 'Lumber', 'LBS=F', 'lumber_close', 'lumber'),
    ('oj', 'Orange Juice', 'OJ=F', 'oj_close', 'oj'),
]

# Read templates
with open(ROOT / 'crude_oil' / 'fetch_data.py') as f:
    fetch_tmpl = f.read()
with open(ROOT / 'crude_oil' / 'features.py') as f:
    feat_tmpl = f.read()
with open(ROOT / 'crude_oil' / 'train.py') as f:
    train_tmpl = f.read()

for key, name, ticker, price_col, dir_name in commodities:
    d = ROOT / dir_name
    (d / 'data').mkdir(parents=True, exist_ok=True)
    (d / 'models').mkdir(parents=True, exist_ok=True)

    # fetch_data.py
    fd = fetch_tmpl.replace('Crude Oil', name).replace('CL=F', ticker)
    fd = fd.replace('crude_oil_close', price_col).replace('crude_oil_prices', f'{key}_prices')
    (d / 'fetch_data.py').write_text(fd)

    # features.py
    ft = feat_tmpl.replace('Crude Oil', name).replace('crude_oil_close', price_col)
    (d / 'features.py').write_text(ft)

    # train.py
    tr = train_tmpl.replace('Crude Oil', name)
    tr = tr.replace('CRUDE OIL', name.upper())
    tr = tr.replace('crude_oil_close', price_col)
    tr = tr.replace('crude_oil_reg', f'{key}_reg')
    tr = tr.replace('crude_oil_clf', f'{key}_clf')
    tr = tr.replace('"crude_oil"', f'"{key}"')
    tr = tr.replace('"CL=F"', f'"{ticker}"')
    (d / 'train.py').write_text(tr)

    print(f'Created {dir_name}/ files')

print('Done!')
