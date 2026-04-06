import pandas as pd
import sys

DATA_DIR = r'f:\Graduation-Project\10k_synthea_covid19_csv'
OUT = r'f:\Graduation-Project\schema_out.txt'

files = ['patients', 'conditions', 'observations', 'encounters', 'careplans']
lines = []
for f in files:
    df = pd.read_csv(f'{DATA_DIR}/{f}.csv', nrows=3)
    lines.append(f'\n=== {f}.csv ===')
    lines.append(f'Columns: {list(df.columns)}')
    lines.append(df.head(2).to_string())
    lines.append('')

with open(OUT, 'w', encoding='utf-8') as fh:
    fh.write('\n'.join(lines))

print('Done. Check schema_out.txt')
