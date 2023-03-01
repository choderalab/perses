#!/bin/env python
"""
Filter CDD export for main medicinal chemistry series

https://app.collaborativedrug.com/vaults/5549/searches/11042338-xOBBXlC_s3dSQsHW3UaO2Q#search_results

Header:
Molecule Name,Canonical PostEra ID,suspected_SMILES,ProteaseAssay_Fluorescence_Dose-Response_Weizmann: IC50 (µM),ProteaseAssay_Fluorescence_Dose-Response_Weizmann: IC50 CI (Lower) (µM),ProteaseAssay_Fluorescence_Dose-Response_Weizmann: IC50 CI (Upper) (µM),ProteaseAssay_Fluorescence_Dose-Response_Weizmann: Hill slope,ProteaseAssay_Fluorescence_Dose-Response_Weizmann: Curve class

"""

cdd_csv_filename = 'CDD CSV Export.csv'
output_csv_filename = 'mainseries.csv'

# Load in data to pandas dataframe
import pandas as pd
df = pd.read_csv(cdd_csv_filename, dtype=str)
print(f'{len(df)} records read')

# Drop NaNs
print(f'Dropping NaNs...')
df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)
print(f'{len(df)} records remain')

# Drop any with ambiguity in suspected SMILES (which will have spaces)
df = df[df['suspected_SMILES'].apply(lambda x: True if len(x.split()) == 1 else False)]

# Rename
df['Title'] = df['Canonical PostEra ID']
df['SMILES'] = df['suspected_SMILES']

# Compute 95%CI width
import numpy as np
def pIC50(IC50_series):
    return -np.log10(IC50_series.astype(float) * 1e-6)

def DeltaG(pIC50):
    kT = 0.593 # kcal/mol for 298 K (25C)
    return - kT * np.log(10.0) * pIC50

df['pIC50_95%CI_LOW'] = pIC50(df['ProteaseAssay_Fluorescence_Dose-Response_Weizmann: IC50 CI (Upper) (µM)'])
df['pIC50_95%CI_HIGH'] = pIC50(df['ProteaseAssay_Fluorescence_Dose-Response_Weizmann: IC50 CI (Lower) (µM)'])
df['95% pIC50 width'] = abs(pIC50(df['ProteaseAssay_Fluorescence_Dose-Response_Weizmann: IC50 CI (Upper) (µM)']) - pIC50(df['ProteaseAssay_Fluorescence_Dose-Response_Weizmann: IC50 CI (Lower) (µM)']))
df['pIC50'] = pIC50(df['ProteaseAssay_Fluorescence_Dose-Response_Weizmann: IC50 (µM)'])
df['dpIC50'] = df['95% pIC50 width'] / 4.0 # estimate of standard error

df['EXP_BINDING_AFFINITY_IN_KCAL_PER_MOL'] = DeltaG(df['pIC50'])
df['EXP_BINDING_AFFINITY_IN_KCAL_PER_MOL_STDERR'] = abs(DeltaG(df['dpIC50']))
df['EXP_BINDING_AFFINITY_IN_KCAL_PER_MOL_95%CI_LOW'] = DeltaG(df['pIC50_95%CI_HIGH'])
df['EXP_BINDING_AFFINITY_IN_KCAL_PER_MOL_95%CI_HIGH'] = DeltaG(df['pIC50_95%CI_LOW'])

# Filter molecules
print('Keeping best measurements for each molecules, sorting by curve class and then 95% pIC50 width')
for molecule_name, molecule_group in df.groupby('Canonical PostEra ID', sort=False):
    print(molecule_name)
    molecule_group.sort_values(by=['ProteaseAssay_Fluorescence_Dose-Response_Weizmann: Curve class', '95% pIC50 width'], inplace=True, ascending=True)
    print(molecule_group)

print('Resulting measurements')
df = df.groupby('Canonical PostEra ID', sort=False).first()
print(df)
print(f'{len(df)} records remain')

# Write molecules
df.to_csv(
    output_csv_filename,
    columns=[
        'SMILES',
        'Title',
        'pIC50',
        'dpIC50',
        'EXP_BINDING_AFFINITY_IN_KCAL_PER_MOL',
        'EXP_BINDING_AFFINITY_IN_KCAL_PER_MOL_STDERR',
        'EXP_BINDING_AFFINITY_IN_KCAL_PER_MOL_95%CI_LOW',
        'EXP_BINDING_AFFINITY_IN_KCAL_PER_MOL_95%CI_HIGH',
        ],
    index=False,
    )
