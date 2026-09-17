import pandas as pd
import numpy as np
import os

def fetch_cftc_data(years_back=None):
    """
    Regola 1: Legge il database COT sincronizzato dal bridge locale.
    Se il file manca, si arresta. Nessun dato fittizio.
    """
    file_path = "cot_history.csv"
    if not os.path.exists(file_path):
        raise FileNotFoundError("🚨 BASE DATI COT MANCANTE. Esegui cftc_updater.py in locale e fai il push di 'cot_history.csv'.")
        
    df_clean = pd.read_csv(file_path, low_memory=False)
    df_clean['Data'] = pd.to_datetime(df_clean['Data'])
    return df_clean.sort_values('Data')

def calculate_cot_zscores(df_cot, window_weeks=156):
    """Regola 2: Calcolo Z-Score a 156 settimane."""
    if df_cot.empty:
        return pd.DataFrame()

    df = df_cot.sort_values(by=['Asset', 'Data']).copy()
    min_obs = 52 
    
    df['NC_Mean'] = df.groupby('Asset')['Net_NC'].transform(lambda x: x.rolling(window_weeks, min_periods=min_obs).mean())
    df['NC_Std'] = df.groupby('Asset')['Net_NC'].transform(lambda x: x.rolling(window_weeks, min_periods=min_obs).std())
    df['Z_NC'] = (df['Net_NC'] - df['NC_Mean']) / (df['NC_Std'] + 1e-9)
    
    df['Comm_Mean'] = df.groupby('Asset')['Net_Comm'].transform(lambda x: x.rolling(window_weeks, min_periods=min_obs).mean())
    df['Comm_Std'] = df.groupby('Asset')['Net_Comm'].transform(lambda x: x.rolling(window_weeks, min_periods=min_obs).std())
    df['Z_Comm'] = (df['Net_Comm'] - df['Comm_Mean']) / (df['Comm_Std'] + 1e-9)
    
    latest = df.groupby('Asset').tail(1).copy()
    latest = latest.dropna(subset=['Z_NC', 'Z_Comm'])
    
    latest['Z_NC'] = latest['Z_NC'].round(2)
    latest['Z_Comm'] = latest['Z_Comm'].round(2)
    
    latest['Alert_NC'] = np.where(latest['Z_NC'].abs() >= 1.8, "⭐", "")
    latest['Alert_Comm'] = np.where(latest['Z_Comm'].abs() >= 1.8, "⭐", "")
    
    cols_to_return = ['Data', 'Asset', 'Net_NC', 'Z_NC', 'Alert_NC', 'Net_Comm', 'Z_Comm', 'Alert_Comm']
    latest = latest[cols_to_return].sort_values(by='Z_Comm', key=abs, ascending=False).reset_index(drop=True)
    
    return latest