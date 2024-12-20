import pandas as pd

def load_data(data_source):
    """
    Charge les données d'un fichier CSV, Parquet ou d'un DataFrame
    """
    if isinstance(data_source, pd.DataFrame):
        return data_source
    elif isinstance(data_source, str):
        if data_source.endswith('.csv'):
            return pd.read_csv(data_source, index_col=0, parse_dates=True)
        elif data_source.endswith('.parquet'):
            return pd.read_parquet(data_source)
    raise ValueError("Le format de données n'est pas supporté. Veuillez fournir un DataFrame ou un fichier CSV/Parquet.")
