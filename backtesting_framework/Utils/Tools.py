import pandas as pd

def load_data(data_source):
    """
    Chargement des données à partir de différentes sources :
    Support des fichiers CSV, Parquet, DataFrame pandas.

    :param data_source: Source des données (fichier ou structure de données en mémoire).
    :return: Données chargées sous forme de DataFrame pandas.
    :raises ValueError: Format de données non supporté.
    """
    if isinstance(data_source, pd.DataFrame):
        return data_source
    elif isinstance(data_source, str):
        if data_source.endswith('.csv'):
            return pd.read_csv(data_source, index_col=0, parse_dates=True)
        elif data_source.endswith('.parquet'):
            return pd.read_parquet(data_source)
    raise ValueError("Le format de données n'est pas supporté. "
                     "Veuillez fournir un dict, un DataFrame ou un fichier CSV/Parquet.")
