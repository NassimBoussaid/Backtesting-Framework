import pytest
import pandas as pd
import os
from backtesting_framework.Utils.Tools import load_data

def test_load_data_dataframe():
    # Vérifie que la fonction load_data retourne le même DataFrame lorsqu'elle reçoit un DataFrame en entrée.
    sample_dataframe = pd.DataFrame(
        {
            "A": [1, 2, 3],
            "B": [4, 5, 6],
            "C": [7, 8, 9]
        },
        index=pd.date_range("2022-01-01", periods=3)
    )
    result = load_data(sample_dataframe)
    pd.testing.assert_frame_equal(result, sample_dataframe)

def test_load_data_csv():
    # Vérifie que la fonction load_data peut charger correctement un fichier CSV.
    sample_dataframe = pd.DataFrame(
        {
            "A": [1, 2, 3],
            "B": [4, 5, 6],
            "C": [7, 8, 9]
        },
        index=pd.date_range("2022-01-01", periods=3)
    )
    csv_path = "sample.csv"
    sample_dataframe.to_csv(csv_path, index=True)  # Sauvegarde dans un fichier CSV

    try:
        result = load_data(csv_path)
        pd.testing.assert_frame_equal(result, sample_dataframe, check_freq=False)  # Vérifie le contenu sans la fréquence
    finally:
        os.remove(csv_path)  # Nettoyage du fichier temporaire

def test_load_data_parquet():
    # Vérifie que la fonction load_data peut charger correctement un fichier Parquet.
    sample_dataframe = pd.DataFrame(
        {
            "A": [1, 2, 3],
            "B": [4, 5, 6],
            "C": [7, 8, 9]
        },
        index=pd.date_range("2022-01-01", periods=3)
    )
    parquet_path = "sample.parquet"
    sample_dataframe.to_parquet(parquet_path)

    try:
        result = load_data(parquet_path)
        pd.testing.assert_frame_equal(result, sample_dataframe, check_freq=False)  # Vérifie le contenu sans la fréquence
    finally:
        os.remove(parquet_path)  # Nettoyage du fichier temporaire

def test_load_data_invalid_file_format():
    # Vérifie que la fonction load_data lève une erreur lorsqu'elle tente de charger un fichier non supporté.
    invalid_file = "sample.txt"
    with open(invalid_file, "w") as f:
        f.write("Invalid content")

    try:
        with pytest.raises(ValueError, match="Le format de données n'est pas supporté"):
            load_data(invalid_file)
    finally:
        os.remove(invalid_file)  # Nettoyage du fichier temporaire

def test_load_data_invalid_source():
    # Vérifie que la fonction load_data lève une erreur lorsqu'elle reçoit une source non valide.
    with pytest.raises(ValueError, match="Le format de données n'est pas supporté"):
        load_data(12345)  # Entrée invalide (ni fichier ni DataFrame)
