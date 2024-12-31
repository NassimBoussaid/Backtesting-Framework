import pytest
from datetime import datetime
from backtesting_framework.Core.Calendar import Calendar

def test_calendar_initialization():
    # Vérifie que l'initialisation du calendrier fonctionne correctement avec des dates valides.
    calendar = Calendar(frequency='daily', start_date='2025-01-01', end_date='2025-01-31')
    assert calendar.start_date == datetime(2025, 1, 1)
    assert calendar.end_date == datetime(2025, 1, 31)
    assert len(calendar.all_dates) > 0
    assert len(calendar.rebalancing_dates) > 0

def test_calendar_holidays():
    # Vérifie que les jours fériés sont correctement identifiés dans le calendrier.
    calendar = Calendar(frequency='daily', start_date='2025-01-01', end_date='2025-01-31')
    holidays = calendar.holidays
    assert datetime(2025, 1, 1).date() in holidays  # Nouvel An

def test_is_rebalancing_date():
    # Vérifie si une date est une date de rebalancement valide ou non.
    calendar = Calendar(frequency='daily', start_date='2025-01-01', end_date='2025-01-31')
    assert calendar.is_rebalancing_date('2025-01-02')  # Valide
    assert not calendar.is_rebalancing_date('2025-01-01')  # Jour férié

def test_add_existing_rebalancing_date():
    # Vérifie que l'ajout d'une date de rebalancement déjà existante lève une erreur.
    calendar = Calendar(frequency='daily', start_date='2025-01-01', end_date='2025-01-31')
    with pytest.raises(ValueError, match="The date is already a rebalancing date."):
        calendar.add_rebalancing_date('2025-01-15')

def test_remove_rebalancing_date():
    # Vérifie que la suppression d'une date de rebalancement fonctionne correctement.
    calendar = Calendar(frequency='daily', start_date='2025-01-01', end_date='2025-01-31')
    calendar.remove_rebalancing_date('2025-01-15')
    assert datetime(2025, 1, 15) not in calendar.rebalancing_dates

def test_adjust_to_next_trading_day():
    # Vérifie que l'ajustement au prochain jour de trading fonctionne correctement.
    calendar = Calendar(frequency='daily', start_date='2025-01-01', end_date='2025-01-31')
    adjusted_date = calendar._adjust_to_next_trading_day(datetime(2025, 1, 4))  # Samedi
    assert adjusted_date == datetime(2025, 1, 6)  # Lundi

def test_invalid_frequency():
    # Vérifie que l'utilisation d'une fréquence non supportée lève une erreur.
    with pytest.raises(ValueError, match="Unsupported frequency 'invalid'."):
        Calendar(frequency='invalid', start_date='2025-01-01', end_date='2025-01-31')

def test_invalid_date_format():
    # Vérifie que des dates mal formatées lève une erreur lors de l'initialisation.
    with pytest.raises(ValueError, match="Error parsing dates"):
        Calendar(frequency='daily', start_date='2025-13-01', end_date='2025-01-31')  # Mois invalide

def test_start_date_after_end_date():
    # Vérifie que le calendrier ne peut pas être initialisé avec une start_date après end_date.
    with pytest.raises(ValueError, match="start_date must be earlier than end_date."):
        Calendar(frequency='daily', start_date='2025-02-01', end_date='2025-01-01')

def test_adjust_to_next_trading_day_out_of_range():
    # Vérifie que l'ajustement au prochain jour de trading hors de la plage lève une erreur.
    calendar = Calendar(frequency='daily', start_date='2025-01-01', end_date='2025-01-10')
    with pytest.raises(ValueError, match="Adjusted date exceeds the end_date range."):
        calendar._adjust_to_next_trading_day(datetime(2025, 1, 11))  # Hors plage

def test_add_rebalancing_date_out_of_range():
    # Vérifie que l'ajout d'une date hors de la plage lève une erreur.
    calendar = Calendar(frequency='daily', start_date='2025-01-01', end_date='2025-01-31')
    with pytest.raises(ValueError, match="The new rebalancing date must be within the start_date and end_date range."):
        calendar.add_rebalancing_date('2025-02-01')  # Hors plage

def test_remove_nonexistent_rebalancing_date():
    # Vérifie que la suppression d'une date inexistante lève une erreur.
    calendar = Calendar(frequency='daily', start_date='2025-01-01', end_date='2025-01-31')
    with pytest.raises(ValueError, match="The date is not in the list of rebalancing dates."):
        calendar.remove_rebalancing_date('2025-01-01')  # Date inexistante

def test_is_rebalancing_date_invalid_format():
    # Vérifie que la vérification d'une date avec un format invalide lève une erreur.
    calendar = Calendar(frequency='daily', start_date='2025-01-01', end_date='2025-01-31')
    with pytest.raises(ValueError, match="Error parsing date 'invalid-date'."):
        calendar.is_rebalancing_date('invalid-date')  # Format invalide

