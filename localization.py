"""
Localization system for Stocker App
Supports multiple languages and currencies
"""
from typing import Dict
import json
from pathlib import Path


class Localization:
    """Manages translations and currency formatting"""
    
    # Exchange rates (approximate, should be fetched from API in production)
    CURRENCY_RATES = {
        'USD': 1.0,
        'EUR': 0.92,  # Approximate rate
        'BGN': 1.80   # Bulgarian Lev (fixed to EUR at 1.95583, but using approximate)
    }
    
    TRANSLATIONS = {
        'en': {
            'app_title': 'STOCKER',
            'app_subtitle': 'Stock Trading & Investing',
            'theme': 'Theme',
            'language': 'Language',
            'currency': 'Currency',
            'strategy': 'Strategy',
            'trading': 'Trading',
            'investing': 'Investing',
            'mixed': 'Mixed',
            'mixed_description': 'Investing & Holding (1 week - 1 month)',
            'stock_lookup': 'Stock Lookup',
            'symbol': 'Symbol',
            'analyze_stock': 'Analyze Stock',
            'scan_market': 'Scan Market',
            'recent_searches': 'Recent Searches',
            'scanning_market': 'Scanning market for recommendations...',
            'market_recommendations': 'Market Recommendations',
            'no_recommendations': 'No buy recommendations found',
            'investment_budget': 'Investment Budget',
            'amount': 'Amount',
            'calculate_potential': 'Calculate Potential',
            'portfolio': 'Portfolio',
            'set_balance': 'Set Balance',
            'wins': 'Wins',
            'losses': 'Losses',
            'win_rate': 'Win Rate',
            'predictions': 'Predictions',
            'view_all': 'View All',
            'accuracy': 'Accuracy',
            'stock_analysis_results': 'Stock Analysis Results',
            'analysis': 'Analysis',
            'charts': 'Charts',
            'indicators': 'Indicators',
            'potential_trade': 'Potential Trade',
            'prediction_statistics': 'Prediction Statistics',
            'total': 'Total',
            'active': 'Active',
            'verified': 'Verified',
            'all_predictions': 'All Predictions',
            'save_current': 'Save Current',
            'verify_all': 'Verify All',
            'clear_old': 'Clear Old',
            'refresh_stats': 'Refresh Stats',
            'no_verified_predictions': 'No verified predictions',
            'pending': 'Pending',
            'correct': 'Correct',
            'incorrect': 'Incorrect',
            'recommendation': 'RECOMMENDATION',
            'confidence': 'Confidence',
            'action': 'Action',
            'shares': 'Shares',
            'budget_used': 'Budget Used',
            'entry_price': 'Entry Price',
            'target_price': 'Target Price',
            'stop_loss': 'Stop Loss',
            'potential_win': 'Potential Win',
            'potential_loss': 'Potential Loss',
            'risk_reward_ratio': 'Risk/Reward Ratio',
            'set_initial_balance': 'Set Initial Balance',
            'initial_balance': 'Initial Balance',
            'save': 'Save',
            'cancel': 'Cancel',
            'error': 'Error',
            'success': 'Success',
            'warning': 'Warning',
            'input_error': 'Input Error',
            'no_analysis': 'No Analysis',
            'please_analyze_stock': 'Please analyze a stock first',
            'no_symbol': 'No Symbol',
            'invalid_prediction': 'Invalid Prediction',
            'prediction_saved': 'Prediction Saved',
            'predictions_verified': 'Predictions Verified',
            'cleared': 'Cleared',
            'confirm': 'Confirm',
            'clear_all_verified': 'Clear all verified predictions?',
            'removed_predictions': 'Removed {count} verified predictions',
            'balance_updated': 'Balance updated successfully',
            'invalid_balance': 'Invalid balance',
            'balance_must_positive': 'Balance must be positive',
            'fetching_data': 'Fetching data for {symbol}...',
            'performing_trading_analysis': 'Performing trading analysis...',
            'performing_mixed_analysis': 'Performing mixed analysis...',
            'performing_fundamental_analysis': 'Performing fundamental analysis...',
            'verifying_predictions': 'Verifying predictions...',
            'loading': 'Loading',
            'close': 'Close',
            'exit': 'Exit',
            'hold_no_trade': 'HOLD Recommendation - No Trade Opportunity',
            'hold_explanation': 'HOLD means the stock is not recommended for trading at this time. Showing hypothetical scenario below.',
            'hypothetical_scenario': 'Hypothetical Scenario (if you were to trade):',
            'no_predictions_verified': 'No new predictions were verified.',
            'info': 'Info',
            'confirm_delete': 'Confirm Delete',
            'delete_prediction': 'Delete prediction',
            'prediction_deleted': 'Prediction deleted',
        },
        'bg': {
            'app_title': 'СТОКЕР',
            'app_subtitle': 'Търговия и Инвестиции с Акции',
            'theme': 'Тема',
            'language': 'Език',
            'currency': 'Валута',
            'strategy': 'Стратегия',
            'trading': 'Търговия',
            'investing': 'Инвестиции',
            'mixed': 'Смесена',
            'mixed_description': 'Инвестиции & Държане (1 седмица - 1 месец)',
            'stock_lookup': 'Търсене на Акция',
            'symbol': 'Символ',
            'analyze_stock': 'Анализирай Акция',
            'scan_market': 'Сканирай Пазара',
            'recent_searches': 'Последни Търсения',
            'scanning_market': 'Сканиране на пазара за препоръки...',
            'market_recommendations': 'Пазарни Препоръки',
            'no_recommendations': 'Не са намерени препоръки за покупка',
            'investment_budget': 'Инвестиционен Бюджет',
            'amount': 'Сума',
            'calculate_potential': 'Изчисли Потенциал',
            'portfolio': 'Портфолио',
            'set_balance': 'Задай Баланс',
            'wins': 'Печалби',
            'losses': 'Загуби',
            'win_rate': 'Процент Печалби',
            'predictions': 'Прогнози',
            'view_all': 'Виж Всички',
            'accuracy': 'Точност',
            'stock_analysis_results': 'Резултати от Анализ на Акция',
            'analysis': 'Анализ',
            'charts': 'Графики',
            'indicators': 'Индикатори',
            'potential_trade': 'Потенциална Търговия',
            'prediction_statistics': 'Статистика на Прогнози',
            'total': 'Общо',
            'active': 'Активни',
            'verified': 'Потвърдени',
            'all_predictions': 'Всички Прогнози',
            'save_current': 'Запази Текуща',
            'verify_all': 'Потвърди Всички',
            'clear_old': 'Изчисти Стари',
            'refresh_stats': 'Обнови Статистика',
            'no_verified_predictions': 'Няма потвърдени прогнози',
            'pending': 'Изчаква',
            'correct': 'Правилна',
            'incorrect': 'Неправилна',
            'recommendation': 'ПРЕПОРЪКА',
            'confidence': 'Увереност',
            'action': 'Действие',
            'shares': 'Акции',
            'budget_used': 'Използван Бюджет',
            'entry_price': 'Входна Цена',
            'target_price': 'Целева Цена',
            'stop_loss': 'Стоп Загуба',
            'potential_win': 'Потенциална Печалба',
            'potential_loss': 'Потенциална Загуба',
            'risk_reward_ratio': 'Коефициент Риск/Печалба',
            'set_initial_balance': 'Задай Начален Баланс',
            'initial_balance': 'Начален Баланс',
            'save': 'Запази',
            'cancel': 'Отказ',
            'error': 'Грешка',
            'success': 'Успех',
            'warning': 'Предупреждение',
            'input_error': 'Грешка при Въвеждане',
            'no_analysis': 'Няма Анализ',
            'please_analyze_stock': 'Моля, анализирайте акция първо',
            'no_symbol': 'Няма Символ',
            'invalid_prediction': 'Невалидна Прогноза',
            'prediction_saved': 'Прогноза Запазена',
            'predictions_verified': 'Прогнози Потвърдени',
            'cleared': 'Изчистено',
            'confirm': 'Потвърди',
            'clear_all_verified': 'Изчисти всички потвърдени прогнози?',
            'removed_predictions': 'Премахнати {count} потвърдени прогнози',
            'balance_updated': 'Балансът е обновен успешно',
            'invalid_balance': 'Невалиден баланс',
            'balance_must_positive': 'Балансът трябва да е положителен',
            'fetching_data': 'Извличане на данни за {symbol}...',
            'performing_trading_analysis': 'Извършване на търговски анализ...',
            'performing_mixed_analysis': 'Извършване на смесен анализ...',
            'performing_fundamental_analysis': 'Извършване на фундаментален анализ...',
            'verifying_predictions': 'Потвърждаване на прогнози...',
            'loading': 'Зареждане',
            'close': 'Затвори',
            'exit': 'Изход',
            'hold_no_trade': 'HOLD Препоръка - Няма Търговска Възможност',
            'hold_explanation': 'HOLD означава, че акцията не се препоръчва за търговия в момента. Показва се хипотетичен сценарий по-долу.',
            'hypothetical_scenario': 'Хипотетичен Сценарий (ако бихте търгували):',
            'no_predictions_verified': 'Няма нови потвърдени прогнози.',
            'info': 'Информация',
            'confirm_delete': 'Потвърди Изтриване',
            'delete_prediction': 'Изтрий прогноза',
            'prediction_deleted': 'Прогнозата е изтрита',
        }
    }
    
    def __init__(self, language: str = 'en', currency: str = 'USD'):
        self.language = language
        self.currency = currency
    
    def t(self, key: str, **kwargs) -> str:
        """Get translated string"""
        translation = self.TRANSLATIONS.get(self.language, self.TRANSLATIONS['en']).get(key, key)
        # Replace placeholders
        if kwargs:
            translation = translation.format(**kwargs)
        return translation
    
    def convert_to_usd(self, amount: float, from_currency: str = None) -> float:
        """Convert amount from specified currency to USD (base currency)"""
        if from_currency is None:
            from_currency = self.currency
        
        # If already USD, no conversion needed
        if from_currency == 'USD':
            return amount
        
        # Get the rate for the source currency
        # Rates are relative to USD, so to convert FROM currency TO USD, we divide
        rate = self.CURRENCY_RATES.get(from_currency, 1.0)
        if rate == 0:
            return amount
        
        # Convert to USD: divide by the rate (e.g., 100 EUR / 0.92 = 108.70 USD)
        return amount / rate
    
    def convert_from_usd(self, amount_usd: float, to_currency: str = None) -> float:
        """Convert amount from USD (base currency) to specified currency"""
        if to_currency is None:
            to_currency = self.currency
        
        # If already USD, no conversion needed
        if to_currency == 'USD':
            return amount_usd
        
        # Get the rate for the target currency
        # Rates are relative to USD, so to convert FROM USD TO currency, we multiply
        rate = self.CURRENCY_RATES.get(to_currency, 1.0)
        return amount_usd * rate
    
    def format_currency(self, amount: float) -> str:
        """Format amount in current currency (assumes amount is in USD)"""
        # Convert from USD to selected currency
        converted = self.convert_from_usd(amount)
        
        # Log conversion for debugging
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"🔍 CURRENCY DEBUG: Converting {amount} USD to {self.currency} = {converted}")
        
        # Format based on currency
        if self.currency == 'EUR':
            return f"€{converted:,.2f}"
        elif self.currency == 'BGN':
            return f"{converted:,.2f} лв"
        else:  # USD
            return f"${converted:,.2f}"
    
    def format_currency_symbol(self) -> str:
        """Get currency symbol"""
        if self.currency == 'EUR':
            return '€'
        elif self.currency == 'BGN':
            return 'лв'
        else:
            return '$'
    
    def set_language(self, language: str):
        """Set language"""
        if language in self.TRANSLATIONS:
            self.language = language
    
    def set_currency(self, currency: str):
        """Set currency"""
        if currency in self.CURRENCY_RATES:
            self.currency = currency
    
    def get_available_languages(self) -> list:
        """Get list of available languages"""
        return list(self.TRANSLATIONS.keys())
    
    def get_available_currencies(self) -> list:
        """Get list of available currencies"""
        return list(self.CURRENCY_RATES.keys())

