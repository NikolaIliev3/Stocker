"""
Investing Strategy Analyzer
Focuses on fundamentals, financial health, valuation, and long-term prospects
"""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class InvestingAnalyzer:
    """Analyzes stocks using fundamental/investing analysis approach"""
    
    def __init__(self):
        # Initialize Market Intelligence components
        try:
            from sentiment_analyzer import SentimentAnalyzer
            from algorithm_improvements import MarketRegimeDetector
            self.sentiment_analyzer = SentimentAnalyzer()
            self.macro_detector = MarketRegimeDetector()
            self.has_market_intel = True
        except ImportError:
            logger.warning("Could not import Market Intelligence components in InvestingAnalyzer")
            self.has_market_intel = False
    
    def analyze(self, stock_data: dict, financials_data: dict, history_data: dict, 
               sentiment_info: dict = None, macro_info: dict = None, current_date=None) -> dict:
        """
        Perform fundamental analysis on stock data
        Returns buy/sell recommendation with reasoning
        """
        try:
            # Filter financials to prevent future data leakage
            if current_date:
                financials_data = self._filter_financials(financials_data, current_date)
            
            # Analyze company fundamentals
            fundamentals = self._analyze_fundamentals(stock_data)
            
            # Analyze financial health
            financial_health = self._analyze_financial_health(financials_data)
            
            # Analyze valuation
            valuation = self._analyze_valuation(stock_data, financials_data)
            
            # Analyze growth prospects
            growth = self._analyze_growth(history_data, financials_data)
            
            # Analyze risk factors
            risk_analysis = self._analyze_risks(stock_data, financials_data)
            
            # Perform Market Intelligence analysis if not provided
            if self.has_market_intel:
                if sentiment_info is None:
                    try:
                        sentiment_info = self.sentiment_analyzer.analyze_sentiment(
                            stock_data.get('symbol', ''), 
                            as_of_date=current_date
                        )
                    except Exception as e:
                        logger.debug(f"Error in sentiment analysis: {e}")
                
                if macro_info is None:
                    try:
                        # Pass current_date to ensure point-in-time macro
                        macro_info = self.macro_detector.detect_regime(current_date=current_date)
                    except Exception as e:
                        logger.debug(f"Error in macro analysis: {e}")
            
            # Make recommendation
            recommendation = self._make_recommendation(
                fundamentals, financial_health, valuation, growth, risk_analysis, stock_data
            )
            
            return {
                "strategy": "investing",
                "recommendation": recommendation,
                "fundamentals": fundamentals,
                "financial_health": financial_health,
                "valuation": valuation,
                "growth": growth,
                "risk_analysis": risk_analysis,
                "reasoning": self._generate_reasoning(
                    recommendation, fundamentals, financial_health,
                    valuation, growth, risk_analysis
                ),
                "sentiment_info": sentiment_info,
                "macro_info": macro_info,
                "market_regime": macro_info
            }
        
        except Exception as e:
            logger.error(f"Error in investing analysis: {e}")
            return {"error": str(e)}
    
    def _filter_financials(self, financials_data: dict, current_date) -> dict:
        """Filter financials to only include data available relative to current_date"""
        if not financials_data or not current_date:
            return financials_data
            
        filtered = financials_data.copy()
        try:
            from datetime import datetime
            # Handle string date if passed
            if isinstance(current_date, str):
                current_date = datetime.strptime(current_date, '%Y-%m-%d')
            
            for key in ['financials', 'balance_sheet', 'cashflow']:
                if key in filtered and isinstance(filtered[key], dict):
                    # Filter keys (which are dates)
                    valid_data = {}
                    for date_str, metrics in filtered[key].items():
                        try:
                            # Parse date string (format is usually YYYY-MM-DD or timestamp str)
                            # Data fetcher converts timestamps to YYYY-MM-DD string
                            report_date = datetime.strptime(str(date_str).split()[0], '%Y-%m-%d')
                            if report_date <= current_date:
                                valid_data[date_str] = metrics
                        except (ValueError, TypeError):
                            # Keep if we can't parse date (safer to keep or drop? drop is safer for backtest)
                            pass
                    filtered[key] = valid_data
                    
        except Exception as e:
            logger.warning(f"Error filtering financials: {e}")
            
        return filtered
    
    def _analyze_fundamentals(self, stock_data: dict) -> dict:
        """Analyze company fundamentals and business model"""
        info = stock_data.get('info', {})
        
        return {
            "company_name": info.get('name', 'Unknown'),
            "sector": info.get('sector', 'Unknown'),
            "industry": info.get('industry', 'Unknown'),
            "market_cap": info.get('market_cap', 0),
            "market_cap_category": self._categorize_market_cap(info.get('market_cap', 0))
        }
    
    def _categorize_market_cap(self, market_cap: float) -> str:
        """Categorize market cap"""
        if market_cap >= 200_000_000_000:
            return "Mega Cap"
        elif market_cap >= 10_000_000_000:
            return "Large Cap"
        elif market_cap >= 2_000_000_000:
            return "Mid Cap"
        elif market_cap >= 300_000_000:
            return "Small Cap"
        else:
            return "Micro Cap"
    
    def _analyze_financial_health(self, financials_data: dict) -> dict:
        """Analyze financial health from financial statements with advanced metrics"""
        try:
            info = financials_data.get('info', {}) if isinstance(financials_data, dict) else {}
            financials = financials_data.get('financials', {})
            balance_sheet = financials_data.get('balance_sheet', {})
            cashflow = financials_data.get('cashflow', {})
            
            health_score = 50  # Base score
            factors = []
            
            # Get key metrics from info dict (yfinance provides these)
            total_debt = info.get('totalDebt', 0) or info.get('totalDebt', 0)
            total_cash = info.get('totalCash', 0) or info.get('totalCashPerShare', 0)
            total_assets = info.get('totalAssets', 0)
            total_liabilities = info.get('totalLiab', 0)
            shareholders_equity = info.get('totalStockholderEquity', 0)
            current_assets = info.get('totalCurrentAssets', 0)
            current_liabilities = info.get('totalCurrentLiabilities', 0)
            operating_cashflow = info.get('operatingCashflow', 0)
            free_cashflow = info.get('freeCashflow', 0)
            revenue = info.get('totalRevenue', 0) or info.get('revenue', 0)
            net_income = info.get('netIncomeToCommon', 0) or info.get('netIncome', 0)
            roe = info.get('returnOnEquity', 0)
            roa = info.get('returnOnAssets', 0)
            profit_margin = info.get('profitMargins', 0)
            operating_margin = info.get('operatingMargins', 0)
            
            # Debt-to-Equity Ratio
            if shareholders_equity and shareholders_equity > 0:
                debt_to_equity = total_debt / shareholders_equity
                if debt_to_equity < 0.3:
                    health_score += 15
                    factors.append(f"Low debt-to-equity ({debt_to_equity:.2f}) - strong balance sheet")
                elif debt_to_equity < 0.6:
                    health_score += 5
                    factors.append(f"Moderate debt-to-equity ({debt_to_equity:.2f})")
                elif debt_to_equity > 1.5:
                    health_score -= 20
                    factors.append(f"High debt-to-equity ({debt_to_equity:.2f}) - financial risk")
                elif debt_to_equity > 1.0:
                    health_score -= 10
                    factors.append(f"Elevated debt-to-equity ({debt_to_equity:.2f})")
            
            # Current Ratio (Liquidity)
            if current_liabilities and current_liabilities > 0:
                current_ratio = current_assets / current_liabilities
                if current_ratio > 2.0:
                    health_score += 10
                    factors.append(f"Strong liquidity (Current Ratio: {current_ratio:.2f})")
                elif current_ratio > 1.5:
                    health_score += 5
                    factors.append(f"Good liquidity (Current Ratio: {current_ratio:.2f})")
                elif current_ratio < 1.0:
                    health_score -= 15
                    factors.append(f"Poor liquidity (Current Ratio: {current_ratio:.2f}) - solvency risk")
                elif current_ratio < 1.2:
                    health_score -= 5
                    factors.append(f"Tight liquidity (Current Ratio: {current_ratio:.2f})")
            
            # Cash Position
            if total_assets and total_assets > 0:
                cash_ratio = total_cash / total_assets
                if cash_ratio > 0.2:
                    health_score += 10
                    factors.append(f"Strong cash position ({cash_ratio*100:.1f}% of assets)")
                elif cash_ratio < 0.05:
                    health_score -= 10
                    factors.append(f"Low cash position ({cash_ratio*100:.1f}% of assets)")
            
            # Cash Flow Analysis
            if operating_cashflow:
                if operating_cashflow > 0:
                    health_score += 10
                    factors.append("Positive operating cash flow")
                else:
                    health_score -= 15
                    factors.append("Negative operating cash flow - red flag")
            
            if free_cashflow:
                if free_cashflow > 0:
                    health_score += 5
                    factors.append("Positive free cash flow")
                else:
                    health_score -= 10
                    factors.append("Negative free cash flow")
            
            # Profitability Metrics
            if roe:
                if roe > 0.20:  # 20%
                    health_score += 15
                    factors.append(f"Excellent ROE ({roe*100:.1f}%) - strong profitability")
                elif roe > 0.15:  # 15%
                    health_score += 10
                    factors.append(f"Good ROE ({roe*100:.1f}%)")
                elif roe < 0.05:  # 5%
                    health_score -= 10
                    factors.append(f"Low ROE ({roe*100:.1f}%) - weak profitability")
            
            if roa:
                if roa > 0.10:  # 10%
                    health_score += 10
                    factors.append(f"Strong ROA ({roa*100:.1f}%)")
                elif roa < 0.03:  # 3%
                    health_score -= 10
                    factors.append(f"Low ROA ({roa*100:.1f}%)")
            
            if profit_margin:
                if profit_margin > 0.20:  # 20%
                    health_score += 10
                    factors.append(f"Excellent profit margin ({profit_margin*100:.1f}%)")
                elif profit_margin > 0.10:  # 10%
                    health_score += 5
                    factors.append(f"Good profit margin ({profit_margin*100:.1f}%)")
                elif profit_margin < 0.05:  # 5%
                    health_score -= 10
                    factors.append(f"Low profit margin ({profit_margin*100:.1f}%)")
            
            if operating_margin:
                if operating_margin > 0.15:  # 15%
                    health_score += 5
                    factors.append(f"Strong operating margin ({operating_margin*100:.1f}%)")
                elif operating_margin < 0.05:  # 5%
                    health_score -= 5
                    factors.append(f"Tight operating margin ({operating_margin*100:.1f}%)")
            
            # Revenue Growth (if we can calculate from history)
            # This would require parsing financial statements properly
            
            # Overall health assessment
            overall_score = max(0, min(100, health_score))
            health_level = "excellent" if overall_score >= 80 else "good" if overall_score >= 65 else "fair" if overall_score >= 50 else "poor"
            
            return {
                "health_score": overall_score,
                "overall_score": overall_score,  # Alias for compatibility
                "health_level": health_level,
                "factors": factors,
                "has_positive_cashflow": operating_cashflow > 0 if operating_cashflow else None,
                "debt_level": "low" if (total_debt and shareholders_equity and total_debt/shareholders_equity < 0.5) else "moderate" if (total_debt and shareholders_equity and total_debt/shareholders_equity < 1.0) else "high",
                "debt_to_equity": total_debt / shareholders_equity if (total_debt and shareholders_equity and shareholders_equity > 0) else None,
                "current_ratio": current_assets / current_liabilities if (current_assets and current_liabilities and current_liabilities > 0) else None,
                "roe": roe,
                "roa": roa,
                "profit_margin": profit_margin
            }
        except Exception as e:
            logger.error(f"Error analyzing financial health: {e}")
            return {
                "health_score": 50,
                "overall_score": 50,
                "health_level": "unknown",
                "factors": ["Limited financial data available"],
                "has_positive_cashflow": None,
                "debt_level": "unknown"
            }
    
    def _analyze_valuation(self, stock_data: dict, financials_data: dict) -> dict:
        """Analyze valuation metrics with advanced indicators"""
        info = stock_data.get('info', {})
        current_price = stock_data.get('price', 0)
        pe_ratio = info.get('trailingPE') or info.get('pe_ratio')
        forward_pe = info.get('forwardPE')
        peg_ratio = info.get('pegRatio')
        dividend_yield = info.get('dividendYield') or info.get('dividend_yield')
        payout_ratio = info.get('payoutRatio')
        price_to_book = info.get('priceToBook')
        price_to_sales = info.get('priceToSalesTrailing12Months')
        enterprise_value = info.get('enterpriseValue')
        ebitda = info.get('ebitda')
        ev_to_ebitda = enterprise_value / ebitda if (enterprise_value and ebitda and ebitda > 0) else None
        
        valuation_score = 50
        factors = []
        
        # P/E Analysis (enhanced)
        if pe_ratio:
            if pe_ratio < 12:
                valuation_score += 25
                factors.append(f"Very low P/E ratio ({pe_ratio:.2f}) - strong undervaluation signal")
            elif pe_ratio < 15:
                valuation_score += 20
                factors.append(f"Low P/E ratio ({pe_ratio:.2f}) suggests undervaluation")
            elif pe_ratio > 30:
                valuation_score -= 25
                factors.append(f"Very high P/E ratio ({pe_ratio:.2f}) - overvaluation risk")
            elif pe_ratio > 25:
                valuation_score -= 20
                factors.append(f"High P/E ratio ({pe_ratio:.2f}) suggests overvaluation")
            else:
                factors.append(f"P/E ratio ({pe_ratio:.2f}) is reasonable")
        
        # Forward P/E (better indicator than trailing)
        if forward_pe:
            if forward_pe < pe_ratio and pe_ratio:
                valuation_score += 5
                factors.append(f"Forward P/E ({forward_pe:.2f}) lower than trailing - earnings growth expected")
            elif forward_pe > pe_ratio * 1.2 and pe_ratio:
                valuation_score -= 5
                factors.append(f"Forward P/E ({forward_pe:.2f}) higher - earnings may decline")
        
        # PEG Ratio (Price/Earnings to Growth) - better than P/E alone
        if peg_ratio:
            if peg_ratio < 0.5:
                valuation_score += 15
                factors.append(f"Excellent PEG ratio ({peg_ratio:.2f}) - strong growth relative to price")
            elif peg_ratio < 1.0:
                valuation_score += 10
                factors.append(f"Good PEG ratio ({peg_ratio:.2f}) - reasonable valuation")
            elif peg_ratio > 2.0:
                valuation_score -= 15
                factors.append(f"High PEG ratio ({peg_ratio:.2f}) - overvalued relative to growth")
            elif peg_ratio > 1.5:
                valuation_score -= 10
                factors.append(f"Elevated PEG ratio ({peg_ratio:.2f}) - growth concerns")
        
        # Price-to-Book Ratio
        if price_to_book:
            if price_to_book < 1.0:
                valuation_score += 15
                factors.append(f"Price below book value (P/B: {price_to_book:.2f}) - potential value play")
            elif price_to_book < 2.0:
                valuation_score += 5
                factors.append(f"Reasonable P/B ratio ({price_to_book:.2f})")
            elif price_to_book > 5.0:
                valuation_score -= 10
                factors.append(f"High P/B ratio ({price_to_book:.2f}) - premium valuation")
        
        # Price-to-Sales Ratio
        if price_to_sales:
            if price_to_sales < 1.0:
                valuation_score += 10
                factors.append(f"Low P/S ratio ({price_to_sales:.2f}) - attractive valuation")
            elif price_to_sales > 5.0:
                valuation_score -= 10
                factors.append(f"High P/S ratio ({price_to_sales:.2f}) - expensive relative to sales")
        
        # EV/EBITDA (Enterprise Value to EBITDA)
        if ev_to_ebitda:
            if ev_to_ebitda < 8:
                valuation_score += 10
                factors.append(f"Low EV/EBITDA ({ev_to_ebitda:.2f}) - attractive valuation")
            elif ev_to_ebitda > 15:
                valuation_score -= 10
                factors.append(f"High EV/EBITDA ({ev_to_ebitda:.2f}) - expensive")
        
        # Dividend Analysis
        if dividend_yield:
            div_yield_pct = dividend_yield if dividend_yield < 1 else dividend_yield / 100
            if div_yield_pct > 0.04:  # 4%
                valuation_score += 10
                factors.append(f"Attractive dividend yield ({div_yield_pct*100:.2f}%)")
            elif div_yield_pct > 0.02:  # 2%
                valuation_score += 5
                factors.append(f"Moderate dividend yield ({div_yield_pct*100:.2f}%)")
            
            # Payout ratio analysis
            if payout_ratio:
                if payout_ratio < 0.5:
                    valuation_score += 5
                    factors.append(f"Low payout ratio ({payout_ratio*100:.1f}%) - room for dividend growth")
                elif payout_ratio > 0.9:
                    valuation_score -= 5
                    factors.append(f"High payout ratio ({payout_ratio*100:.1f}%) - dividend sustainability concern")
        
        return {
            "pe_ratio": pe_ratio,
            "forward_pe": forward_pe,
            "peg_ratio": peg_ratio,
            "price_to_book": price_to_book,
            "price_to_sales": price_to_sales,
            "ev_to_ebitda": ev_to_ebitda,
            "dividend_yield": dividend_yield,
            "payout_ratio": payout_ratio,
            "valuation_score": valuation_score,
            "factors": factors,
            "is_undervalued": valuation_score > 60,
            "is_overvalued": valuation_score < 40
        }
    
    def _analyze_growth(self, history_data: dict, financials_data: dict) -> dict:
        """Analyze growth prospects"""
        try:
            data = history_data.get('data', [])
            if not data:
                return {"growth_score": 50, "factors": ["Insufficient data"]}
            
            # Calculate price growth over time
            prices = [d['close'] for d in data]
            if len(prices) < 2:
                return {"growth_score": 50, "factors": ["Insufficient data"]}
            
            # Calculate CAGR (simplified)
            start_price = prices[0]
            end_price = prices[-1]
            periods = len(prices)
            
            if start_price > 0:
                total_return = (end_price - start_price) / start_price
                # Approximate annualized return
                annualized_return = total_return * (252 / periods) if periods > 0 else 0
            else:
                annualized_return = 0
            
            growth_score = 50
            factors = []
            
            if annualized_return > 0.15:  # 15% annual return
                growth_score += 20
                factors.append("Strong historical price growth")
            elif annualized_return > 0.05:  # 5% annual return
                growth_score += 10
                factors.append("Moderate historical price growth")
            elif annualized_return < -0.1:  # -10% annual return
                growth_score -= 20
                factors.append("Negative historical price growth")
            
            return {
                "growth_score": growth_score,
                "annualized_return": annualized_return * 100,
                "factors": factors,
                "is_growing": annualized_return > 0.05
            }
        except Exception as e:
            logger.error(f"Error analyzing growth: {e}")
            return {"growth_score": 50, "factors": [f"Error: {str(e)}"]}
    
    def _analyze_risks(self, stock_data: dict, financials_data: dict) -> dict:
        """Analyze risk factors"""
        risks = []
        risk_score = 0
        
        info = stock_data.get('info', {})
        market_cap = info.get('market_cap', 0)
        
        # Market cap risk
        if market_cap < 2_000_000_000:  # Less than $2B
            risks.append("Small market cap - higher volatility risk")
            risk_score += 10
        
        # Sector risk (simplified)
        sector = info.get('sector', '')
        if sector in ['Technology', 'Biotechnology']:
            risks.append("Sector may be subject to high volatility")
            risk_score += 5
        
        return {
            "risk_score": risk_score,
            "risks": risks,
            "risk_level": "low" if risk_score < 10 else "moderate" if risk_score < 20 else "high"
        }
    
    def _make_recommendation(self, fundamentals: dict, financial_health: dict,
                           valuation: dict, growth: dict, risk_analysis: dict,
                           stock_data: dict) -> dict:
        """Make buy/sell recommendation based on fundamental analysis"""
        score = 0
        reasons = []
        
        # Financial health (enhanced)
        health_score = financial_health.get('health_score', 50) or financial_health.get('overall_score', 50)
        health_level = financial_health.get('health_level', 'fair')
        
        if health_score >= 80:
            score += 3
            reasons.append("Excellent financial health")
        elif health_score >= 65:
            score += 2
            reasons.append("Strong financial health")
        elif health_score < 40:
            score -= 3
            reasons.append("Weak financial health - high risk")
        elif health_score < 50:
            score -= 2
            reasons.append("Poor financial health")
        
        # Specific health factors
        roe = financial_health.get('roe')
        if roe and roe > 0.20:
            score += 1
            reasons.append(f"Excellent ROE ({roe*100:.1f}%)")
        elif roe and roe < 0.05:
            score -= 1
            reasons.append(f"Low ROE ({roe*100:.1f}%)")
        
        debt_to_equity = financial_health.get('debt_to_equity')
        if debt_to_equity and debt_to_equity < 0.3:
            score += 1
            reasons.append("Low debt burden")
        elif debt_to_equity and debt_to_equity > 1.5:
            score -= 2
            reasons.append("High debt burden - financial risk")
        
        # Valuation (enhanced)
        valuation_score = valuation.get('valuation_score', 50)
        if valuation_score >= 70:
            score += 3
            reasons.append("Stock appears significantly undervalued")
        elif valuation.get('is_undervalued'):
            score += 2
            reasons.append("Stock appears undervalued")
        elif valuation_score <= 30:
            score -= 3
            reasons.append("Stock appears significantly overvalued")
        elif valuation.get('is_overvalued'):
            score -= 2
            reasons.append("Stock appears overvalued")
        
        # PEG ratio (better than P/E alone)
        peg_ratio = valuation.get('peg_ratio')
        if peg_ratio and peg_ratio < 0.5:
            score += 2
            reasons.append(f"Excellent PEG ratio ({peg_ratio:.2f})")
        elif peg_ratio and peg_ratio > 2.0:
            score -= 2
            reasons.append(f"Poor PEG ratio ({peg_ratio:.2f})")
        
        # Growth
        growth_score = growth.get('growth_score', 50)
        if growth.get('is_growing'):
            score += 1
            reasons.append("Positive growth trajectory")
        else:
            score -= 1
            reasons.append("Limited or negative growth")
        
        # Risk
        risk_level = risk_analysis.get('risk_level', 'moderate')
        if risk_level == 'high':
            score -= 1
            reasons.append("High risk factors present")
        
        # Determine recommendation (Aggressive Value Investing)
        # STANDARD LOGIC RESTORED:
        # Good Fundamentals + Undervalued = BUY (Accumulate)
        # Good Fundamentals + Overvalued = SELL (Scale Out) or HOLD
        # Bad Fundamentals = SELL (Avoid/Liquidate)
        
        # Determine strict signal states
        is_undervalued = valuation.get('is_undervalued', False)
        is_overvalued = valuation.get('is_overvalued', False)
        health_excellent = health_score >= 80
        health_poor = health_score < 50
        
        # Default
        action = "HOLD"
        confidence = 50
        
        if score >= 2: # Good Fundamentals
            if is_undervalued:
                 # The Holy Grail: Good Company + Cheap Price
                 action = "BUY"
                 confidence = 90
                 reasons.append("💎 VALUE INVESTING: High Quality Company at Discount Price")
            elif is_overvalued:
                 # Good Company but Expensive
                 action = "SELL"
                 confidence = 70
                 reasons.append("⚠️ Good company but SIGNIFICANTLY OVERVALUED - Consider taking profits")
            else:
                 # Good Company, Fair Price -> BUY/HOLD
                 action = "BUY" # Aggressive: Keep adding to winners
                 confidence = 75
                 reasons.append("📈 High Quality Business - Accumulate/Hold for long term")
                 
        elif score <= -2: # Bad Fundamentals
             action = "SELL"
             confidence = 80
             reasons.append("📉 Deteriorating Fundamentals - High Risk Investment")
             if is_undervalued:
                 reasons.append("⚠️ Stock is 'cheap' for a reason (Value Trap?)")
                 
        else: # Mixed / Average Fundamentals
             if is_undervalued:
                 action = "BUY"
                 confidence = 65
                 reasons.append("💡 Speculative Value Play (Average Fundamentals but Cheap)")
             elif is_overvalued:
                 action = "SELL"
                 confidence = 65
                 reasons.append("⚠️ Overvalued with average fundamentals - Risk to downside")
             else:
                 # Truly middle of the road
                 action = "HOLD"
                 confidence = 50
                 reasons.append("↔️ Average fundamentals and fair valuation - Wait for better setup")

        current_price = stock_data.get('price', 0)
        
        # Long-term targets (1-3 years)
        entry_price = current_price
        
        # Multipliers
        target_mult = 1.2 # Default 20% upside per year? 
        stop_mult = 0.85
        
        if action == "BUY":
            # Target = Intrinsic Value (or estimate)
            # If undervalued, target is "Fair Value" which is higher
            if is_undervalued:
                target_price = current_price * 1.50 # Expect 50% mean reversion + growth
            else:
                target_price = current_price * 1.25 # Expect 25% growth
            
            # Stop Loss (Wide for investing)
            stop_loss = current_price * 0.80 # 20% trailing stop idea
            
            # Entry: If we can get it cheaper?
            # Implied: User wants to enter NOW unless it's a "Wait for dip"
            
        elif action == "SELL":
            # Exit Strategy
            target_price = current_price * 1.10 # Maybe eke out 10% more?
            stop_loss = current_price * 0.90 # Protect gains?
            
        else: # HOLD
            target_price = current_price * 1.10
            stop_loss = current_price * 0.90
        
        return {
            "action": action,
            "confidence": confidence,
            "entry_price": float(current_price),
            "target_price": float(target_price),
            "stop_loss": float(stop_loss),
            "stop_loss": float(stop_loss),
            "time_horizon": "1-3 years",
            "estimated_days": 547, # Approx 1.5 years
            "score": score,
            "reasons": reasons
        }
    
    def _generate_reasoning(self, recommendation: dict, fundamentals: dict,
                          financial_health: dict, valuation: dict,
                          growth: dict, risk_analysis: dict) -> str:
        """Generate human-readable reasoning for the recommendation"""
        action = recommendation.get('action', 'HOLD')
        reasons = recommendation.get('reasons', [])
        
        reasoning = f"Fundamental Analysis Recommendation: {action}\n\n"
        reasoning += f"Confidence: {recommendation.get('confidence', 0)}%\n\n"
        
        reasoning += "Key Factors:\n"
        for reason in reasons:
            reasoning += f"• {reason}\n"
        
        reasoning += f"\nCompany Overview:\n"
        reasoning += f"• Company: {fundamentals.get('company_name', 'Unknown')}\n"
        reasoning += f"• Sector: {fundamentals.get('sector', 'Unknown')}\n"
        reasoning += f"• Market Cap: {fundamentals.get('market_cap_category', 'Unknown')}\n"
        
        reasoning += f"\nValuation Metrics:\n"
        pe = valuation.get('pe_ratio')
        if pe:
            reasoning += f"• P/E Ratio: {pe:.2f}\n"
        forward_pe = valuation.get('forward_pe')
        if forward_pe:
            reasoning += f"• Forward P/E: {forward_pe:.2f}\n"
        peg = valuation.get('peg_ratio')
        if peg:
            reasoning += f"• PEG Ratio: {peg:.2f} ({'Excellent' if peg < 0.5 else 'Good' if peg < 1.0 else 'Fair' if peg < 1.5 else 'Poor'})\n"
        pb = valuation.get('price_to_book')
        if pb:
            reasoning += f"• Price-to-Book: {pb:.2f}\n"
        ps = valuation.get('price_to_sales')
        if ps:
            reasoning += f"• Price-to-Sales: {ps:.2f}\n"
        ev_ebitda = valuation.get('ev_to_ebitda')
        if ev_ebitda:
            reasoning += f"• EV/EBITDA: {ev_ebitda:.2f}\n"
        div_yield = valuation.get('dividend_yield')
        if div_yield:
            div_yield_pct = div_yield if div_yield < 1 else div_yield / 100
            reasoning += f"• Dividend Yield: {div_yield_pct * 100:.2f}%\n"
        payout = valuation.get('payout_ratio')
        if payout:
            reasoning += f"• Payout Ratio: {payout * 100:.1f}%\n"
        
        reasoning += f"\nGrowth Analysis:\n"
        annual_return = growth.get('annualized_return', 0)
        reasoning += f"• Historical Annualized Return: {annual_return:.2f}%\n"
        
        reasoning += f"\nFinancial Health:\n"
        health_level = financial_health.get('health_level', 'unknown')
        health_score = financial_health.get('health_score', 50) or financial_health.get('overall_score', 50)
        reasoning += f"• Overall Health: {health_level.upper()} (Score: {health_score:.0f}/100)\n"
        
        roe = financial_health.get('roe')
        if roe:
            reasoning += f"• ROE: {roe * 100:.1f}%\n"
        roa = financial_health.get('roa')
        if roa:
            reasoning += f"• ROA: {roa * 100:.1f}%\n"
        profit_margin = financial_health.get('profit_margin')
        if profit_margin:
            reasoning += f"• Profit Margin: {profit_margin * 100:.1f}%\n"
        debt_to_equity = financial_health.get('debt_to_equity')
        if debt_to_equity:
            reasoning += f"• Debt-to-Equity: {debt_to_equity:.2f}\n"
        current_ratio = financial_health.get('current_ratio')
        if current_ratio:
            reasoning += f"• Current Ratio: {current_ratio:.2f}\n"
        
        reasoning += f"\nRisk Assessment:\n"
        reasoning += f"• Risk Level: {risk_analysis.get('risk_level', 'unknown').upper()}\n"
        for risk in risk_analysis.get('risks', []):
            reasoning += f"• {risk}\n"
        
        if action == "BUY":
            reasoning += f"\nInvestment Strategy:\n"
            reasoning += f"• Entry Price: ${recommendation.get('entry_price', 0):.2f}\n"
            reasoning += f"• Target Price (1-3 years): ${recommendation.get('target_price', 0):.2f}\n"
            reasoning += f"• Stop Loss: ${recommendation.get('stop_loss', 0):.2f}\n"
            reasoning += f"• Time Horizon: {recommendation.get('time_horizon', 'N/A')}\n"
        
        return reasoning

