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
    
    def analyze(self, stock_data: dict, financials_data: dict, history_data: dict) -> dict:
        """
        Perform fundamental analysis on stock data
        Returns buy/sell recommendation with reasoning
        """
        try:
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
                )
            }
        
        except Exception as e:
            logger.error(f"Error in investing analysis: {e}")
            return {"error": str(e)}
    
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
        """Analyze financial health from financial statements"""
        try:
            financials = financials_data.get('financials', {})
            balance_sheet = financials_data.get('balance_sheet', {})
            cashflow = financials_data.get('cashflow', {})
            
            # Extract key metrics (simplified - in production, parse properly)
            health_score = 50  # Base score
            factors = []
            
            # Check for revenue growth (if available)
            # This is simplified - real implementation would parse financial statements properly
            
            return {
                "health_score": health_score,
                "factors": factors,
                "has_positive_cashflow": True,  # Would check actual cashflow
                "debt_level": "moderate"  # Would calculate from balance sheet
            }
        except Exception as e:
            logger.error(f"Error analyzing financial health: {e}")
            return {
                "health_score": 50,
                "factors": ["Limited financial data available"],
                "has_positive_cashflow": None,
                "debt_level": "unknown"
            }
    
    def _analyze_valuation(self, stock_data: dict, financials_data: dict) -> dict:
        """Analyze valuation metrics"""
        info = stock_data.get('info', {})
        current_price = stock_data.get('price', 0)
        pe_ratio = info.get('pe_ratio')
        dividend_yield = info.get('dividend_yield')
        
        valuation_score = 50
        factors = []
        
        # P/E Analysis
        if pe_ratio:
            if pe_ratio < 15:
                valuation_score += 20
                factors.append("Low P/E ratio suggests undervaluation")
            elif pe_ratio > 25:
                valuation_score -= 20
                factors.append("High P/E ratio suggests overvaluation")
            else:
                factors.append("P/E ratio is reasonable")
        
        # Dividend Yield
        if dividend_yield:
            if dividend_yield > 0.03:  # 3%
                valuation_score += 10
                factors.append("Attractive dividend yield")
        
        return {
            "pe_ratio": pe_ratio,
            "dividend_yield": dividend_yield,
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
        
        # Financial health
        health_score = financial_health.get('health_score', 50)
        if health_score > 60:
            score += 2
            reasons.append("Strong financial health")
        elif health_score < 40:
            score -= 2
            reasons.append("Weak financial health")
        
        # Valuation
        valuation_score = valuation.get('valuation_score', 50)
        if valuation.get('is_undervalued'):
            score += 2
            reasons.append("Stock appears undervalued")
        elif valuation.get('is_overvalued'):
            score -= 2
            reasons.append("Stock appears overvalued")
        
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
        
        # Determine recommendation
        if score >= 3:
            action = "BUY"
            confidence = min(90, 60 + (score * 5))
        elif score <= -3:
            action = "AVOID"
            confidence = min(90, 60 + (abs(score) * 5))
        else:
            action = "HOLD"
            confidence = 50
        
        current_price = stock_data.get('price', 0)
        
        # Long-term targets (for investing)
        if action == "BUY":
            # Conservative 1-year target (10-20% growth)
            target_price = current_price * 1.15
            # Conservative stop loss (20% down)
            stop_loss = current_price * 0.80
        else:
            target_price = current_price
            stop_loss = current_price
        
        return {
            "action": action,
            "confidence": confidence,
            "entry_price": float(current_price),
            "target_price": float(target_price),
            "stop_loss": float(stop_loss),
            "time_horizon": "1-3 years",
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
        div_yield = valuation.get('dividend_yield')
        if div_yield:
            reasoning += f"• Dividend Yield: {div_yield * 100:.2f}%\n"
        
        reasoning += f"\nGrowth Analysis:\n"
        annual_return = growth.get('annualized_return', 0)
        reasoning += f"• Historical Annualized Return: {annual_return:.2f}%\n"
        
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

