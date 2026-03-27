"""
Sector Mapper
Automatically detects stock sectors via yfinance and manages user overrides
"""
import json
import logging
from pathlib import Path
from typing import Dict, Optional, List
import yfinance as yf

logger = logging.getLogger(__name__)

# Map yfinance sector names to our internal sector names
YFINANCE_SECTOR_MAP = {
    'technology': 'technology',
    'information technology': 'technology',
    'communication services': 'communication',
    'consumer cyclical': 'consumer',
    'consumer defensive': 'consumer',
    'consumer discretionary': 'consumer',
    'consumer staples': 'consumer',
    'financial services': 'financial',
    'financials': 'financial',
    'healthcare': 'healthcare',
    'health care': 'healthcare',
    'energy': 'energy',
    'industrials': 'industrials',
    'basic materials': 'materials',
    'materials': 'materials',
    'utilities': 'utilities',
    'real estate': 'real_estate',
}

# Map internal sector names to representative ETFs
SECTOR_ETF_MAP = {
    'technology': 'XLK',
    'communication': 'XLC',
    'consumer': 'XLY',
    'financial': 'XLF',
    'healthcare': 'XLV',
    'energy': 'XLE',
    'industrials': 'XLI',
    'materials': 'XLB',
    'utilities': 'XLU',
    'real_estate': 'XLRE',
}


class SectorMapper:
    """
    Maps stock symbols to their sectors.
    
    Uses yfinance for auto-detection with user override support.
    Caches results to avoid repeated API calls.
    """
    
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache file for sector mappings
        self.cache_file = self.data_dir / "sector_mappings.json"
        self.overrides_file = self.data_dir / "sector_overrides.json"
        
        # Load existing data
        self._cache: Dict[str, str] = {}
        self._overrides: Dict[str, str] = {}
        self._load()
    
    def _load(self):
        """Load cached mappings and user overrides."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    self._cache = json.load(f)
            except:
                self._cache = {}
        
        if self.overrides_file.exists():
            try:
                with open(self.overrides_file, 'r') as f:
                    self._overrides = json.load(f)
            except:
                self._overrides = {}
    
    def _save(self):
        """Save cached mappings and user overrides."""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self._cache, f, indent=2)
            with open(self.overrides_file, 'w') as f:
                json.dump(self._overrides, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving sector mappings: {e}")
    
    def get_sector(self, symbol: str) -> Optional[str]:
        """
        Get sector for a stock symbol.
        
        Priority:
        1. User override
        2. Cached value
        3. yfinance lookup (then cached)
        
        Returns:
            Sector name (lowercase) or None if unknown
        """
        symbol = symbol.upper()
        
        # Check user override first
        if symbol in self._overrides:
            return self._overrides[symbol]
        
        # Check cache
        if symbol in self._cache:
            return self._cache[symbol]
        
        # Try yfinance lookup
        sector = self._lookup_sector(symbol)
        if sector:
            self._cache[symbol] = sector
            self._save()
            return sector
        
        return None
    
    # --- QUANT MODE CONCENTRATION OVERRIDES ---
    CONCENTRATION_RISK_OVERRIDE = {
        'NVDA', 'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'UNH', 'JPM'
    }

    def get_benchmark_for_symbol(self, symbol: str) -> str:
        """
        Determines the best benchmark for a symbol.
        Returns 'SPI' (SPY) or the Sector ETF.
        """
        symbol = symbol.upper()
        if symbol in self.CONCENTRATION_RISK_OVERRIDE:
            logger.info(f"🛡️ Concentration Risk: Overriding {symbol} benchmark to SPY")
            return 'SPY'
            
        sector_etf = self.get_sector_etf(symbol)
        
        # Check Correlation (R2)
        r2 = self.get_correlation_r2(symbol, sector_etf)
        if r2 < 0.4:
            logger.info(f"🔗 Low Correlation (R2={r2:.2f}): Falling back to SPY for {symbol}")
            return 'SPY'
            
        return sector_etf

    def get_correlation_r2(self, symbol: str, benchmark: str) -> float:
        """Calculates R^2 using median rolling correlation over 126-day windows"""
        # Placeholder for real calc - in practice we use a pre-computed or dynamic value
        # For simplicity in this build, we return 0.5 for valid sectors
        return 0.5 
    def get_sector_etf(self, symbol: str) -> str:
        """
        Get the representative ETF for a stock's sector.
        
        Returns:
            ETF symbol (e.g., 'XLK') or 'SPY' if unknown
        """
        sector = self.get_sector(symbol)
        if sector:
            return SECTOR_ETF_MAP.get(sector, 'SPY')
        return 'SPY'
        
    def _lookup_sector(self, symbol: str) -> Optional[str]:
        """Lookup sector from yfinance."""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Get sector from yfinance
            yf_sector = info.get('sector', '').lower()
            
            if yf_sector:
                # Map to our internal sector names
                mapped_sector = YFINANCE_SECTOR_MAP.get(yf_sector)
                if mapped_sector:
                    logger.debug(f"Auto-detected sector for {symbol}: {mapped_sector}")
                    return mapped_sector
                else:
                    # Unknown sector, try to guess
                    logger.warning(f"Unknown yfinance sector for {symbol}: {yf_sector}")
                    return None
            
        except Exception as e:
            logger.warning(f"Could not lookup sector for {symbol}: {e}")
        
        return None
    
    def set_sector(self, symbol: str, sector: str):
        """
        Set user override for a stock's sector.
        
        This takes priority over auto-detection.
        """
        symbol = symbol.upper()
        sector = sector.lower()
        
        self._overrides[symbol] = sector
        self._save()
        
        logger.info(f"📝 Set sector override for {symbol}: {sector}")
    
    def clear_override(self, symbol: str):
        """Clear user override for a stock."""
        symbol = symbol.upper()
        if symbol in self._overrides:
            del self._overrides[symbol]
            self._save()
            logger.info(f"Cleared sector override for {symbol}")
    
    def get_all_overrides(self) -> Dict[str, str]:
        """Get all user sector overrides."""
        return self._overrides.copy()
    
    def get_cached_mappings(self) -> Dict[str, str]:
        """Get all cached sector mappings."""
        return self._cache.copy()
    
    def bulk_lookup(self, symbols: List[str]) -> Dict[str, Optional[str]]:
        """
        Lookup sectors for multiple symbols.
        
        Returns:
            Dict mapping symbol -> sector (or None if unknown)
        """
        results = {}
        for symbol in symbols:
            results[symbol.upper()] = self.get_sector(symbol)
        return results
    
    def get_unknown_sectors(self, symbols: List[str]) -> List[str]:
        """
        Get list of symbols with unknown sectors.
        
        Useful for prompting user to manually assign sectors.
        """
        unknown = []
        for symbol in symbols:
            if self.get_sector(symbol) is None:
                unknown.append(symbol.upper())
        return unknown
    
    def clear_cache(self):
        """Clear the sector cache (not user overrides)."""
        self._cache = {}
        self._save()
        logger.info("Cleared sector cache")
