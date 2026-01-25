from data_fetcher import StockDataFetcher
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_tickers():
    fetcher = StockDataFetcher()
    
    # Test 1: International Ticker (SIE.DE) - Expected to fail currently
    logger.info("--- Testing SIE.DE ---")
    data_sie = fetcher.fetch_stock_data("SIE.DE")
    if data_sie and 'error' not in data_sie:
        logger.info(f"SUCCESS: SIE.DE price: {data_sie.get('price')}")
    else:
        logger.error(f"FAILURE: SIE.DE data: {data_sie}")

    # Test 2: Dot Ticker that needs replacement (BRK.B) - Expected to succeed (converted to BRK-B)
    logger.info("\n--- Testing BRK.B ---")
    data_brk = fetcher.fetch_stock_data("BRK.B")
    if data_brk and 'error' not in data_brk:
        logger.info(f"SUCCESS: BRK.B price: {data_brk.get('price')}")
    else:
        logger.error(f"FAILURE: BRK.B data: {data_brk}")

    # Test 3: Normal Ticker (AAPL)
    logger.info("\n--- Testing AAPL ---")
    data_aapl = fetcher.fetch_stock_data("AAPL")
    if data_aapl and 'error' not in data_aapl:
        logger.info(f"SUCCESS: AAPL price: {data_aapl.get('price')}")
    else:
        logger.error(f"FAILURE: AAPL data: {data_aapl}")

if __name__ == "__main__":
    test_tickers()
