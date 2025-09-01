import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
import arcticdb as adb
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Initialize ArcticDB library
arctic = adb.Arctic('lmdb://../data/arctic')
library = arctic.get_library('market_data', create_if_missing=True)


# Function to generate synthetic market data with nanosecond precision
def generate_market_data(start_date, end_date, symbols):
    trading_start = time(9, 30)
    trading_end = time(16, 0)
    date_range = pd.date_range(
        start=start_date, end=end_date, freq='S'  # Second-level frequency
    )

    for date in date_range:
        data = []
        index = []
        if trading_start <= date.time() <= trading_end:
            # Random set of symbols for each second to reduce data volume
            num_symbols = int(np.random.uniform(500, len(symbols)))
            sampled_symbols = np.random.choice(
                symbols, size=num_symbols, replace=False
            )
            np.random.shuffle(sampled_symbols)

            for symbol in sampled_symbols:
                logger.info(
                    f"Generating data for {symbol} at {date}"
                )

                open_price = np.random.uniform(100, 200)
                close_price = open_price + np.random.uniform(-10, 10)
                high_price = max(open_price, close_price) + \
                    np.random.uniform(0, 5)
                low_price = min(open_price, close_price) - \
                    np.random.uniform(0, 5)
                last_price = close_price + np.random.uniform(-2, 2)
                volume = np.random.randint(1000, 10000)

                data.append({
                    'symbol': symbol,
                    'open': open_price,
                    'close': close_price,
                    'high': high_price,
                    'low': low_price,
                    'last': last_price,
                    'volume': volume
                })
                index.append(date)

            index.sort()
            df = pd.DataFrame(data, index=index, columns={
                'symbol': str,
                'open': float,
                'close': float,
                'high': float,
                'low': float,
                'last': float,
                'volume': int
            })
            df['symbol'] = df['symbol'].astype('|S')
            try:
                library.append("tick_data", df)
            except Exception as e:
                print(df)
                raise

# Generate market data
# Placeholder for all NYSE symbols. Replace with actual NYSE symbol list.
symbols_df = pd.read_csv('../data/nyse-listed.csv')
symbols = symbols_df['ACT Symbol'].tolist()
start_date = datetime.now() - timedelta(days=3)  # Simulate one day of data
end_date = datetime.now()
generate_market_data(start_date, end_date, symbols)

print(
    "Market data for all NYSE stocks has been generated and saved to ArcticDB."
)

print(library.head("tick_data").data)
