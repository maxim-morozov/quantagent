import arcticdb as adb
from arcticdb.version_store.library import SymbolDescription, DataError
from arcticdb import LibraryOptions
import pandas as pd
import logging


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


arctic = None
library = None
quant_data = None


def get_symbols_for_transformation():
    description = library.get_description("tick_data")
    symbols = [column.name for column in description.columns if column.name != 'symbol']
    return symbols, description.date_range


def prep_delta_instruction_for_each_symbol(symbols, date_range):
    transform_instructions = []
    batch_descriptions = library.get_description_batch(symbols)
    for index, description in enumerate(batch_descriptions):
        match description:
            case SymbolDescription(date_range=symbol_data_range):
                transform_instructions.append(
                    (symbols[index], symbol_data_range[1], date_range[1])
                )
            case DataError():
                transform_instructions.append(
                    (symbols[index], date_range)
                )
    return transform_instructions


def transform_symbol_data(symbol, start_date, end_date):
    logger.info(f"Transforming data for {symbol} from {start_date} to {end_date}")
    date_range = pd.date_range(
        start=start_date, end=end_date, freq='1min', inclusive='right'
    )
    start = start_date
    for date in date_range:
        logger.info(f"Processing data between {start} to {date} for {symbol}")
        df = library.read("tick_data",
                          columns=["symbol", symbol],
                          date_range=(start, date)).data
        data = {}
        for index, row in df.iterrows():
            if index == start:
                continue

            ticker = str(row['symbol'].decode('utf-8'))
            if ticker not in data:
                data[ticker] = {"index": [], "value": []}
            data[ticker]["index"].append(index)
            data[ticker]["value"].append(row[symbol])

        df = pd.DataFrame()
        for ticker, values in data.items():
            temp_df = pd.DataFrame(values["value"],
                                   index=values["index"],
                                   columns=[str(ticker)])
            df = df.join(temp_df, how='outer')

        quant_data.stage(symbol, df)
        start = date
    logger.info(f"Finalizing staged data for {symbol}")
    quant_data.finalize_staged_data(symbol)


if __name__ == "__main__":
    # Initialize ArcticDB library
    arctic = adb.Arctic('lmdb://../data/arctic')
    library = arctic.get_library('market_data')
    quant_data = arctic.get_library('quant_data',
                                    create_if_missing=True,
                                    library_options=LibraryOptions(dynamic_schema=True))

    symbols_for_transformation = get_symbols_for_transformation()
    delta_instructions = prep_delta_instruction_for_each_symbol(*symbols_for_transformation)

    for symbol, (start_date, end_date) in delta_instructions:
        transform_symbol_data(symbol, start_date, end_date)
