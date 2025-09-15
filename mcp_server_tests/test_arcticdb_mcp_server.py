import pytest
from arcticdb import Arctic
import pandas as pd
import os

import sys
sys.path.append("../mcp-server")

from mcp_server.arcticdb_mcp_server import list_datasets

# Setup LMDB for testing
TEST_DB_PATH = "./test_lmdb"
os.environ["ARCTICDB_URI"] = f"lmdb://{TEST_DB_PATH}"


@pytest.fixture(scope="module")
def setup_lmdb():
    if os.path.exists(TEST_DB_PATH):
        os.system(f"rm -rf {TEST_DB_PATH}")
    store = Arctic(f"lmdb://{TEST_DB_PATH}")
    library = store.get_library("test_lib", create_if_missing=True)

    # Populate with sample data
    df1 = pd.DataFrame({
        "col1": [1, 2, 3],
        "col2": [4, 5, 6]
    }, index=pd.date_range(start="2023-01-01", periods=3))
    library.write("symbol1", df1)

    df2 = pd.DataFrame({
        "colA": [10, 20, 30],
        "colB": [40, 50, 60]
    }, index=pd.date_range(start="2023-01-02", periods=3))
    library.write("symbol2", df2)

    yield store

    # Cleanup
    store.delete_library("test_lib")



def test_list_datasets(setup_lmdb):
    datasets = list_datasets()
    assert "test_lib" in datasets


# def test_list_symbols(setup_lmdb):
#     symbols = list_symbols("test_lib")
#     assert "symbol1" in symbols
#     assert "symbol2" in symbols


# def test_list_symbol_info(setup_lmdb):
#     description = list_symbol_info("test_lib", "symbol1")
#     assert description.name == "symbol1"


# def test_query_symbol_basic(setup_lmdb):
#     args = {
#         "library": "test_lib",
#         "symbol": "symbol1",
#         "start": "2023-01-01",
#         "end": "2023-01-02",
#         "columns": ["col1"],
#         "resample_freq": None,
#         "resample_method": "last",
#         "rows_start": None,
#         "rows_end": None
#     }

#     result = query_symbol(args)
#     assert result["symbol"] == "symbol1"
#     assert result["columns"] == ["col1"]
#     assert len(result["rows"]) == 2


# def test_query_symbol_resample(setup_lmdb):
#     args = {
#         "library": "test_lib",
#         "symbol": "symbol1",
#         "start": "2023-01-01",
#         "end": "2023-01-03",
#         "columns": None,
#         "resample_freq": "D",
#         "resample_method": "mean",
#         "rows_start": None,
#         "rows_end": None
#     }

#     result = query_symbol(args)
#     assert result["symbol"] == "symbol1"
#     assert len(result["rows"]) == 3


# def test_query_symbol_row_range(setup_lmdb):
#     args = {
#         "library": "test_lib",
#         "symbol": "symbol2",
#         "start": None,
#         "end": None,
#         "columns": None,
#         "resample_freq": None,
#         "resample_method": "last",
#         "rows_start": 0,
#         "rows_end": 2
#     }

#     result = query_symbol(args)
#     assert result["symbol"] == "symbol2"
#     assert len(result["rows"]) == 2


# def test_query_symbol_multiple_columns(setup_lmdb):
#     args = {
#         "library": "test_lib",
#         "symbol": "symbol2",
#         "start": "2023-01-02",
#         "end": "2023-01-04",
#         "columns": ["colA", "colB"],
#         "resample_freq": None,
#         "resample_method": "last",
#         "rows_start": None,
#         "rows_end": None
#     }

#     result = query_symbol(args)
#     assert result["symbol"] == "symbol2"
#     assert result["columns"] == ["colA", "colB"]
#     assert len(result["rows"]) == 3
