import asyncio
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict
from mcp.server.fastmcp import FastMCP
import arcticdb as adb
from arcticdb.version_store.library import SymbolDescription
import pandas as pd
import os

# from utils import df_to_base64_png, safe_json_loads

CONFIG = {
    "uri": os.getenv("ARCTICDB_URI", "lmdb://./data"),
}

app = FastMCP("arcticdb-mcp")

# -----------------------
# Pydantic Schemas
# -----------------------


class QueryArgs(BaseModel):
    library: str
    symbol: str
    start: Optional[str] = None
    end: Optional[str] = None
    columns: Optional[List[str]] = None
    resample_freq: Optional[str] = None
    resample_method: str = "last"
    rows_start: Optional[int] = None
    rows_end: Optional[int] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ComputeFactorArgs(BaseModel):
    symbols: List[str]
    factor: str = "momentum"
    window: int = 60
    price_col: str = "close"

    model_config = ConfigDict(arbitrary_types_allowed=True)


class AggregateArgs(BaseModel):
    symbols: List[str]
    func: str = "mean"
    group_by: Optional[str] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class PlotArgs(BaseModel):
    symbol: str
    columns: Optional[List[str]] = None
    title: Optional[str] = None
    tail: int = 500

    model_config = ConfigDict(arbitrary_types_allowed=True)


class StoreNoteArgs(BaseModel):
    topic: str
    text: str
    tags: Optional[List[str]] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

# -----------------------
# Tools
# -----------------------


@app.tool()
def list_datasets() -> List[str]:
    """
    List all libraries (datasets) in the ArcticDB store.
    Returns:
        List[str]: A list of library names.
    """
    store = adb.Arctic(CONFIG["uri"])
    return store.list_libraries()


# @app.tool()
# def list_symbols(library: str) -> List[str]:
#     """
#     List all symbols in a given library.
#     Args:
#         library (str): The name of the library.
#     Returns:
#         List[str]: A list of symbol names."""
#     lib = adb.Arctic(CONFIG["uri"]).get_library(library)
#     return lib.list_symbols()


# @app.tool()
# def list_symbol_info(library: str, symbol: str) -> SymbolDescription:
#     """
#     Get metadata information about a specific symbol in a library.
#     Args:
#         library (str): The name of the library.
#         symbol (str): The symbol name.
#     Returns:
#         SymbolDescription: Metadata information about the symbol.
#     """
#     description = adb.Arctic(CONFIG["uri"]).\
#         get_library(library).\
#         get_description(symbol)
#     return description


# @app.tool()
# def get_library_info(lib) -> List[SymbolDescription]:
#     """
#     Get metadata information about all symbols in a library.
#     Args:
#         lib: The ArcticDB library object.
#     Returns:
#         List[SymbolDescription]: A list of metadata information for all symbols.
#     """
#     lib = adb.Arctic(CONFIG["uri"]).get_library(lib)
#     syms = lib.list_symbols()
#     description = lib.get_description_batch(syms)
#     return description


# @app.tool()
# def query_symbol(args: QueryArgs) -> Dict[str, any]:
#     """
#     Query a symbol with various options like date range, columns, resampling, and row slicing.
#     Args:
#         args (QueryArgs): The query parameters.
#     Returns:
#         Dict[str, Any]: The query result including symbol, columns, index, rows, and number of rows.
#     """
#     query = adb.QueryBuilder()

#     date_range = None
#     if args.start or args.end:
#         date_range = (pd.Timestamp(args.start),
#                       pd.Timestamp(args.end))
#     elif args.start:
#         date_range = (pd.Timestamp(args.start),
#                       None)
#     else:
#         date_range = (None,
#                       pd.Timestamp(args.end))

#     if args.resample_freq and args.resample_method:
#         query = query.resample(args.resample_freq).agg(args.resample_method)

#     row_range = (
#         (args.rows_start, args.rows_end)
#         if args.rows_start is not None or args.rows_end is not None
#         else None
#     )

#     lib = adb.Arctic(CONFIG["uri"]).get_library(args.library)
#     df = lib.read(args.symbol,
#                   date_range=date_range,
#                   query_builder=query,
#                   columns=args.columns,
#                   row_range=row_range).data

#     return {
#         "symbol": args.symbol,
#         "columns": list(df.columns),
#         "index": [str(i) for i in df.index],
#         "rows": df.reset_index().to_dict(orient="records"),
#         "nrows": len(df),
#     }


# @app.tool()
# def compute_factor(args: ComputeFactorArgs):
#     lib = get_store()
#     results = {}
#     for s in args.symbols:
#         try:
#             df = lib.read(s).data
#         except Exception as e:
#             results[s] = {"error": str(e)}
#             continue
#         price = df[args.price_col].astype(float)
#         if len(price) < args.window:
#             results[s] = {"error": "insufficient data"}
#             continue
#         if args.factor == "momentum":
#             val = price / price.shift(args.window) - 1
#         elif args.factor == "volatility":
#             val = price.pct_change().rolling(args.window).std()
#         elif args.factor == "SMA":
#             val = price.rolling(args.window).mean()
#         elif args.factor == "EMA":
#             val = price.ewm(span=args.window, adjust=False).mean()
#         elif args.factor == "sharpe":
#             rets = price.pct_change().dropna()
#             mean = rets.rolling(args.window).mean()
#             std = rets.rolling(args.window).std()
#             val = mean / std * (252 ** 0.5)
#         else:
#             results[s] = {"error": f"unknown factor {args.factor}"}
#             continue
#         trail = val.dropna().tail(100)
#         results[s] = {
#             "last": float(trail.iloc[-1]) if len(trail) else None,
#             "tail": [float(x) for x in trail.values.tolist()],
#             "tail_index": [str(i) for i in trail.index.tolist()],
#         }
#     return {"factor": args.factor, "window": args.window, "results": results}


# @app.tool()
# def aggregate(args: AggregateArgs):
#     lib = get_store()
#     joined = None
#     for s in args.symbols:
#         try:
#             df = lib.read(s).data
#         except Exception:
#             continue
#         df2 = df.copy()
#         df2.columns = [f"{s}::{c}" for c in df2.columns]
#         if joined is None:
#             joined = df2
#         else:
#             joined = joined.join(df2, how="outer")
#     if joined is None:
#         return {"error": "no data"}
#     # simple column-wise aggregation
#     if args.func == "mean":
#         res = joined.mean(axis=1)
#     elif args.func == "sum":
#         res = joined.sum(axis=1)
#     else:
#         return {"error": f"unknown func {args.func}"}
#     return {
#         "index": [str(i) for i in res.index],
#         "values": [float(x) for x in res.values.tolist()]
#     }


# @app.tool()
# def plot_timeseries(args: PlotArgs):
#     lib = get_store()
#     df = lib.read(args.symbol).data
#     if args.columns:
#         df = df[args.columns]
#     tail = args.tail
#     plot_df = df.tail(tail)
#     b64 = df_to_base64_png(plot_df, title=args.title or args.symbol)
#     return {"image_base64": b64, "nrows": len(df)}


# @app.tool()
# def store_research_note(args: StoreNoteArgs):
#     lib = get_store()
#     ts = pd.Timestamp.utcnow()
#     row = {"ts": ts, "text": args.text, "tags": ",".join(
#         args.tags) if args.tags else None}
#     df = pd.DataFrame([row]).set_index("ts")
#     symbol = f"notes/{args.topic}"
#     lib.write(symbol, df, prune_previous=False)
#     return {"ok": True, "symbol": symbol, "ts": str(ts)}


# @app.tool()
# def get_research_log(topic: Optional[str] = None, limit: int = 50):
#     lib = get_store()
#     syms = [s for s in lib.list_symbols() if s.startswith("notes/")]
#     if topic:
#         syms = [s for s in syms if s == f"notes/{topic}"]
#     notes = []
#     for s in syms:
#         df = lib.read(s).data
#         df = df.sort_index(ascending=False).head(limit)
#         for ts, row in df.iterrows():
#             notes.append({"symbol": s, "ts": str(ts), "text": row.get(
#                 "text"), "tags": row.get("tags")})
#     notes = sorted(notes, key=lambda x: x["ts"], reverse=True)[:limit]
#     return {"notes": notes}


try:
    import uvloop
except ImportError:
    uvloop = None

if __name__ == "__main__":
    uvloop.install()
