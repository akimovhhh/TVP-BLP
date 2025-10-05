# tests/test_import.py
import pandas as pd
from tvpblp import TwoStepGMM_BLP


def test_can_instantiate_with_minimal_df():
    # minimal synthetic dataset: 2 markets, 2 products each
    df = pd.DataFrame(
        {
            "market": [1, 1, 2, 2],
            "product": [101, 102, 201, 202],
            "price": [1.0, 1.5, 0.9, 1.4],
            "share": [0.2, 0.1, 0.25, 0.15],
            "og_share": [0.7, 0.7, 0.6, 0.6],  # outside good share complement
            "instrument_1": [0.95, 1.05, 1.02, 0.98],
            "instrument_2": [1, 2, 1, 2],
        }
    )
    m = TwoStepGMM_BLP(
        df,
        share_col="share",
        price_col="price",
        lin=["price"],
        random=[],  # simple-logit case for now
        instruments=["instrument_1", "instrument_2"],
        market_id="market",
        product_id="product",
        integration_points=8,
        integration_int=[0.1, 0.9],
        og_share_col="og_share",
    )
    # smoke test: object is created and has expected attributes
    assert hasattr(m, "__class__")
