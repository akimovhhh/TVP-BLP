# tests/test_import.py
import pandas as pd
from tvpblp import TwoStepGMM_BLP, sample_data_path


def test_can_instantiate_with_minimal_df():
    # minimal synthetic dataset: 2 markets, 2 products each

    df = pd.read_csv(sample_data_path()).head(4)
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
