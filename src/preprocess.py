import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer

preprocess_pipeline = ColumnTransformer(
    transformers=[
        ("sqrt_transformer", FunctionTransformer(np.sqrt), ["windspeed"])
    ],
    remainder="passthrough",
    verbose_feature_names_out=False,
)
preprocess_pipeline.set_output(transform="pandas")
