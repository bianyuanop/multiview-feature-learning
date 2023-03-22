import numpy as np

filters = {
    "raw": {
        "X": None  
    }, 
    "bandpass": {
        "X": None,
        "lowcut": 0.015,
        "highcut": 0.1,
        "interval": 2,
        "order": 5,
    }
}

preprocessers = {
    "pca": {
        "X": None,
        "n_components": 50,
    },
    "lasso": {
        "X": None,
        "y": None,
    }
}