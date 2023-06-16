# buildings_bench.models

Available models:

- Encoder-decoder time series transformer
- Persistence Ensemble (`AveragePersistence`)
- Previous Day Persistence (`CopyLastDayPersistence`)
- Previous Week Persistence (`CopyLastWeekPersistence`)
- Linear regression
- DLinear

Main entry point for loading a BuildingsBench model is `model_factory()`.

---

## model_factory

::: buildings_bench.models.model_factory
    options:
        show_source: false
        heading_level: 3
        show_root_heading: true

---

## BaseModel

::: buildings_bench.models.base_model.BaseModel
    options:
        show_source: false
        heading_level: 3
        

---

## Time Series Transformer

::: buildings_bench.models.transformers.LoadForecastingTransformer
    options:
        show_source: false
        heading_level: 3
        

## TokenEmbedding

::: buildings_bench.models.transformers.TokenEmbedding
    options:
        show_source: false
        heading_level: 3
        show_root_heading: true
        
## PositionalEncoding

::: buildings_bench.models.transformers.PositionalEncoding
    options:
        show_source: false
        heading_level: 3
        
## TimeSeriesSinusoidalPeriodicEmbedding

::: buildings_bench.models.transformers.TimeSeriesSinusoidalPeriodicEmbedding
    options:
        show_source: false
        heading_level: 3
        
## ZeroEmbedding

::: buildings_bench.models.transformers.ZeroEmbedding
    options:
        show_source: false
        heading_level: 3
        
        
---

## Persistence Ensemble

::: buildings_bench.models.persistence.AveragePersistence
    options:
        show_source: false
        heading_level: 3
        

## Previous Day Persistence

::: buildings_bench.models.persistence.CopyLastDayPersistence
    options:
        show_source: false
        heading_level: 3
        

## Previous Week Persistence

::: buildings_bench.models.persistence.CopyLastWeekPersistence
    options:
        show_source: false
        heading_level: 3
        

---

## Linear Regression 

::: buildings_bench.models.linear_regression.LinearRegression
    options:
        show_source: false
        heading_level: 3
        

---

## DLinear

::: buildings_bench.models.dlinear_regression.DLinearRegression
    options:
        show_source: false
        heading_level: 3
        
