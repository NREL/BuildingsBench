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
        heading_level: 4
        show_root_heading: true

---

## BaseModel

::: buildings_bench.models.base_model.BaseModel
    options:
        show_source: false
        heading_level: 4
        show_root_heading: true

---

## LoadForecastingTransformer

::: buildings_bench.models.transformers.LoadForecastingTransformer
    options:
        show_source: false
        heading_level: 4
        show_root_heading: true
        

## TokenEmbedding

::: buildings_bench.models.transformers.TokenEmbedding
    options:
        show_source: false
        heading_level: 4
        show_root_heading: true
        
## PositionalEncoding

::: buildings_bench.models.transformers.PositionalEncoding
    options:
        show_source: false
        heading_level: 4
        show_root_heading: true
        
## TimeSeriesSinusoidalPeriodicEmbedding

::: buildings_bench.models.transformers.TimeSeriesSinusoidalPeriodicEmbedding
    options:
        show_source: false
        heading_level: 4
        show_root_heading: true
        
## ZeroEmbedding

::: buildings_bench.models.transformers.ZeroEmbedding
    options:
        show_source: false
        heading_level: 4
        show_root_heading: true      
        
---

## Persistence Ensemble

::: buildings_bench.models.persistence.AveragePersistence
    options:
        show_source: false
        heading_level: 4
        show_root_heading: true
        

## Previous Day Persistence

::: buildings_bench.models.persistence.CopyLastDayPersistence
    options:
        show_source: false
        heading_level: 4
        show_root_heading: true
        

## Previous Week Persistence

::: buildings_bench.models.persistence.CopyLastWeekPersistence
    options:
        show_source: false
        heading_level: 4
        show_root_heading: true        

---

## Linear Regression 

::: buildings_bench.models.linear_regression.LinearRegression
    options:
        show_source: false
        heading_level: 4
        show_root_heading: true        

---

## DLinear

::: buildings_bench.models.dlinear_regression.DLinearRegression
    options:
        show_source: false
        heading_level: 4
        show_root_heading: true
        
