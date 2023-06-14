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

::: buildings_bench.models
    options:
        show_source: false
        heading_level: 2
        show_root_heading: true

---

::: buildings_bench.models.base_model
    options:
        show_source: false
        heading_level: 2
        show_root_heading: true

---

::: buildings_bench.models.transformers
    options:
        show_source: false
        heading_level: 2
        show_root_heading: true

---

::: buildings_bench.models.persistence
    options:
        show_source: false
        heading_level: 2
        show_root_heading: true

---

::: buildings_bench.models.linear_regression
    options:
        show_source: false
        heading_level: 2
        show_root_heading: true

---

::: buildings_bench.models.dlinear_regression
    options:
        show_source: false
        heading_level: 2
        show_root_heading: true
