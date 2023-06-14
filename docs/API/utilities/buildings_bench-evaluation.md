# buildings_bench.evaluation

The `buildings_bench.evaluation` module contains the main functionality for evaluting a model
on the benchmark tasks.

The `buildings_bench.evaluation.managers.DatasetMetricsManager` class is the main entry point.

## Simple usage

```python
from buildings_bench import BuildingTypes
from buildings_bench.evaluation.managers import DatasetMetricsManager

# By default, the DatasetMetricsManager keeps track of NRMSE, NMAE, and NMBE
metrics_manager = DatasetMetricsManager()

# Iterate over the dataset using our building dataset generator
for building_name, building_dataset in buildings_datasets_generator:

    # Register a new building with the manager
    metrics_manager.add_building_to_dataset_if_missing(
        dataset_name, building_name,
    )

    # Your model makes predictions
    # ...

    # Register the predictions with the manager
    metrics_manager(
        dataset_name,           	  # the name of the dataset, e.g., electricity
        building_name,          	  # the name of the building, e.g., MT_001
        continuous_targets,      	  # the ground truth 24 hour targets
        predictions,           		  # the model's 24 hour predictions
        BuildingTypes.RESIDENTIAL_INT,    # an int indicating the building type
    )
```

## Advanced usage (with scoring rule)

```python
from buildings_bench.evaluation.managers import DatasetMetricsManager
from buildings_bench.evaluation import scoring_rule_factory

metrics_manager = DatasetMetricsManager(scoring_rule = scoring_rule_factory('crps'))

# Iterate over the dataset
for building_name, building_dataset in buildings_datasets_generator:

    # Register a new building with the manager
    metrics_manager.add_building_to_dataset_if_missing(
        dataset_name, building_name,
    )

    # Your model makes predictions
    # ...

    # Register the predictions with the manager
    metrics_manager(
        dataset_name,           # the name of the dataset, e.g., electricity
        building_name,          # the name of the building, e.g., MT_001
        continuous_targets,     # the ground truth 24 hour targets
        predictions,            # the model's 24 hour predictions
        building_types_mask,    # a boolean tensor indicating building type
        y_categories=targets,   # for scoring rules, the ground truth (discrete categories if using tokenization)
        y_distribution_params=distribution_params, # for scoring rules, the distribution parameters
        centroids=centroids   # for scoring rules with categorical variables, the centroid values
    )
```

---

::: buildings_bench.evaluation
    options:
        show_source: false
        heading_level: 2
        show_root_heading: true

---

::: buildings_bench.evaluation.managers
    options:
        show_source: false
        heading_level: 2
        show_root_heading: true

---

::: buildings_bench.evaluation.metrics
    options:
        show_source: false
        heading_level: 2
        show_root_heading: true

---

::: buildings_bench.evaluation.scoring_rules
    options:
        show_source: false
        heading_level: 2
        show_root_heading: true

---

::: buildings_bench.evaluation.aggregate
    options:
        show_source: false
        heading_level: 2
        show_root_heading: true

