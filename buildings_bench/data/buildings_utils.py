# utility functions for building characteristics loading
# Author: Zhaonan Li zli4@nrel.gov

res_chars = [
    "in.bedrooms",
    "in.cec_climate_zone",
    "in.ceiling_fan",
    "in.census_division",
    "in.census_division_recs",
    "in.census_region",
    "in.clothes_dryer",
    "in.clothes_washer",
    "in.clothes_washer_presence",
    "in.cooking_range",
    "in.cooling_setpoint",
    "in.cooling_setpoint_offset_magnitude",
    "in.dishwasher",
    "in.ducts",
    "in.geometry_floor_area",
    "in.geometry_floor_area_bin",
    "in.geometry_foundation_type",
    "in.geometry_garage",
    "in.geometry_stories",
    "in.geometry_stories_low_rise",
    "in.geometry_wall_exterior_finish",
    "in.geometry_wall_type",
    "in.geometry_wall_type_and_exterior_finish",
    "in.has_pv",
    "in.heating_fuel",
    "in.heating_setpoint",
    "in.heating_setpoint_has_offset",
    "in.heating_setpoint_offset_magnitude",
    "in.hvac_cooling_efficiency",
    "in.hvac_heating_efficiency",
    "in.hvac_shared_efficiencies",
    "in.infiltration",
    "in.insulation_slab",
    "in.insulation_wall",
    "in.lighting",
    "in.misc_extra_refrigerator",
    "in.misc_freezer",
    "in.misc_gas_fireplace",
    "in.misc_gas_grill",
    "in.misc_gas_lighting",
    "in.misc_hot_tub_spa",
    "in.misc_pool_pump",
    "in.misc_pool_heater",
    "in.misc_well_pump",
    "in.natural_ventilation",
    "in.neighbors",
    "in.occupants",
    "in.plug_loads",
    "in.refrigerator",
    "in.water_heater_efficiency",
    "in.windows"
]

# categorical characteristics
com_chars = [
    "in.building_subtype",
    "in.building_type",
    "in.rotation",
    "in.number_of_stories",
    "in.sqft",
    "in.hvac_system_type",
    "in.weekday_operating_hours",
    "in.weekday_opening_time",
    "in.weekend_operating_hours",
    "in.weekend_opening_time",
    "in.heating_fuel",
    "in.service_water_heating_fuel",
    "stat.average_boiler_efficiency",
    "stat.average_gas_coil_efficiency"
]

total_chars = res_chars + com_chars

if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    import pickle
    from sklearn.preprocessing import OneHotEncoder
    from pathlib import Path

    dataset_path = Path(os.environ.get('BUILDINGS_BENCH', ''))

    types = ["com", "res"]
    categories = []

    for t in types:
        df1 = pd.read_parquet(f"{dataset_path}/metadata/{t}stock_amy2018.parquet",
                            engine="pyarrow")
        df2 = pd.read_parquet(f"{dataset_path}/metadata/{t}stock_tmy3.parquet",
                            engine="pyarrow")
        df = pd.concat([df1, df2])

        chars = {"com": com_chars, "res": res_chars}[t]

        for char in chars:      
            uniq = df[char].unique()
            categories.append(uniq)    

        with open(f"{t}_categories.pickle", "ab") as f:
            pickle.dump(categories, f)

        with open(f"{t}_categories.pickle", "rb") as f:
            print(pickle.load(f))
