import pandas as pd
import numpy as np
import os
import torch
from pathlib import Path
import glob

building_years = ["comstock_tmy3", "resstock_tmy3", "comstock_amy2018", "resstock_amy2018"]
metadata_path = Path(os.environ.get('BUILDINGS_BENCH', '')) / "metadata_dev"

# randomly samples (10k by default) buildings and top 10 most important characteristics
# and saves the corresponding dataframes to simcap folder under metadata
def sample_buildings(seed=1, length=10000):
    # 10 most characteristics for com and res buildings
    com_names = [
        'in.building_subtype', 'in.building_type', 'in.rotation',
        'in.number_of_stories', 'in.sqft', 'in.hvac_system_type',
        'in.weekday_operating_hours', 'in.weekday_opening_time',
        'in.weekend_operating_hours', 'in.weekend_opening_time'
    ]
    res_names = [
        'in.sqft', 'in.ceiling_fan', 'in.clothes_washer_presence',
       'in.geometry_floor_area_bin', 'in.has_pv', 'in.heating_fuel',
       'in.heating_setpoint_has_offset', 'in.lighting', 'in.misc_freezer',
       'in.misc_gas_grill'
    ]
    
    if not os.path.exists(metadata_path / "simcap" ):
        os.makedirs(metadata_path / "simcap" )

    for name in building_years:
        df = pd.read_parquet(metadata_path / f"{name}.parquet", engine="pyarrow")
        sample_df = df.sample(n=length//4, random_state=seed)
        if "com" in name:
            sample_df = sample_df[com_names]
        else:
            sample_df = sample_df[res_names]
        sample_df.to_csv(metadata_path / "simcap" / f"{name}_simcap_{length}.csv")
    
# post-process llama2's response, removing undesired output such as "sure! here's the building..."
def post_processing():
    for bd_yr in ["comstock_amy2018", "resstock_amy2018"]:
        files = glob.glob(str(metadata_path / "simcap" / f"{bd_yr}/*.txt"))
        for f_name in files:
            with open(f_name, "r") as f:
                out = f.read()
            
            ans = []
            for line in out.split("\n"):
                if "Sure" in line or "Here" in line or line == "":
                    continue
                ans.append(line)

            with open(f_name, "w+") as f:
                f.write(" ".join(ans))

            print(f_name)
            print(out)
            print("-" * 20)
            print(" ".join(ans))
            print("-" * 20)

def get_BERT_embeddings():
    from transformers import BertTokenizer, BertModel
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained("bert-base-uncased")

    for bd_yr in ["comstock_amy2018", "resstock_amy2018"]:
        files = glob.glob(str(metadata_path / "simcap" / f"{bd_yr}/*.txt"))
        for f_name in files:
            with open(f_name, "r") as f:
                text = f.read()
            
            encoded_input = tokenizer(text, return_tensors='pt') 
            with torch.no_grad():
                output = model(**encoded_input)
                embedding = output[0].squeeze().mean(dim=0)
                embedding = embedding.cpu().numpy()
                base = os.path.basename(f_name).split("_")[0]
                print(base)
                np.save(metadata_path / "simcap" / bd_yr / f"{base}_emb.npy", embedding)

# post_processing()
# get_BERT_embeddings()