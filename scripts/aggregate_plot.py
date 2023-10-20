from buildings_bench import load_pretraining
import torch
import numpy as np
import matplotlib.pyplot as plt

pred = torch.load("prediction.pt")
targ = torch.load("target.pt")
time = torch.load("time.pt")
bd_char = torch.load("building_chars.pt")
bd_id = torch.load("building_ids.pt")

print(pred)
print(targ)
print(time)
print(bd_char)
print(bd_id)

bd_types = ['FullServiceRestaurant', 'Hospital', 'LargeHotel', 'LargeOffice',
       'MediumOffice', 'Outpatient', 'PrimarySchool',
       'QuickServiceRestaurant', 'RetailStandalone', 'RetailStripmall',
       'SecondarySchool', 'SmallHotel', 'SmallOffice', 'Warehouse']

fig=plt.figure(figsize=(12,8), dpi= 100, facecolor='w', edgecolor='k')
for i in range(14):
    mask = bd_char[:, 10+i] == 1
    plt.subplot(4, 4, i+1)
    idx = time[mask, :].argmin(dim=1)
    p = pred[mask, :].repeat(1, 2)
    t = targ[mask, :].repeat(1, 2)
    p = torch.cat([p[j, id:id + 24].unsqueeze(0) for j, id in enumerate(idx)]).numpy()
    t = torch.cat([t[j, id:id + 24].unsqueeze(0) for j, id in enumerate(idx)]).numpy()
    num = np.unique(bd_id[mask, :].cpu().numpy(), axis=0).shape[0]
    plt.plot(np.mean(p, axis=0), label="prediction")
    plt.plot(np.mean(t, axis=0), label="target")
    plt.title(bd_types[i] + f" {num}" )
    plt.legend()
    plt.tight_layout()

plt.savefig('plot.png')