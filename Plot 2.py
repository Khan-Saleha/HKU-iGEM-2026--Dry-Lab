import numpy as np
import matplotlib.pyplot as plt

D_water = 3.3e-5
t_half = 5.0        
k_decay = np.log(2) / t_half

porosity = np.linspace(0.1, 1.0, 100)    
tortuosity = np.linspace(1.0, 3.0, 100)   
E, T = np.meshgrid(porosity, tortuosity)

D_eff = D_water * (E / T)
L_d_cm = np.sqrt(D_eff / k_decay)
L_d_microns = L_d_cm * 10000              

plt.figure(figsize=(8, 6))
heatmap = plt.pcolormesh(E, T, L_d_microns, shading='auto', cmap='viridis')
cbar = plt.colorbar(heatmap)
cbar.set_label('Characteristic Penetration Depth $L_d$ ($\\mu m$)')

BC_porosity, BC_tortuosity = 0.94, 1.03     
HG_porosity, HG_tortuosity = 0.618, 1.27    

L_d_microns_BC = np.sqrt((D_water * (BC_porosity / BC_tortuosity)) / k_decay) * 10000
L_d_microns_HG = np.sqrt((D_water * (HG_porosity / HG_tortuosity)) / k_decay) * 10000
print(f"Characteristic Penetration Depth for Bacterial Cellulose: {L_d_microns_BC:.2f} μm")
print(f"Characteristic Penetration Depth for Hydrogel: {L_d_microns_HG:.2f} μm")

plt.plot(BC_porosity, BC_tortuosity, '*', color='yellow', markersize=15, label='Bacterial Cellulose')
plt.plot(HG_porosity, HG_tortuosity, 'o', color='white', markersize=10, label='Hydrogel')

plt.xlabel('Matrix Porosity ($\\epsilon$)')
plt.ylabel('Matrix Tortuosity ($\\tau$)')
plt.title('Plot 2: Material Structure Selection Heatmap')
plt.legend(loc='upper left')
plt.show()