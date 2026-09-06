# HKU-iGEM-2026--Dry-Lab

Section 1.1: Upstream Wound Bed Generation Kinetics (Model 1A)
This section models how the inflammation status drives the baseline Nitric Oxide (NO) concentration entering the patch. Shows the model links variations in patient wound severity.
#Plot 1: Inflammation-Driven NO Generation Curve 
What it shows: How tissue inflammation scales up steady-state wound NO via macrophage recruitment.
X-Axis: Nuclear NF-kB concentration ([p65]_{nuc}).
Y-Axis: Steady-state wound NO concentration ([NO]_{wound}).
Plot 3 separate lines on this graph representing low, medium, and high Macrophage Densities (N_{mac}). 


Sections 1.2 & 1.3: Material Setup & Transport Parameters (Model 1B)
Evaluates the structural properties of porous biomaterials control the physical degradation and travel limits of the NO gas.
#Plot 2: Material Structure Selection Heatmap
what it shows: The structural landscape of the patch material, mapping how porosity and tortuosity interact to dictate NO survival distance.
X-Axis (Horizontal): Matrix Porosity (varepsilon), from 0.1 to 1.0).
Y-Axis (Vertical): Matrix Tortuosity (tau), from 1.0 to 3.0).
Color Scale (Z-Axis Intensity): Characteristic Penetration Depth (L_{d}) in microns.
Use a bright color gradient (like viridis). Plot two distinct markers on top of the map: a star marking the Bacterial Cellulose parameters, and a circle for your Hydrogel parameters.iGEM Wiki Insight: 
It should visually justifies why your chosen materials structurally allow the NO signal to pass through.


Sections 1.4 & 1.5: Boundary Conditions & Analytical Solutions
This section tracks the exact spatial drop-off of the gas as it crosses the physical boundaries of the dual-layer device.
#Plot 3: 1D Spatial NO Gradient Across Two Domains
What it shows: A steady-state snapshot of the NO concentration at every single micron inside your patch.
X-Axis: Distance through the patch (x, from 0 to (L_{total}).
Y-Axis: Nitric Oxide Concentration ([NO](x)).
Draw a vertical dashed line to mark the exact boundary interface where the Cellulose Membrane ends (L_{cellulose}) and the Hydrogel Matrix begins. The curve must show a distinct change in slope right at this dashed line to prove your interfacial flux continuity math is working.


Plot 4: Patch Thickness Optimization Design Curve
What it shows: The concentration of NO hitting your bacteria ([NO]_{chassis}) based on how thick you manufacture the patch.
X-Axis: Total Patch Thickness (L_{total}).
Y-Axis: Biosensor boundary NO concentration ([NO]_{chassis}).
Draw a horizontal line marking the known promoter activation threshold (K_{A}). The point where your curve drops below this line dictates your maximum manufacturing limit. Translates dry lab predictions into physical guidelines for the wet lab. 


To compile all 5 of these figures, you will need to install these libraries:
pip install numpy scipy matplotlib seaborn
