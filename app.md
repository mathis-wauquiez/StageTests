# Démo d'inpainting de textures avec le modèle de flot + classifier free guidance

Cette démo sert à:
- Tester nos différentes modèles
- Tester l'impact du paramètre de CFG scale
- Tester différentes configurations du solver d'équation différentielle

J'utilise ```torchdiffeq.odeint``` pour intégrer le champ de vitesse. Celui-ci possède plusieurs solveurs possibles, divisés en deux catégories: les solveurs à pas fixes et les solveurs à pas adaptatif. Les solveurs à pas fixe incluent ```euler```, ```midpoint```, ```rk4```. Ceux à pas adaptatif incluent ```dopri5``` et ```dopri8```. Les solvers à pas fixe utilisent une option ```step_size```. Etant donné que l'on intègre de 0 à 1, un ```step_size``` de ```0.01``` va intégrer en utilisant 100 pas. Les solveurs à pas adaptatifs, eux, possèdent un paramètre de tolérance absolue, ```atol```, ainsi qu'un paramètre de tolérance relative, ```rtol```. Pour plus d'informations, se référer à la [documentation de torchdiffeq](https://github.com/rtqichen/torchdiffeq).


Je conseille de laisser Date / Time aux valeurs par défaut.