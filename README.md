<h1><center>Flow Matching</center></h1>


Ce repository contient tout le code nécecssaire pour faire du **flow matching**. Il est implémenté en utilisant PyTorch 2.7.0.

Les tests et expériences sont contenus dans le dossier ```/notebooks```. Ils comprennent:
- L'entraînement de modèles et la visualisation du flot en 2D
- L'entraînement de modèles sur MNIST


## Structure

Le code est structuré de cette façon:

```text
.
├── confs                                   # Configurations pour les tests
├── data
├── notebooks
│   ├── stable_diffusion_generation.ipynb   # Test en utilisant un modèle pré-entraîné
│   ├── test_module.ipynb                   # Test du module en 2D
│   └── train_model.ipynb                   # Entraînement d'un modèle sur MNIST
└── src
    ├── experiments                         # Module contenant du code pour ne pas polluer les notebooks
    │   ├── callback
    │   ├── datasets
    │   │   └── datasets.py
    │   ├── models
    │   │   ├── layers.py
    │   │   └── models.py
    │   ├── utils
    │   │   └── mlflow.py
    │   └── visualization
    │       └── plots.py
    └── flows                               # Module principal, à tester 
        ├── losses
        │   └── loss.py
        ├── models                          # Contient un ModelWrapper
        ├── flow.py                         # Classe principale
        ├── path.py                         # Chemins (uniquement affine pour l'instant)
        ├── schedulers.py                   # Schedulers (LinearScheduler, CosineScheduler)
        └── utils.py
README.md
