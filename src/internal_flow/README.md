Here how this submodule is subdivided:


- ```callbacks.py``` contains the class ```SaveConfigAndMetrics```, to ensure we log the hyperparameters and metrics properly.
- ```data.py``` contains the class ```SingleImageTestDataset``` and ```SingleImageTrainDataset```
- ```flow_model.py``` contains a class that inherits from the ```Flow``` class from our module, to add support for test metrics like LPIPS and PSNR
- ```model.py``` contains the model, copied from the code of Nicolas Cherel
- ```utils.py```
- ```vizualization.py```, with functions called from ```flow_model.py```