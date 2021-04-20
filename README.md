# COMP-6721
## README for group NS_01

Our code submission contains the following files:

├── dataset
│   ├── NotPerson                           # this folder contains the images of Not Person
│   ├── WithMask                            # this folder contains the images of With Mask
│   └── WithoutMask                         # this folder contains the images of Without Mask
├── model
│   ├── Executor.py                         # a wrapper on our CNN model, which has all the model paramters set.
│   ├── __init__.py
│   └── convolutional_neural_network.py     # CNN model
├── preprocessing.py                        # some utility for the preprocessing
├── AI project phase1.ipynb                 # notebook to generate the project step by step and show the results
├── AI_Project_report.pdf                   # report of the project
└── trained_model.pt                        # the trained model


The link to the complete dataset:
https://drive.google.com/drive/folders/1vxW3UCwwEXcB3nyD-Of6KmsQaqxuCRCi?usp=sharing

* In order to train the model, please create instance of it and run train_model_executor method with number of epochs and data source(as provided in the jupyter notebook training model block). Also runing the jupyter notebook file that we provided, you can see the steps that we took for this project and you can see the output.