# ALTEGRAD-Challenge-2020-2021
https://www.kaggle.com/c/altegrad-2020/overview


# Predict author's h-index based on text and graph data

### Run the *best score* model :
 - The corresponding script is located in ModelGraphFeatures ; in the console, type *python ModelGraphFeatures/main.py* 

### A few instructions and recommendations :

 - This github repository does not host the base data for this challenge, the data should be loaded independently in the /code folder (see kaggle link)
 - For the code to run correctly, you should set your working directory at root /code when running a python script
 - The python files launching models and generating subsequent predictions are located in folders named Model*suffix* (e.g. ModelGNN). These folders contain a *main.py* file that launch the actual model. A typical model launching would be done for instance by typing in the console "python ModelGNN/main.py" with the /code folder as root directory
 - The model classes are in the models.py at the root directory, they are called specifically by each of the files launching models (see above)
