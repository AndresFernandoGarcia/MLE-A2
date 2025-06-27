# Credi_Score - MLE Assignment 2
- This is an assignment for Machine Learning Engineering.
- The main objective was to continue building a machine learning pipeline with different technologies and as a final test to combine all of what we have learned.

### Overall Proces: 
<p>Initiate Docker Container for effective model tracking and airflow functionality</p>
<p>If manual test: run "python src/preprocessing.py" to build the datamart with the current available data (can be changed in preprocess.json).</p>
<p>Then run "python src/train_model.py" to run xgboost model training. These settings can be changed within a json file. Check default_model_train.json for structure.</p>
<p>Inference.py is run the same way: "python src/inference.py" and config can also be defined within the arguments passed to the script. This script saves the predictions to csv within the project directory. </p>
<p>If a custom config file is used it must be passed as an argument while trying to run the script. i.e. "python src/preprocessing.py --config configs/custom_preprocess.json". Same applies to train_model.py and inference.py.</p>

### Links to Airlfow and MLflow used: 
<p>localhost:8000 (MLflow)</p>
<p>localhost:8080 (Apache Airflow) </p>

### Project Repository
https://github.com/AndresFernandoGarcia/MLE-A2

### Special Thanks to JENNIFER POERNOMO for providing the original code which this project is based on
<p> Her original code was used as the basis for this project. It is modified, however, similarities will nonetheless be present. Similarities will be found during the preprocessing stages. Beyond that the way data is split will also be reminiscent of her original code.</p>

### JENNIFER POERNOMO's Original Project Github Link
https://github.com/Jenpoer/creditkarma
