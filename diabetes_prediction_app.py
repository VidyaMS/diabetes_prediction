from flask import Flask, render_template, request, send_file
import pickle as p
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score , roc_auc_score , classification_report, confusion_matrix

from werkzeug.utils import secure_filename


app = Flask(__name__)

@app.route('/')
def features():
   return render_template('Upload_file.html')

@app.route('/predict' , methods =  ['GET' , 'POST'])
def upload_and_predict():
	if request.method == 'POST':
      		## read in the file  for model prediction.
		f = request.files['file']
		f.save(secure_filename(f.filename))
		test = pd.read_csv(f.filename)
		cols = test.columns.tolist()
      		## hot encode categorical variables.
		def hot_encode(df, cols):
			for c in cols:
				col_dummies = pd.get_dummies(df[c], prefix_sep = '_' , prefix = c)
				df = pd.concat([df, col_dummies] , axis = 1)
				df.drop(c, inplace = True, axis = 1)
			return df

		test_encoded = hot_encode(test[cols[:-1]], cols[1:-1]) 
		X_test = test_encoded
		y_test = test['class_num']
		##
		## predict for the uploaded file 
		predicted_class = model.predict(X_test)
		test['predicted_class'] = predicted_class
		test['probability'] = model.predict_proba(X_test)[:,1]
		(tn,fp,fn,tp) = confusion_matrix(y_test, predicted_class).ravel()
		result_dict =  {'Correctly predicted non-diabetic cases ': tn , 'Falsely predicted non-diabetic cases' : fp , 'Falsely predicted diabetic cases': fn , 'Correctly predicted diabetic cases ': tp}

		## save the file 
		test.to_csv("predictions.csv", index =False)
		return render_template("early_stage_diabetes_prediction_1.html" , result = result_dict)

@app.route('/download' , methods = ['GET', 'POST'])
def download_file():
	try:
		return send_file('predictions.csv')
	except Exception as e:
		return str(e)
	
		
	
if __name__ == '__main__':
    modelfile = 'diabetes_prediction_model.sav'
    model = p.load(open(modelfile, 'rb'))
    print("model loaded")
    app.run(debug = True,host = '0.0.0.0' , port = 5000 )
