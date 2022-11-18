import os, sys, math, argparse
import numpy as np
import pandas as pd

## model
import xgboost as xgb
#from xgboost import plot_importance
#from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV   #Perforing grid search
from sklearn import ensemble, preprocessing, metrics
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score

class ob_name:
	def __init__(self):
		self.check_arg(sys.argv[1:])
		self.main()

		pass

	def check_arg(self,args=None):
		parser = argparse.ArgumentParser(description='Script to perform intracellular oxidative stress status prediction.\n \
			\n \
			=============================\n \
			|| Example of the command: ||\n \
			=============================\n \
			python OSpred.py --i expression.csv --o output.csv --t 8 --m prob'
			,formatter_class=argparse.RawTextHelpFormatter)

		parser.add_argument(	'--i' ,type=str,
					help='please input the expresion matrix (LogNormalized, scale factor=10000) in csv format.')

		parser.add_argument(	'--m' ,type=str,
					default='prob',
					help='output probability or binary class. \n \
					\n \
	- binary\n \
	- prob\n \
					')

		parser.add_argument(	'--o' ,type=str,
					default='output.csv',
					help='prefix for the prediction output. (default: output.csv)')

		parser.add_argument(	'--t' ,type=int,
					default=8,
					help='threads to use. (default: 8)')

		self.args = parser.parse_args(args)

	def XG(self, X_train, y_train,params):
		model = xgb.XGBClassifier(**params, use_label_encoder=False)
		model.fit(X_train, y_train)

		return model

	def pre_processing(self, df, mode, featuresss):
		t_exp_data = df.T
		t_exp_data.columns = t_exp_data.iloc[0]
		t_exp_data.drop(t_exp_data.index[0],inplace=True)
		#print(type(t_exp_data))
		if mode == "ft":
			t_exp_data_final = t_exp_data[featuresss].astype(float, errors = 'raise')
		else:
			t_exp_data_final = t_exp_data.astype(float, errors = 'raise')

		return t_exp_data_final


	def main(self):
		OSpred_path = sys.path[0]
		params = {
			#general setting
			'booster': 'gbtree',
			'n_estimators':1000,
			'verbosity':1,
			#'gpu_id': 0,
			#'tree_method':'gpu_hist',
			#learning step
			'eta':0.01,
			#task parameters
			'objective': 'binary:logistic',
			'gamma':0,
			'colsample_bytree':0.6,
			'max_depth':7,
			'min_child_weight':2,
			'reg_alpha':0.05,
			'reg_lambda':0.05,
			'subsample':0.6
		}
		#===========================================================================
		#  Read training data
		#===========================================================================
		training_df = pd.read_csv(OSpred_path+'/docs/train_matrix.csv',index_col=0)
		label = pd.read_csv(OSpred_path+'/docs/train_label.csv')

		#===========================================================================
		#  data processing
		#===========================================================================
		DEG_feature =  pd.read_csv(OSpred_path+'/docs/features.csv')
		featuresss = list(DEG_feature["gene"])
		testing = pd.read_csv(self.args.i)
		testing_matrix = self.pre_processing(testing,"ft", featuresss)

		#===========================================================================
		#  Prediction
		#===========================================================================
		model = self.XG(training_df.values, label["output"], params)
		if self.args.m.lower() == 'binary':
			ans = model.predict(testing_matrix)

			pass

		elif self.args.m.lower() == 'prob':
			ans_prob = model.predict_proba(testing_matrix)

			pass

		else:
			raise Exception('Wrong output mode')

		cell_name = list(testing_matrix.index.values)
		ans_df = pd.DataFrame({'cell':cell_name, 'ros_pred':ans})
		if self.args.m.lower() == 'binary': print(ans_df.groupby(['ros_pred']).count())
		ans_df.to_csv(self.args.o, index = False)

if __name__ == '__main__':
	ob=ob_name()
