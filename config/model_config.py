from model.roberta_classifier import RobertaClassifier
from model.lstm_classifier import LSTMClassifier
from model.sklearn_classifier import RandomForestSKLearnClassifier, LogisticRegressionSKLearnClassifier

model_dict1 = {'model': ["roberta", "lstm"]}
model_dict2 = {'model': ["random_forest", "logistic_regression"]}

model_info = {
	'roberta': {
		'class': RobertaClassifier,
		"tunable_model_args": {
			# "hidden_dropout_prob": [0.1, 0.2, 0.3]
			"hidden_dropout_prob": [0.1]
		}
	},
	"lstm": {
		"class": LSTMClassifier,
		"tunable_model_args": {
			"hidden_size": [200],
			"pad_packing": True,
		}
	},
	"random_forest": {
		"class": RandomForestSKLearnClassifier,
		"tunable_model_args": {
			'n_estimators': [4, 16, 64, 256, 512]
		}
	},
	"logistic_regression": {
		"class": LogisticRegressionSKLearnClassifier,
		"tunable_model_args": {
			'C': 1
		}
	}
}