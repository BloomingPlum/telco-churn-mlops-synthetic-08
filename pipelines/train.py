"""Train a churn model and save it to disk.

This script is intentionally robust: MLflow integration is optional and
the script will still train and save a local model when MLflow is not
available or not configured.
"""

import os
import sys
import json
from datetime import datetime, timezone
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Optional progress bar
try:
	from tqdm import tqdm
	TQDM_AVAILABLE = True
except Exception:
	TQDM_AVAILABLE = False


# Paths (can be overridden with env vars)
DATA_PATH = os.getenv('DATA_PATH', 'data/telco_customers.csv')
MODEL_PATH = os.getenv('MODEL_PATH', 'models/churn_model.pkl')
# Evaluation metrics are saved next to the model; the API reads them at startup
# and exposes them as model_accuracy / model_f1_score / model_auc_roc gauges.
METRICS_PATH = os.getenv('METRICS_PATH', os.path.join(os.path.dirname(MODEL_PATH) or 'models', 'model_metrics.json'))
MODEL_VERSION = os.getenv('MODEL_VERSION', 'v1.0-local')

# Exactly the fields accepted by the API (src/api/models.py -> CustomerFeatures).
# The generated CSV also contains customerID and RecordDate; if those reach the
# pipeline, every /predict call fails with "columns are missing: {'RecordDate'}".
FEATURES = [
	'tenure', 'MonthlyCharges', 'TotalCharges', 'gender', 'SeniorCitizen',
	'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
	'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
	'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
]

# Depth-limited forest: an unlimited RandomForest on 50k rows overfits
# (accuracy ~0.65) and pickles to ~190 MB, over GitHub's 100 MB file limit.
# These settings give ~0.68 accuracy and a ~2.5 MB (compressed) model file.
RF_PARAMS = {'n_estimators': 100, 'max_depth': 10, 'min_samples_leaf': 10, 'random_state': 42}

# MLflow configuration via environment variables (optional)
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', '').strip()
MLFLOW_EXPERIMENT = os.getenv('MLFLOW_EXPERIMENT', 'telco_churn_experiment')
MLFLOW_REGISTER = os.getenv('MLFLOW_REGISTER_MODEL', 'false').lower() == 'true'
MLFLOW_REGISTERED_NAME = os.getenv('MLFLOW_REGISTERED_NAME', 'ChurnModel')

# Import mlflow only if available - don't fail when it's not installed
try:
	import mlflow
	import mlflow.sklearn
	MLFLOW_AVAILABLE = True
except Exception:
	MLFLOW_AVAILABLE = False


def load_data(path: str) -> pd.DataFrame:
	if not os.path.exists(path):
		print(f"Error: data file not found: {path}", file=sys.stderr)
		sys.exit(2)

	df = pd.read_csv(path)
	return df


def build_pipeline(X: pd.DataFrame) -> Pipeline:
	categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
	numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

	# Build a OneHotEncoder in a way that's compatible with multiple
	# scikit-learn versions (some use `sparse`, newer ones use `sparse_output`).
	try:
		encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
	except TypeError:
		# Fallback for newer sklearn versions
		encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

	preprocessor = ColumnTransformer(
		transformers=[
			('num', 'passthrough', numerical_cols),
			('cat', encoder, categorical_cols),
		]
	)

	model = Pipeline(steps=[
		('preprocessor', preprocessor),
		# n_jobs stays at 1: the API predicts one row per request, and a parallel
		# forest spends more time on thread start-up than on the 100 trees.
		('classifier', RandomForestClassifier(**RF_PARAMS))
	])

	return model


def compute_metrics(model: Pipeline, X_test, y_test) -> dict:
	proba = model.predict_proba(X_test)[:, 1]
	pred = (proba >= 0.5).astype(int)
	return {
		'accuracy': round(float(accuracy_score(y_test, pred)), 4),
		'f1': round(float(f1_score(y_test, pred)), 4),
		'roc_auc': round(float(roc_auc_score(y_test, proba)), 4),
		'high_risk_share': round(float((proba > 0.8).mean()), 4),
	}


def save_metrics(metrics: dict, n_train: int, n_test: int) -> None:
	payload = {
		'model_version': MODEL_VERSION,
		'trained_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
		'n_train': int(n_train),
		'n_test': int(n_test),
		'params': RF_PARAMS,
		'features': FEATURES,
		**metrics,
	}
	os.makedirs(os.path.dirname(METRICS_PATH) or 'models', exist_ok=True)
	with open(METRICS_PATH, 'w') as f:
		json.dump(payload, f, indent=2)
	print(f'Metrics saved to {METRICS_PATH}: {metrics}')


def train_and_evaluate(model: Pipeline, X_train, X_test, y_train, y_test, show_progress: bool = True):
	"""Train the pipeline. If the classifier supports warm_start, train in chunks
	and show a tqdm progress bar (if available). Returns (accuracy, trained_pipeline).
	"""
	# Extract preprocessor and classifier
	preprocessor = model.named_steps['preprocessor']
	classifier = model.named_steps['classifier']

	# Fit/transform preprocessors once
	X_train_t = preprocessor.fit_transform(X_train)
	X_test_t = preprocessor.transform(X_test)

	total_estimators = getattr(classifier, 'n_estimators', None)
	supports_warm = hasattr(classifier, 'warm_start')

	if supports_warm and total_estimators and total_estimators > 1:
		# Train in chunks to provide progress updates
		chunk = max(1, total_estimators // 10)
		trained = 0

		if TQDM_AVAILABLE and show_progress:
			iterator = list(range(chunk, total_estimators + 1, chunk))
			if iterator[-1] != total_estimators:
				iterator.append(total_estimators)
			for n in tqdm(iterator, desc='Training trees', unit='trees'):
				classifier.warm_start = True
				classifier.n_estimators = n
				classifier.fit(X_train_t, y_train)
				trained = n
		else:
			print(f"Training RandomForest in chunks up to {total_estimators} trees...")
			for n in range(chunk, total_estimators + 1, chunk):
				classifier.warm_start = True
				classifier.n_estimators = n
				classifier.fit(X_train_t, y_train)
				trained = n
	else:
		# Fallback: single fit
		classifier.fit(X_train_t, y_train)

	# Build final pipeline with trained components
	trained_pipeline = Pipeline([('preprocessor', preprocessor), ('classifier', classifier)])

	# Evaluate
	y_pred = trained_pipeline.predict(X_test)
	acc = accuracy_score(y_test, y_pred)
	return acc, trained_pipeline


def main():
	df = load_data(DATA_PATH)

	# Basic preprocessing
	df = df.drop(['customerID'], axis=1, errors='ignore')
	if 'TotalCharges' in df.columns:
		df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
	df = df.dropna()

	if 'Churn' not in df.columns:
		print('Error: target column "Churn" not found in data', file=sys.stderr)
		sys.exit(2)

	missing = [c for c in FEATURES if c not in df.columns]
	if missing:
		print(f'Error: feature columns missing from data: {missing}', file=sys.stderr)
		sys.exit(2)

	X = df[FEATURES]
	y = df['Churn'].map({'Yes': 1, 'No': 0})

	model = build_pipeline(X)

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
	metrics = None

	# If MLflow is available and a tracking URI was provided, attempt to log there.
	if MLFLOW_AVAILABLE and MLFLOW_TRACKING_URI:
		try:
			mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
			mlflow.set_experiment(MLFLOW_EXPERIMENT)
		except Exception as e:
			print(f'Warning: could not configure mlflow ({e}), continuing without mlflow logging')

	if MLFLOW_AVAILABLE and MLFLOW_TRACKING_URI:
		try:
			with mlflow.start_run():
				try:
					mlflow.log_params(RF_PARAMS)
				except Exception:
					pass

				acc, model = train_and_evaluate(model, X_train, X_test, y_train, y_test, show_progress=True)
				print(f'Accuracy: {acc:.4f}')
				metrics = compute_metrics(model, X_test, y_test)
				try:
					mlflow.log_metrics(metrics)
				except Exception:
					pass

				try:
					if MLFLOW_REGISTER:
						mlflow.sklearn.log_model(model, 'model', registered_model_name=MLFLOW_REGISTERED_NAME)
					else:
						mlflow.sklearn.log_model(model, 'model')
				except Exception as e:
					print(f'Warning: failed to log/register model to MLflow: {e}')
		except Exception as e:
			print(f'Warning: mlflow run failed ({e}), training locally instead')
			metrics = None
			acc, model = train_and_evaluate(model, X_train, X_test, y_train, y_test, show_progress=False)
			print(f'Accuracy (local): {acc:.4f}')
	else:
		# Train without MLflow logging
		acc, model = train_and_evaluate(model, X_train, X_test, y_train, y_test, show_progress=True)
		print(f'Accuracy (no mlflow): {acc:.4f}')

	# Always save local model + its evaluation metrics
	os.makedirs(os.path.dirname(MODEL_PATH) or 'models', exist_ok=True)
	joblib.dump(model, MODEL_PATH, compress=3)
	print(f'Model saved to {MODEL_PATH}')
	if metrics is None:
		metrics = compute_metrics(model, X_test, y_test)
	save_metrics(metrics, len(X_train), len(X_test))


if __name__ == '__main__':
	main()
