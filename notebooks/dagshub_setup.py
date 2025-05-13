import dagshub
import mlflow
dagshub.init(repo_owner='hanitkaur', repo_name='mlops-miniproject', mlflow=True)
mlflow.set_tracking_uri('https://dagshub.com/hanitkaur/mlops-miniproject.mlflow')

with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)