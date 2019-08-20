# recommendation-system

PYTHONPATH="." luigi --module recommendation.task.model.auto_encoder UnconstrainedAutoEncoderTraining --project yelp_business_autoencoder --local-scheduler


PYTHONPATH="." luigi --module recommendation.task.data_preparation PrepareDataFrames --local-scheduler