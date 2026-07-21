import sys

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

from src.exception import CustomException
from src.logger import logging


if __name__ == "__main__":

    try:

        logging.info("Training pipeline started")

        # Data ingestion
        data_ingestion = DataIngestion()

        train_path, test_path = (
            data_ingestion.initiate_data_ingestion()
        )


        # Data transformation
        data_transformation = DataTransformation()

        train_arr, test_arr, preprocessor_path = (
            data_transformation.initiate_data_transformation(
                train_path,
                test_path
            )
        )


        logging.info(
            f"Preprocessor saved at: {preprocessor_path}"
        )


        # Model training
        model_trainer = ModelTrainer()

        r2_score = (
            model_trainer.initiate_model_trainer(
                train_arr,
                test_arr
            )
        )


        print("Model R2 Score:", r2_score)


    except Exception as e:
        raise CustomException(e, sys)