import pickle
import numpy as np
import warnings


warnings.filterwarnings("ignore")


class GesturePredictor:


    def __init__(self):

        # Load trained Random Forest model

        with open(
            "training/gesture_model.pkl",
            "rb"
        ) as file:

            self.model = pickle.load(file)



    def predict(
        self,
        landmarks
    ):


        # Convert landmarks into ML input format

        data = np.array(
            landmarks
        ).reshape(
            1,
            -1
        )


        # Predict alphabet

        prediction = self.model.predict(
            data
        )[0]


        # Prediction confidence

        confidence = max(
            self.model.predict_proba(
                data
            )[0]
        )


        return (
            prediction,
            confidence
        )