from collections import deque
from collections import Counter


class PredictionStabilizer:


    def __init__(
        self,
        buffer_size=8,
        confidence_threshold=0.70
    ):

        self.predictions = deque(
            maxlen=buffer_size
        )


        self.confidence_threshold = confidence_threshold



    def update(
        self,
        prediction,
        confidence
    ):


        # ignore weak predictions

        if confidence < self.confidence_threshold:

            return None



        self.predictions.append(
            prediction
        )



        # find most common prediction

        stable_prediction = Counter(
            self.predictions
        ).most_common(1)[0][0]


        return stable_prediction