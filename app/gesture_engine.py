import mediapipe as mp
import time

from predictor import GesturePredictor
from stabilizer import PredictionStabilizer


class GestureEngine:


    def __init__(self):


        # Load Random Forest model

        self.predictor = GesturePredictor()


        # Load prediction smoother

        self.stabilizer = PredictionStabilizer()



        # MediaPipe setup

        BaseOptions = mp.tasks.BaseOptions

        HandLandmarker = mp.tasks.vision.HandLandmarker

        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

        VisionRunningMode = mp.tasks.vision.RunningMode



        options = HandLandmarkerOptions(

            base_options=BaseOptions(

                model_asset_path="hand_landmarker.task"

            ),


            running_mode=VisionRunningMode.IMAGE,


            num_hands=1
        )



        self.detector = HandLandmarker.create_from_options(

            options

        )



        # displayed result

        self.display_prediction = None

        self.display_confidence = 0



        # backend result

        self.current_prediction = None

        self.current_confidence = 0



        # timer

        self.last_update_time = time.time()



    def process_frame(

        self,

        frame

    ):



        mp_image = mp.Image(

            image_format=mp.ImageFormat.SRGB,

            data=frame

        )



        result = self.detector.detect(

            mp_image

        )



        # -------------------------
        # HAND DETECTED
        # -------------------------


        if result.hand_landmarks:



            landmarks = result.hand_landmarks[0]


            features = []



            for point in landmarks:


                features.extend(

                    [

                        point.x,

                        point.y,

                        point.z

                    ]

                )



            alphabet, confidence = self.predictor.predict(

                features

            )



            stable_prediction = self.stabilizer.update(

                alphabet,

                confidence

            )



            if stable_prediction:


                self.current_prediction = stable_prediction


                self.current_confidence = confidence



        # -------------------------
        # NO HAND
        # -------------------------


        else:


            self.current_prediction = None


            self.current_confidence = 0




        # -------------------------
        # 5 SECOND UPDATE LOGIC
        # -------------------------


        if time.time() - self.last_update_time >= 5:



            if self.current_prediction:



                self.display_prediction = self.current_prediction


                self.display_confidence = self.current_confidence



            else:



                self.display_prediction = None


                self.display_confidence = 0




            self.last_update_time = time.time()




        return {


            "hand_detected": result.hand_landmarks is not None,


            "prediction": self.display_prediction,


            "confidence": self.display_confidence

        }