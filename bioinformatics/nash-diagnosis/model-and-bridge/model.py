from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np



class Model():
    def start(self):
        print('Loading model...')

        self.model = load_model("model.h5")

        print('Model has been loaded...')
        print('Ready for inference...')
        print()




    def predict(self, inputs):
        # inputs = [
        #     {
        #         "Age": 35.0,
        #         "Gender (female)": 0.0,
        #         "Nationality": 2.0,
        #         "BMI": 28.035981272865783,
        #         "History.of.hbv.vaccine": 0.1223021582733813,
        #         "Diabetes": 0.0,
        #         "Hypertension": 0.0,
        #         "Fatty.Liver": 0.0,
        #         "High.blood.fats": 0.0,
        #         "FBS": 74.0,
        #         "TG": 213.0,
        #         "Chol": 267.0,
        #         "HDL": 55.0,
        #         "LDL": 165.0,
        #         "SGOT": 17.0,
        #         "SGPT": 14.0,
        #         "Alk": 172.0,
        #         "GGT": 17.0,
        #         "Alb.s": 5.2,
        #         "HCVAb": 1.0,
        #         "Thyroid_activity": 2.0
        #     },
        # ]



        columns = [
            "Age",
            "Gender (female)",
            "Nationality",
            "BMI",
            "History.of.hbv.vaccine",
            "Diabetes",
            "Hypertension",
            "Fatty.Liver",
            "High.blood.fats",
            "FBS",
            "TG",
            "Chol",
            "HDL",
            "LDL",
            "SGOT",
            "SGPT",
            "Alk",
            "GGT",
            "Alb.s",
            "HCVAb",
            "Thyroid_activity",
        ]



        labels = np.array([
            'LE homogene',
            'LE grade 1',
            'LE grade 2',
        ])


        modelInput = pd.DataFrame.from_dict(inputs).astype(float)

        # modelInput = modelInput.reindex(columns, axis=1)

        predictions = self.model.predict(modelInput)
        # print(predictions)

        labelProbabilities = [dict(zip(labels, [str(p) for p in prediction])) for prediction in predictions]
        # print(labelProbabilities)

        labels = labels[np.argmax(predictions, axis=1)]
        # print(labels)

        return {'labels': labels.tolist(), 'labelProbabilities': labelProbabilities}
