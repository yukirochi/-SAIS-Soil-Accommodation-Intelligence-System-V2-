from anfis_toolbox import ANFISRegressor
import numpy as np

class sais:

    def __init__(self, model_url = 'model.json'):
         self.model = ANFISRegressor.load(model_url)
    def predict(self, moisture, temperature, humidity):
         model = self.model
         input = np.array([[moisture,temperature,humidity]], dtype=float)
         result = float(model.predict(input)[0])

         return max(0.0, result)



model = sais()
print(model.predict(moisture=15.0,temperature=31.0,humidity=68.0))