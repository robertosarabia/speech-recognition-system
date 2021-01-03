import tensorflow.keras as keras


MODEL_PATH = "model.h5"

class _Keyword_Spotting_Service:

    model = None
    _mappings = [
        "down",
        "go",
        "left",
        "no",
        "off",
        "on",
        "right",
        "stop",
        "up",
        "yes",
        "zero"
    ]
    _instance = None


    def predict(selfself, file_path):

        # extract MFCCs

        # convert 2d MFCCs array into 4d array

        # make prediction


# elegant way of implementing singleton class in Python
# checks if instance is available and loads otherwise creates new using MODEL_PATH
def Keyword_Spotting_Service():

    # ensure that we only have 1 instance of KSS
    if _Keyword_Spotting_Service._instance is None:
        _Keyword_Spotting_Service._instance = _Keyword_Spotting_Service
        Keyword_Spotting_Service.model = keras.models.load_model(MODEL_PATH)
    return _Keyword_Spotting_Service._instance