from joblib import load

class Sentiment():
    """
    This module is simple as the name suggests,
    Sentiment, allows a user to provide text in English, and determines the probabilities, for the text provided to be negative or positive
    in terms of Sentiment.

    However, this module only captures, Either Negative or Positive, there is no Nuetral label!
    """
    #Sentiment Attributes:
    __vectorizer = None
    __model = None
    __postivie_proba = None
    __negative_proba = None
    __postivie_log_proba = None
    __negative_log_proba = None
    __prediction = None
    def __init__(self) -> None:
        """
        Constructor is responsible for loading the following:
        Tf-idf vectorizer: this was trained and saved previously.
        LogisticRegression: A saved instance of LR that was fitted to Sentiment data.
        """
        try:
            self.__vectorizer = load("fitted_vectorizer.joblib")
            self.__model = load("logisitic_regression.joblib")

            #Initialize the attributes to their types.
            self.__postivie_proba = float()
            self.__negative_proba = float()
            self.__postivie_log_proba = float()
            self.__negative_log_proba = float()
        except Exception as e:
            print("[ERR] The following error occured while trying to initialize Sentiment(): "+str(e))

    def get_sentiment(self, user_input: str) -> dict:
        """
        Get sentiment is the core method of the Sentiment module, that receives the user input text, and yields the output.
        """
        try:
            user_input_transformed = self.__vectorizer.transform([user_input]) #We represent the user input in sparse matrix.
            print(user_input_transformed)
        except Exception as e:
            print("[ERR] The following error occured while trying to determine sentiment of your input: "+str(e))

obj = Sentiment()
obj.get_sentiment("Hello what is up my g?")