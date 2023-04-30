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
    __user_input_tfidf = None
    __response = None

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

            #response
            self.__response = dict()
        except Exception as e:
            msg = self.build_error_msg("The following error occured while trying to initialize Sentiment(): "+str(e))
            return msg
    
    def transform_input(self, user_input: str) -> None:
        """
        As the name suggests, this method makes use of the vectorizer, to transform the input to sparse matrix representation of term frquency - inverse document frequency.
        """
        try:
            self.__user_input_tfidf = self.__vectorizer.transform([user_input]) #We represent the user input in sparse matrix.
        except Exception as e:
            msg = self.build_error_msg("The following error occured while trying to transform the user input: "+str(e))
            return msg

    def make(self) -> None:
        """
        The make method, makes predictions, and assigns attributes their respective values.
        """
        try:
            probabilities = self.__model.predict_proba(self.__user_input_tfidf)
            self.__postivie_proba = probabilities[0][1]
            self.__negative_proba = probabilities[0][0]
            log_probabilities = self.__model.predict_log_proba(self.__user_input_tfidf)
            self.__postivie_log_proba = log_probabilities[0][1]
            self.__negative_log_proba = log_probabilities[0][0]
            self.__prediction = self.__model.predict(self.__user_input_tfidf)[0]
        except Exception as e:
            msg = self.build_error_msg("The following err occured while trying to make predictions: "+str(e))
            return msg

    def build_response(self, user_input) -> dict:
        """
        Builds a JSON response and returns it..
        """
        try:
            self.__response.clear()
            self.__response["Developer"] = "Harsha Vardhan Khurdula"
            self.__response["Status"] = "Success"
            self.__response["Text"] = user_input
            self.__response["Positive Class"] = self.__postivie_proba
            self.__response["Negative Class"] = self.__negative_proba
            self.__response["Positive Class Log"] = self.__postivie_log_proba
            self.__response["Negative Class Log"] = self.__negative_log_proba
            self.__response["Prediction"] = self.__prediction

            if self.__prediction == 0:
                self.__response["Label"] = "Negative"
            else:
                self.__response["Label"] = "Positive"

            return self.__response
        except Exception as e:
            msg = self.build_error_msg("The following error occured while building a JSON response: "+str(e))
            return msg
        
    def get_sentiment(self, user_input: str) -> dict:
        """
        Get sentiment is the core method of the Sentiment module, that receives the user input text, and yields the output.
        """
        try:
            self.transform_input(user_input=user_input)
            self.make()
            return self.build_response(user_input=user_input)
        except Exception as e:
            msg = self.build_error_msg("The following error occured while trying to determine sentiment of your input: "+str(e))
            return msg


    def build_error_msg(self, msg: str) -> dict:
        """
        Accepts a string that explains an error that might have occured while performing the task and returns a json reponse.
        """
        try:
            error_msg = dict()
            error_msg["Developer"] = "Harsha Vardhan Khurdula"
            error_msg["Status"] = "Failed"
            error_msg["Error"] = msg

            return error_msg

        except Exception as e:
            error_msg = dict()
            error_msg["Developer"] = "Harsha Vardhan Khurdula"
            error_msg["Status"] = "Failed"
            error_msg["Error"] = "The following error occured while trying to build an error reponse: "+str(e)
            return error_msg

obj = Sentiment()
while(True):
    print(obj.get_sentiment(input("Statement: ")))