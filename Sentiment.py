
class Sentiment():
    """
    This module is simple as the name suggests,
    Sentiment, allows a user to provide text in English, and determines the probabilities, for the text provided to be negative or positive
    in terms of Sentiment.

    However, this module only captures, Either Negative or Positive, there is no Nuetral label!
    """

    def __init__(self) -> None:
        """
        Constructor is responsible for loading the following:
        Tf-idf vectorizer: this was trained and saved previously.
        LogisticRegression: A saved instance of LR that was fitted to Sentiment data.
        """
        try:
            pass
        except Exception as e:
            print("[ERR] The following error occured while trying to initialize Sentiment(): "+str(e))