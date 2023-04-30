from flask import Flask, request, jsonify
from joblib import load
from Sentiment import Sentiment

app = Flask(__name__)

# Initialize the Sentiment class
sentiment_obj = Sentiment()

@app.route('/sentiment', methods=['POST'])
def get_sentiment():
    
    data = request.get_json()

    
    text = data['text']

    
    response = sentiment_obj.get_sentiment(text)
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
