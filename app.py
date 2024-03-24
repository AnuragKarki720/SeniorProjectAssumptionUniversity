from flask import Flask, request
import webview

# Create a new Flask app
app = Flask(__name__)
window = webview.create_window('LifeFactors', app)
# Import route definitions from diabetesapp.py
from appdiabetes import *




from appdiabetes2022 import *
# Import route definitions from appheart.py
from appHeart import *

# Import route definitions from appheartblood.py
from appheartblood import *

if __name__ == "__main__":
    # Run the Flask app
    #app.run(debug=True)
    webview.start()
