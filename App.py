from flask import Flask, request, render_template # type: ignore
from werkzeug.utils import escape # type: ignore
import pickle

vector=pickle.load(open("vectorizer.pkl",'rb'))
Model=pickle.load(open("finalized_model.pkl",'rb'))

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/prediction", methods=['GET','POST'])
def prediction():
    if request.method=="POST":
        news=str(request.form['news'])
        print(news)

        predict=Model.predict(vector.transform([news]))
        print(predict)
        if predict ==0:
            return render_template("prediction.html",prediction_text="The News is Real")
        else:
            return render_template("prediction.html",prediction_text="The News is Fake")
    else:
        return render_template("prediction.html")

if __name__=='__main__':
    app.run()