from flask import Flask, request, render_template

from src.pipeline.predict_pipeline import CustomData, PredictPipeline


application = Flask(__name__)       # Entry point

app = application

## Route for homepage

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictcharges', methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data = CustomData(
            age=request.form.get('age'),
            sex=request.form.get('sex'),
            bmi=float(request.form.get('bmi')),
            children=request.form.get('children'),
            smoker=request.form.get('smoker'),
            region=request.form.get('region')
        )

        pred_df = data.get_data_as_dataframe()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template('home.html', results=results[0])
    

if __name__=="__main__":
    app.run(host="0.0.0.0", debug=True)