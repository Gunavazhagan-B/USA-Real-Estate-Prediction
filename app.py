from flask import Flask,render_template,request
from src.pipeline.predict_pipeline import CustomData,predictPipeline
from chatbot import RealEstateChatbot
import numpy as np

bot=RealEstateChatbot()

app=Flask(__name__)

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('predict.html')
    else:
        form_data = request.form.to_dict()

        data = CustomData(
            brokered_by=form_data.get('brokered_by'),
            status=form_data.get('status'),
            bed=form_data.get('bed'),
            bath=form_data.get('bath'),
            acre_lot=form_data.get('acre_lot'),
            street=form_data.get('street'),
            city=form_data.get('city'),
            state=form_data.get('state'),
            zip_code=form_data.get('zip_code'),
            house_size=form_data.get('house_size')
        )

        df = data.get_data_as_data_frame()
        pred_pipeline = predictPipeline()
        result = pred_pipeline.predict_result(df)
        predicted_value = result[0]
        predicted_value = np.expm1(predicted_value)

        return render_template('predict.html', results=predicted_value, form_data=form_data)

    
@app.route('/chat',methods=['GET','POST'])
def chat():
    
    if request.method=='GET':
        return render_template('bot.html')
    else:
        question=request.form.get('question')
        response=bot.ask(question=question)
        return render_template('bot.html',response=response)

if __name__=='__main__':
    app.run(debug=True)