import pandas as pd
from flask import Flask, request, render_template
import joblib
import numpy as np
import pickle
#import MySQLdb

#db = MySQLdb.connect(host="localhost",user="root",password="your_password",db="chronic_disease",charset="utf8")


app= Flask(__name__)

model = pickle.load(open('ckd_lightgbm_model.pickle', 'rb'))

@app.route('/')
def home():
    return render_template('h1.html')


@app.route('/go_ahead')
def go_ahead():
    return render_template('calculation.html')

@app.route('/predict',methods=['POST'])
def predict():

    if request.method == 'POST':

        age = float(request.form['age'])
        al = float(request.form['albumin'])
        bg = float(request.form['bloodglucose'])
        sc = float(request.form['serumcreat'])
        hemo = float(request.form['hemoglobin'])

        data = np.array([[age, al, bg, sc, hemo]])
        my_prediction = model.predict(data)
        output = my_prediction[0]

        if output == 'ckd':
            res_val = " have  CKD "

        else:
            res_val = "do not have CKD"
        
        #cursor = db.cursor()
        #cursor.execute("""INSERT INTO ckd_data(age,al,bg,sc,hemo,class)values(%s,%s,%s,%s,%s,%s)""",(age,al,bg,sc,hemo,output)) 
        #db.commit()
        #cursor.close()
        #print("data inserted successfully!!1")
        return render_template('calculation.html', prediction_text='Patient {}'.format(res_val))
    else:
        print("something went wrong")

@app.route('/calc',methods=['POST'])
def calc():
        if request.method == 'POST':

            age = int(request.form['age'])
            gender = int(request.form['gender'])
            race = int(request.form['race'])
            scr = float(request.form['serum'])

            if(gender==0 and race==0):
                k=0.7
                alpha = -0.329
                gfr = 141 * min(scr/k,1)**alpha * max(scr/k,1)**-1.209 * 0.993**age * 1.018

            elif(gender==0 and race==1):
                k=0.7
                alpha = -0.329
                gfr = 141 * min(scr/k,1)**alpha * max(scr/k,1)**-1.209 * 0.993**age * 1.018 * 1.159


            elif(gender==1 and race==0):
                k=0.9
                alpha = -0.411
                gfr = 141 * min(scr/k,1)**alpha * max(scr/k,1)**-1.209 * 0.993**age
            else:
                k=0.9
                alpha = -0.411
                gfr = 141 * min(scr/k,1)**alpha * max(scr/k,1)**-1.209 * 0.993**age * 1.159

        gfr_value = int(gfr)

        if(gfr>90):
            msg ="Normal condition!!"

        elif(gfr>=60 and gfr<=89 ):
            msg = " \nSorry! you are suffering from stage 2 CKD condition"

        elif(gfr>=45 and gfr<=59 ):
            msg ="Sorry! you are suffering from stage 3A CKD condition"

        elif(gfr>=30 and gfr<=44 ):
            msg =("Sorry! you are suffering from stage 3B CKD condition")

        elif(gfr>=15 and gfr<=29 ):
            msg ="Sorry! you are suffering from stage 4 Severe CKD condition"

        else:
            msg ="Sorry! you are suffering from stage 5 end stage CKD condition"

        return render_template('calculation.html', pred='Your GFR value is : {}{}'.format(gfr_value,msg))

# ===============================================================================================
if __name__ == "__main__":
    app.run(debug=True)

