import uvicorn
from flask import Flask, jsonify, request
import pandas as pd
import pickle
import numpy as np
import pandas as pd 
from lightgbm import LGBMClassifier
app = Flask(__name__)
def load_data():
    
    # Données générales
   
    initial_data = pd.read_csv('X_test_clean.csv', index_col='sk_id_curr')
 
    return initial_data 

def load_model():

    '''loading the trained model'''
    pickle_in = open('model_LGBM.pkl', 'rb') 
    clf = pickle.load(pickle_in)
    
    return clf

def load_customer_score(data, customer_id, model):  
    
    customer_score = model.predict_proba(data[data.index == int(customer_id)])[:,1]        
    return  customer_score[0]
    

initial_data= load_data()
model = load_model()
customer_ids = initial_data.index.values



@app.route('/get_score_credit', methods=['GET'])
def get_score_credit():
    """Récupération du score client à partir de son identifiant.
    Score client = probabilité d'être insolvable.
    """
    if 'id_client' in request.args:
       id_client = int(request.args['id_client'])
    else:
       return "Error: No id field provided. Please specify an id."
    print("id_client: ", id_client)
    score = load_customer_score(initial_data, id_client, model)
    #return{"score ": round(score, 2)}
    
    # renvoyer la prediction 
    
    dict_final = {
                  'score' :round(score*100,2)
    }             
    
    return jsonify(dict_final)

    
#lancement de l'application
if __name__ == "__main__":
    app.debug = True
    app.run()
    print("api start ! ")

