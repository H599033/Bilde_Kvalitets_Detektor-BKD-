from flask import Flask, render_template, jsonify
import os
import pickle
import DbService 
import os



app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/display')
def display():
    # Get all the Bil objects
    biler = DbService.hent_alle_biler()
    
    # Prepare the data for JSON
    data = []
    for bil in biler:
        data.append({
            'ID': bil.ID,
            'tid': bil.tid,
            'dato': bil.dato,
            'sted': bil.sted,
            'orginal_bilder': bil.orginal_bilder,
            'redigerte_bilder': bil.redigerte_bilder
        })
    
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)
