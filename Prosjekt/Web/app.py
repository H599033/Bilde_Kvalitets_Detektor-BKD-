from flask import Flask, render_template, jsonify, send_file
import os
import pickle
import DbService 
import os



app = Flask(__name__, static_folder='Prosjekt')

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
            'orginal_bilder': ['/image/' + img for img in bil.orginal_bilder],
            'redigerte_bilder': ['/image/' + img for img in bil.redigerte_bilder]
        })
    
    return jsonify(data)
#TODO kanskje legge til egen DbService funksjon
@app.route('/image/<path:filename>')
def serve_image(filename):
    base_dir = os.path.abspath(os.path.dirname(__file__))
    base_dir = base_dir.replace('\\Prosjekt', '')  # remove '\Prosjekt' from the base_dir
    base_dir = base_dir.replace('\\Web', '')  # remove '\\Web' from the base_dir
    image_path = os.path.join(base_dir, filename)
    return send_file(image_path, mimetype='image/png')

@app.route('/css/<path:filename>')
def serve_css(filename):
    return send_file(os.path.join('templates', filename), mimetype='text/css')

@app.route('/logo/<path:filename>')
def serve_logo(filename):
    return send_file(os.path.join('templates', filename), mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
