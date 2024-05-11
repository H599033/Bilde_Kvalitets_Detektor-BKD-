from flask import Flask, render_template, jsonify, send_file
import os
import sys
sys.path.append('Prosjekt/Web')
import DbService 

class Nettside():
    def start(self):
        """ Start web-applikasjonen.
        """
        app = Flask(__name__, static_folder='Prosjekt')
        
        @app.route('/')
        def home():
            """ Server hjemmesiden.

            Returns:
                index.html : HTML fil til nettsiden
            """
            return render_template('index.html')

        @app.route('/display')
        def display():
            """   Server JSON-data av alle Bil-objekter.

            Returns:
                data : Forbereder dataene i JSON-format, og returnerer dem som respons.
            """
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
                    'motion_blur': bil.motion_blur,
                    'lav_belysning' : bil.lav_belysning,
                    'vaatt_dekk' : bil.vaatt_dekk,
                    'original_bilder': ['/image/' + img for img in bil.original_bilder],
                    'korrigerte_bilder': ['/image/' + img for img in bil.korrigerte_bilder]
                })
            
            return jsonify(data)

        @app.route('/image/<path:filename>')
        def serve_image(filename):
            """Server bildefiler.

            Args:
                filename (path): path til bildefiler

            Returns:
                filename (path): Serverer bildefiler.
            """
            base_dir = os.path.abspath(os.path.dirname(__file__))
            base_dir = base_dir.replace(os.path.join(os.sep, 'Prosjekt'), '')  # remove '\Prosjekt' or '/Prosjekt' from the base_dir
            base_dir = base_dir.replace(os.path.join(os.sep, 'Web'), '')  # remove '\Web' or '/Web' from the base_dir
            image_path = os.path.join(base_dir, filename)
            return send_file(image_path, mimetype='image/png')

        @app.route('/css/<path:filename>')
        def serve_css(filename):
            """Serverer CSS fil

            Args:
                filename (path): path til CSS fil

            Returns:
                filename (path): Serverer CSS fil.
            """
            
            
            return send_file(os.path.join('templates', filename), mimetype='text/css')

        @app.route('/js/<path:filename>')
        def serve_js(filename):
            """Serverer JS fil

            Args:
                filename (path): path til JS fil

            Returns:
                filename (path): Serverer JS fil
            """
            return send_file(os.path.join('templates', filename), mimetype='application/javascript')

        @app.route('/logo/<path:filename>')
        def serve_logo(filename):
            """Serverer logo

            Args:
                filename (path): path til logo

            Returns:
                filename (path): Serverer logo
            """
            return send_file(os.path.join('templates', filename), mimetype='image/png')
        
        app.run(debug=True,use_reloader=False)