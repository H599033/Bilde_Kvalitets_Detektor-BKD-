from flask import Flask, render_template, Response
import time

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/stream')
def stream():
    def event_stream():
        while True:
            # Call your function that detects and saves cars
            # This is just an example, you'd want to replace this with the actual call
            msg = detect_and_save(image_path, output_path)
            yield msg # EKSEMPEL PÅ BRUK: yield 'A new car has been detected!\n\n'
                    # sett yield i funksjonen som skal sende ett bilobjekt her som skal vises på nettsiden. EKS: lagbil() { yield car}
            time.sleep(1)  # you can adjust this as needed

    return Response(event_stream(), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(debug=True, threaded=True)