from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

model_path = './notebooks/modelo_final_knn.pkl'
model = joblib.load(model_path)

@app.route('/')
def home():
    # Carregue o modelo treinado
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtenha os dados de entrada para previsão
        data = request.get_json()

        # Verifique se os dados estão presentes e têm a estrutura correta
        if data is None or 'features' not in data:
            raise ValueError("Dados de entrada ausentes ou formato incorreto.")

        # Extraia as características e reordene conforme necessário
        features = pd.DataFrame([data['features']])
        predictions = model.predict(features)

        # Formate as previsões em um formato adequado para resposta
        output = {'predictions': predictions.tolist()}

        return jsonify(output)

    except Exception as e:
        # Em caso de erro, retorne uma resposta apropriada
        error_message = {'error': str(e)}
        return jsonify(error_message), 400
    
#ROTA DO SOBRE
@app.route('/sobre')
def sobre():
    return render_template('sobre.html')    

if __name__ == '__main__':
    app.run(port=5000)
