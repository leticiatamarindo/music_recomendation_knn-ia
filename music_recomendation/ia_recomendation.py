import os
from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import joblib

app = Flask(__name__)

# Diretório atual do script Flask
base_dir = os.path.abspath(os.path.dirname('datasettest.csv'))

# Caminho absoluto para o arquivo datasettest.csv dentro do diretório base
file_path = os.path.join(base_dir, 'datasettest.csv')

# Carregar o modelo KNN e objetos relacionados
def carregar_modelo_knn():
    knn = joblib.load(os.path.join(base_dir, 'models/knn_model.pkl'))
    scaler = joblib.load(os.path.join(base_dir, 'models/scaler.pkl'))
    pca = joblib.load(os.path.join(base_dir, 'models/pca_model.pkl'))
    return knn, scaler, pca

def clean_numeric_values(df):
    for column in df.select_dtypes(include=['object']).columns:
        try:
            df[column] = df[column].str.replace(',', '.').astype(float)
        except ValueError:
            continue  # Ignora colunas que não podem ser convertidas em float
    return df

# Rota para a página inicial
@app.route("/")
def homepage():
    return render_template("telainicial.html")

# Rota para a página de recomendação
@app.route("/knnrecomendation")
def knnrecomendation():
    return render_template("index.html")

# Rota para processar os resultados da recomendação
@app.route("/resultados", methods=["POST"])
def resultados():
    # Verificar se o arquivo datasettest.csv existe
    if not os.path.isfile(file_path):
        return render_template("resultados.html", mensagem="Erro: arquivo datasettest.csv não encontrado.")

    # Carregar os dados e modelos necessários
    knn, scaler, pca = carregar_modelo_knn()

    # Carregar o dataframe do CSV
    df = pd.read_csv(file_path, decimal=',')

    # Limpar formatos numéricos inconsistentes
    df = clean_numeric_values(df)

    # Obter o input do usuário
    track_name = request.form.get('track_name')
    artist_name = request.form.get('artist_name')

    # Verificar se o input do usuário está vazio
    if not track_name or not artist_name:
        return render_template("resultados.html", mensagem="Erro: Nome da música e artista são obrigatórios.")

    # Recomendação de músicas (limitada a 10 resultados)
    recomendacoes = recomendar_musicas(track_name, artist_name, df, knn, scaler, pca, n_results=10)

    if recomendacoes.empty:
        return render_template("resultados.html", mensagem="Desculpe, não encontramos a música especificada.")

    # Renderizar o template com as recomendações
    return render_template("resultados.html", recomendacoes=recomendacoes.to_dict(orient='records'))

# Função para recomendar músicas com base no modelo KNN
def recomendar_musicas(track_name, artist_name, df, knn, scaler, pca, n_results=10):
    track_name = track_name.strip().lower()
    artist_name = artist_name.strip().lower()
    
    # Encontrar o track_id da música baseado no nome da música e do artista
    mask = (df['track_name'].str.lower() == track_name) & (df['artists'].str.lower().str.contains(artist_name))
    filtered_df = df[mask]
    
    if filtered_df.empty:
        return pd.DataFrame()  # Retornar DataFrame vazio se não encontrar resultados

    track_id = filtered_df['track_id'].values[0]
    
    # Obter características da música base
    track_features = df[df['track_id'] == track_id][['track_genre', 'popularity']]
    
    # Normalizar as características
    track_features_scaled = scaler.transform(track_features)
    
    # Aplicar PCA às características da música base
    track_features_pca = pca.transform(track_features_scaled)
    
    # Encontrar as músicas mais próximas
    distances, indices = knn.kneighbors(track_features_pca)
    
    # Retornar as músicas recomendadas e remover duplicatas
    recomendacoes = df.iloc[indices[0]].drop_duplicates(subset=['track_name', 'artists']).head(n_results)
    return recomendacoes

if __name__ == "__main__":
    app.run(debug=True)
