import streamlit as st
import tensorflow as tf
import numpy as np

# 1. Título e Descrição no Streamlit
st.title("Previsor de Horas de Estudo para o ENEM")
st.write("Um modelo de Inteligência Artificial simples usando TensorFlow para prever quantas horas você precisa estudar para alcançar a nota desejada.")

# 2. Dados de Treinamento (ENEM)
notas = np.array([5.0, 6.0, 7.0, 8.0, 9.0], dtype=float)
horas = np.array([2.0, 4.0, 6.0, 8.0, 10.0], dtype=float)

# 3. Criando e Treinando a Rede Neural
@st.cache_resource
def treinar_modelo():
    modelo = tf.keras.Sequential([
        tf.keras.layers.Dense(units=1, input_shape=[1])
    ])
    modelo.compile(optimizer='sgd', loss='mean_squared_error')
    modelo.fit(notas, horas, epochs=300, verbose=0)
    return modelo

modelo = treinar_modelo()

# 4. Interface do Usuário (Entrada de Dados)
st.subheader("Faça sua simulação")
nota_desejada = st.slider("Qual nota você quer tirar no ENEM?", min_value=1.0, max_value=10.0, value=10.0, step=0.5)

# 5. Fazendo a Previsão
if st.button("Calcular Horas de Estudo"):
    previsao = modelo.predict(np.array([nota_desejada], dtype=float))
    horas_previstas = previsao[0][0]
    st.success(f"Para tirar **{nota_desejada}**, a IA prevê que você precisará estudar cerca de **{horas_previstas:.1f} horas**.")
    