import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go
import numpy as np

def get_clean_data():
        data = pd.read_csv('data/data.csv')
        data = data.drop('id', axis=1)
        return data

def get_scaled_values(input_dict):
        data = get_clean_data()
        X = data.drop(['diagnosis'], axis = 1)
        scaled_dict = {}
        for key, value in input_dict.items():
                max_value = X[key].max()
                min_value = X[key].min()
                scaled_values = (value - min_value)/(max_value - min_value)
                scaled_dict[key] = scaled_values
                
        return scaled_dict

def add_sidebar():
        st.sidebar.header('Cell Nucluei Measurements')
        data = get_clean_data()
        
        slider_labels = [
                        ("Radius (mean)", "radius_mean"),
                        ("Texture (mean)", "texture_mean"),
                        ("Perimeter (mean)", "perimeter_mean"),
                        ("Area (mean)", "area_mean"),
                        ("Smoothness (mean)", "smoothness_mean"),
                        ("Compactness (mean)", "compactness_mean"),
                        ("Concavity (mean)", "concavity_mean"),
                        ("Concave points (mean)", "concave points_mean"),
                        ("Symmetry (mean)", "symmetry_mean"),
                        ("Fractal dimension (mean)", "fractal_dimension_mean"),
                        ("Radius (se)", "radius_se"),
                        ("Texture (se)", "texture_se"),
                        ("Perimeter (se)", "perimeter_se"),
                        ("Area (se)", "area_se"),
                        ("Smoothness (se)", "smoothness_se"),
                        ("Compactness (se)", "compactness_se"),
                        ("Concavity (se)", "concavity_se"),
                        ("Concave points (se)", "concave points_se"),
                        ("Symmetry (se)", "symmetry_se"),
                        ("Fractal dimension (se)", "fractal_dimension_se"),
                        ("Radius (worst)", "radius_worst"),
                        ("Texture (worst)", "texture_worst"),
                        ("Perimeter (worst)", "perimeter_worst"),
                        ("Area (worst)", "area_worst"),
                        ("Smoothness (worst)", "smoothness_worst"),
                        ("Compactness (worst)", "compactness_worst"),
                        ("Concavity (worst)", "concavity_worst"),
                        ("Concave points (worst)", "concave points_worst"),
                        ("Symmetry (worst)", "symmetry_worst"),
                        ("Fractal dimension (worst)", "fractal_dimension_worst")
    ]
        
        input_dict= {}        
        for label, key in slider_labels:
               input_dict[key] = st.sidebar.slider(   
                        label=label,
                        min_value=float(0),#data[key].min(0),
                        max_value=data[key].max(),
                        value=(data[key].mean()),
                        key=key
                )
        return input_dict

def add_predictions(input_data):
        model = pickle.load(open('model/model.pkl', 'rb'))
        scaler = pickle.load(open('model/scaler.pkl', 'rb'))
        
        input_array = np.array(list(input_data.values())).reshape(1, -1)
        input_array_scaled = scaler.transform(input_array)
        
        prediction = model.predict(input_array_scaled)
        
        st.subheader('Cell Cluster Prediction')
        
        st.write('The cell cluster is : ')
        
        if prediction[0] == 0:
                st.markdown('<p style="color:green; font-size:20px;">Benign</p>', unsafe_allow_html=True)
        else:
                st.markdown('<p style="color:red; font-size:20px;">Malicious</p>', unsafe_allow_html=True)
                
        st.write('The Probability of being benign: ', model.predict_proba(input_array_scaled)[0][0])
        st.write('The Probability of being Malicious: ', model.predict_proba(input_array_scaled)[0][1])
       
        st.write('This app is only used to assist professional in making a diagnosis, but not be used as a substitute for a pressional diagnosisi')

def get_radar_chart(input_data):
        input_data = get_scaled_values(input_data)
        catgories  = [  'Radius',
                        'Texture',
                        'Perimeter',
                        'Area',
                        'Smoothness',
                        'Compactness',
                        'Concavity',
                        'Concave',
                        'Symmetry',
                        'Fractal', 
                        ]
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
                r = 
                [
                     input_data['radius_mean'],
                     input_data['texture_mean'],
                     input_data['perimeter_mean'],
                     input_data['area_mean'],
                     input_data['smoothness_mean'],
                     input_data['compactness_mean'],
                     input_data['concavity_mean'],
                     input_data['concave points_mean'],
                     input_data['symmetry_mean'],
                     input_data['fractal_dimension_mean']
                ],
                theta = catgories, 
                fill = 'toself',
                name = 'Mean Value'
        ))
        fig.add_trace(go.Scatterpolar(
                r = 
                [
                     input_data['radius_se'],
                     input_data['texture_se'],
                     input_data['perimeter_se'],
                     input_data['area_se'],
                     input_data['smoothness_se'],
                     input_data['compactness_se'],
                     input_data['concavity_se'],
                     input_data['concave points_se'],
                     input_data['symmetry_se'],
                     input_data['fractal_dimension_se']
                ],
                theta = catgories, 
                fill = 'toself',
                name = 'Standard Error'
        ))
        
        fig.add_trace(go.Scatterpolar(
                r = 
                [
                     input_data['radius_worst'],
                     input_data['texture_worst'],
                     input_data['perimeter_worst'],
                     input_data['area_worst'],
                     input_data['smoothness_worst'],
                     input_data['compactness_worst'],
                     input_data['concavity_worst'],
                     input_data['concave points_worst'],
                     input_data['symmetry_worst'],
                     input_data['fractal_dimension_worst']
                ],
                theta = catgories, 
                fill = 'toself',
                name = 'Worst Value'
        ))
        
        fig.update_layout(
                polar=dict(
                        radialaxis=dict(
                                visible=True,
                                range=[0, 1]
                        )
                ),
                showlegend=True
        )
        
        return fig

def main():
        st.set_page_config(
                page_title="Breast Cancer Prediction",
                page_icon=":syringe:",
                layout="wide", 
                initial_sidebar_state="expanded"
                )
        
        input_data  = add_sidebar()
        #st.write("The input data is", input_data)
        
        with st.container():
                st.title("Breast Cancer Prediction")
                st.write("""
                         Welcome to Breast Cancer Prediction. 
                         This app predicts if a patient has breast cancer or not. 
                         Please connect this app to your cytology lab to help diagnose brerast cancer form your tissue sample. The app predicts using machine learning models. 
                """)
        
        col1, col2 = st.columns([4, 1])
        with col1:
                radar_chart = get_radar_chart(input_data)
                st.plotly_chart(radar_chart)
        with col2:
                add_predictions(input_data)


if __name__ == '__main__':
        main()