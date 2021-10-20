import streamlit as st
import pandas as pd
import os
import main


menu = ["Home", "Dataset", "Model", "Cuteness Meter"]

choice = st.sidebar.selectbox(label = 'Home', options = menu)

if choice == "Home":
    title1 = "-------------Pawpularity--------------"
    title2 = "The Super-duper Cuteness Meter"
    st.markdown(f"<h1 style='text-align: center; color: white;'>{title1}</h1>", unsafe_allow_html=True)
    st.markdown(f"<h1 style='text-align: center; color: white;'>{title2}</h1>", unsafe_allow_html=True)
    title_alignment = "A picture is worth a thousand words. But did you know a picture can save a thousand lives? Millions of stray animals suffer on the streets or are euthanized in shelters every day around the world. We might expect pets with attractive photos to generate more interest and be adopted faster. But what makes a good picture? With the help of data science, you may be able to accurately determine a pet photo’s appeal and even suggest improvements to give these rescue animals a higher chance of loving homes."
    st.markdown(f"<h3 style='text-align: justify; color: white;'>{title_alignment}</h3>", unsafe_allow_html=True)
    st.write("")
    col1, col2, col3 = st.columns([1, 6, 1])
    with col2:
        st.image("title.gif.opdownload",
             use_column_width = 'auto')

elif choice == "Dataset":
    title1 = "-------------Dataset--------------"
    st.markdown(f"<h1 style='text-align: center; color: white;'>{title1}</h1>", unsafe_allow_html=True)
    st.markdown("""
    This data is taken from the "PetFinder.my - Pawpularity Contest"
    * Data source: [https://www.kaggle.com/c/petfinder-pawpularity-score/data](https://www.kaggle.com/c/petfinder-pawpularity-score/data)
    """)
    col1, col2, col3 = st.columns([1, 6, 1])
    with col2:
        st.image("dog.gif.opdownload",
                 use_column_width='auto')
    df = pd.read_csv('petfinder-pawpularity-score/train.csv')
    st.dataframe(df)

elif choice == "Model":
    title1 = "-------------Model--------------"
    st.markdown(f"<h1 style='text-align: center; color: white;'>{title1}</h1>", unsafe_allow_html=True)
    title2 = "Why chose this model?"
    st.markdown(f"<h2 style='text-align: left; color: white;'>{title2}</h2>", unsafe_allow_html=True)
    st.image("0_09AED_CjE-PUFxKC.png",
             use_column_width='auto')
    title3 = "Model Architectural detail:"
    st.markdown(f"<h2 style='text-align: left; color: white;'>{title3}</h2>", unsafe_allow_html=True)
    st.image("The-EffecientNet-B0-general-architecture.png",
             use_column_width='auto')

elif choice == "Cuteness Meter":
    title1 = "-------------Cuteness Meter--------------"
    st.markdown(f"<h1 style='text-align: center; color: white;'>{title1}</h1>", unsafe_allow_html=True)

    def file_selector(folder_path='.'):
        filenames = os.listdir(folder_path)
        selected_filename = st.selectbox('Select a file', filenames)
        return os.path.join(folder_path, selected_filename)


    if st.checkbox('Select a file in current directory'):
        folder_path = 'test_image'
        if st.checkbox('Change directory'):
            folder_path = st.text_input('Enter folder path', '.')
        filename = file_selector(folder_path=folder_path)
        st.write('You selected `%s`' % filename)
        st.image(filename, use_column_width='auto')
        user_input = st.text_area("Number of images you want to train with: (Max = 9000)")
        user_input = int(user_input)
        if st.button('Processing'):
            data = {'Id': [filename], 'Pawpularity': [0]}
            df = pd.DataFrame(data)
            preds_final, final_all_oof_score = main.Pawpularity_Caculation(test_df_2 = df, num_image= user_input)
            st.markdown(f"<h3 style='text-align: left; color: white;'>The Pet's Pawpularity = {preds_final[0]} </h3>", unsafe_allow_html=True)
            st.markdown(f"<h3 style='text-align: left; color: white;'>The RMSE Score = {final_all_oof_score}</h3>", unsafe_allow_html=True)


