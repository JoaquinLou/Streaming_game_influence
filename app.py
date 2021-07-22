import streamlit as st
import pandas as pd
import numpy as np
import joblib, os #Reading machine learning models
import matplotlib.pyplot as plt
import matplotlib.dates as mdates #Customize seaborn xtickers 
import seaborn as sns
import altair as alt
from sklearn import datasets
from sklearn.model_selection import train_test_split # data split

#Page config
st.set_page_config(
	page_title = 'KNN Regression App'
	)

#Data reading
@st.cache
def load_data(dataframe):
	data = pd.read_csv(dataframe)
	return data

#Data load
df = load_data('Total data stream and games.csv')
df_twitch_proc = load_data('Twitch data processed.csv')
df_steam = load_data('Steam data processed.csv')



# Loading Machine Learning Models
def load_prediction_model(model_file):
	loaded_model = joblib.load(open(os.path.join(model_file),"rb"))
	return loaded_model


#Main panel
def main():
	st.markdown("## KNN Regression App - Joaquin Lou 🎮🕹")
	

	st.header('Intro')
	st.write("This app is developed for the Data Science TFM at KSchool. Here you will be able to see the data of the close relationship between the players of the biggest PC gaming platform (Steam) and the biggest live video streaming platform (Twitch)")
	st.write("The data was collected in April 2021 from the following repositories that collect historical data from the platforms.")
	st.write('''
		- Steam data: https://www.kaggle.com/michau96/popularity-of-games-on-steam 
		- Twitch data: https://www.kaggle.com/rankirsh/evolution-of-top-games-on-twitch

		''')

	st.write("---")
	st.header('Evolution of data on the two platforms')

	st.write("First of all, we have to see and plot our data into different graphs to know the evolution of the different platforms.")
	st.write("We will start with Twitch, the most important streaming platform on Internet")



	#Twitch graph of hours watched
	st.subheader('Twitch - Chart with total hours watched by year')
	st.write("On this chart we will see the evolution of hours viewed in the platform")
	
	alt_plot1 = alt.Chart(df_twitch_proc).mark_line().encode(
		alt.X("yearmonthdate(date):T", axis=alt.Axis(title='Date')),
		alt.Y("sum(Hours_watched):Q", axis=alt.Axis(title='Hours watched'))
		).properties(
		title='Hours watched by date',
		width=650,
		height=400
		).interactive()

	st.altair_chart(alt_plot1)
	
	st.write("We could see that 2020 doesn't follow the normal trend right? Well... let's see other graph before ")



	#Twitch graph with the evolution of Streamers
	st.subheader('Twitch - Chart with total Streamers by year')
	st.write("On this other chart we will plot the evolution of the people streaming on this platform. This people are called Streamers and share/create content to entretain other users")

	alt_plot2 = alt.Chart(df_twitch_proc).mark_line().encode(
		alt.X("yearmonthdate(date):T", axis=alt.Axis(title='Date')),
		alt.Y("sum(Streamers):Q", axis=alt.Axis(title='Streamers'))
		).properties(
		title='Streamers by date',
		width=650,
		height=400
		).interactive()

	st.altair_chart(alt_plot2)

	st.write("Hmmm remember the previous spike in 2020, here it is again.")
	st.write("The reason of that spikes in hte number of hours viewed an the number of people streaming is the global pandemic. Restrictions augmented the number of hours people spend in their houses so this is perfect to the platforms which offers entretainment.")



	#Comparison between hours watched and streamers through years
	fig3, ax = plt.subplots(figsize=(10,5))

	grid = sns.FacetGrid(df_twitch_proc, col = "Year", hue = "Year", col_wrap=6)
	grid.map(sns.scatterplot, "Streamers", "Hours_watched")
	grid.add_legend()
	st.set_option('deprecation.showPyplotGlobalUse', False)
	st.pyplot()

	st.write("In these graphs we can see the evolution of the relationship between the variables of hours viewed and streamers. We can see that as the years go by, the points are separating from the 0,0 axis, which indicates that there are more users using the platform. Both consuming and creating content.")



	#Scatterplot variables Twitch
	st.write("If we look at the data from another perspective, we can confirm what we have already mentioned.")

	st.image(
		"Components_streamlit/Twitch-Streamers_Avgviewers_year.png",
		caption='Twitch scatterplot'
		)
	


	#Top games viewed Twitch
	st.write("With the above data, we are going to extract the most played games of the last years with Tableau's tool.")

	st.image(
		"Components_streamlit/Games_watched.png",
		caption='Top games watched on Twitch'
		)
	
	st.write("It's remarkable that the Just Chatting channel which is used to talk live Streamers with users has increased since covid-19. The rest of the games also have an increase since covid-19 and it is because of this change of trend that I have decided to use two machine learning algorithms.")

	st.write("---")


	#Graph Steam players
	st.subheader('Steam - Chart with total Players by year')
	st.write("In this graph we are going to plot the evolution of the number of players Steam has had until the year 2021.")
	fig, ax = plt.subplots(figsize=(10,5))

	alt_plot3 = alt.Chart(df_steam).mark_line().encode(
		alt.X("yearmonthdate(date):T", axis=alt.Axis(title='Date')),
		alt.Y("sum(avg):Q", axis=alt.Axis(title='Avg players'))
		).properties(
		title='Avg players by date',
		width=650,
		height=400
		).interactive()

	st.altair_chart(alt_plot3)



	#Top games played Steam
	st.write("Now let's use Tableau to see the evolution of the most played games and their evolution during the last years.")

	st.image(
		"Components_streamlit/Games_played.png",
		caption='Top games played on Steam'
		)
	
	st.write("We can appreciate that the rise in the number of players in 2018 is due to the PLAYERUNKNOWN'S BATTLEGROUNDS game. It established a new form of gameplay called Battle Royale along with Fortnite.")
	
	st.write("---")



	# Machine learning section
	st.header('Machine Learning - KNN Regressor')
	st.write('''
		Due to what we have seen in our data it has been decided to develop a predictive model for the two trends.
		One prediction model for the total data and another prediction model for the new evolution after the pandemic.
		Our models are limited to a max of 5000 average viewers per month due to our data'''
			)

	#Model selection

	model_selected = ['Select data', 'Prediction based on total data', 'Prediction based on data since covid']
	choice = st.selectbox("Choose your data for prediction model", model_selected)




# Prediction with knn (k=43) and total data
	if choice == 'Prediction based on total data':

		st.subheader("Average of viewers expected per month")

		expected_viewers = st.slider("Number of average month viewers dou you expect to have", 0, 5000)

		if st.button("Calculate"):

			knn_regressor = load_prediction_model("Models/knn_regression_viewers.pkl")
			expected_viewers_reshaped = np.array(expected_viewers).reshape(-1,1)

			predict_players = knn_regressor.predict(expected_viewers_reshaped)

			st.success("With {} viewers on your game you should expect to have {} players per month".format(expected_viewers,(predict_players[0].round(1))))
			
			st.markdown("## Thanks for using this app! 🎮🕹")



# Prediction with knn (k=5) and data since covid-19
	if choice == 'Prediction based on data since covid':

		st.subheader("Average of viewers expected per month")

		expected_viewers_2020 = st.slider("Number of average month viewers do you expect to have", 0, 5000)

		if st.button("Calculate"):

			knn_regressor_2020 = load_prediction_model("Models/knn_regression_viewers_2020.pkl")
			expected_viewers_reshaped_2020 = np.array(expected_viewers_2020).reshape(-1,1)

			predict_players_2020 = knn_regressor_2020.predict(expected_viewers_reshaped_2020)

			st.success("With {} viewers on your game you should expect to have {} players per month".format(expected_viewers_2020,(predict_players_2020[0].round(1))))

			st.markdown("## Thanks for using this app! 🎮🕹")

if __name__ == '__main__':
	main()
