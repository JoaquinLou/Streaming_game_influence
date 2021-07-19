import streamlit as st
import numpy as np
import pandas as pd
import joblib, os #Reading machine learning models
import matplotlib.pyplot as plt
import matplotlib.dates as mdates #Customize seaborn xtickers 
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split # data split


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
@st.cache
def load_prediction_model(model_file):
	loaded_model = joblib.load(open(os.path.join(model_file),"rb"))
	return loaded_model


#Main panel
def main():
	st.title("KNN Regression App - KSchool")
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
	
	fig1, ax = plt.subplots(figsize=(10,5))

	data_twitch_hours = df_twitch_proc.groupby(['date'])['Hours_watched'].sum()
	ax = sns.lineplot(data=pd.DataFrame(data_twitch_hours), x="date", y="Hours_watched", linewidth=3)

	ax.xaxis.set_major_locator(mdates.AutoDateLocator()) #To select dates on the graph without overlapping
	plt.ylabel('Hours watched (mil)',fontsize=16)
	plt.yticks(fontsize=12)
	plt.xlabel('Date',fontsize=16)
	plt.title('Hours viewed on Twitch', fontsize= 16, weight='bold')
	plt.grid(True)

	st.pyplot(fig1)

	st.write("We could see that 2020 doesn't follow the normal trend right? Well... let's see other graph before ")


	#Twitch graph with the evolution of Streamers
	st.subheader('Twitch - Chart with total Streamers by year')
	st.write("On this other chart we will plot the evolution of the people streaming on this platform. This people are called Streamers and share/create content to entretain other users")

	fig2, ax = plt.subplots(figsize=(10,5))

	data_streamers = df_twitch_proc.groupby(['date'])['Streamers'].sum()
	ax = sns.lineplot(data=pd.DataFrame(data_streamers), x="date", y="Streamers", linewidth=3)

	sns.set_context("paper")
	ax.xaxis.set_major_locator(mdates.AutoDateLocator())
	plt.ylabel('Streamers (mil)',fontsize=16)
	plt.yticks(fontsize=12)
	plt.xlabel('Date',fontsize=16)
	plt.title('Streamers on Twitch', fontsize= 16, weight='bold')
	plt.grid(True)

	st.pyplot(fig2)

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


	#Graph Steam players
	st.subheader('Steam - Chart with total Players by year')
	st.write("In this graph we are going to plot the evolution of the number of players Steam has had until the year 2021.")
	fig, ax = plt.subplots(figsize=(10,5))

	data_steam_players = df_steam.groupby(['date'])['avg'].sum()
	ax = sns.lineplot(data=pd.DataFrame(data_steam_players), x="date", y="avg", linewidth=3)

	sns.set_context("paper")
	ax.xaxis.set_major_locator(mdates.AutoDateLocator())
	plt.ylabel('Players (mil)',fontsize=16)
	plt.yticks(fontsize=12)
	plt.xlabel('Date',fontsize=16)
	plt.title('Players on Steam', fontsize= 16, weight='bold')
	plt.grid(True)

	st.pyplot(fig)

	st.write("In this graph we see how there are 2 increases in the number of players, in January 2018 and in March 2020. In our case we will focus on the increase in March 2020 because the main reason is due to the covid-19 confinements.")

	st.write("---")

	# Machine learning section
	st.header('Machine Learning - KNN Regressor')
	st.write('''
		Due to what we have seen in our data it has been decided to develop a predictive model for the two trends.
		One prediction model for the total data and another prediction model for the new evolution after the pandemic.''')

	#Model selection
	model_selected = ["Prediction based on total data", "Prediction based on data since covid"]
	choice = st.selectbox("Please. Choose your prediction model", model_selected)


# Prediction with knn (k=43) and total data
	if choice == 'Prediction based in total data':

		st.subheader("Average of viewers expected per month")

		expected_viewers = st.slider("Number of viewers dou you expect to have", 0, 1000000)

		if st.button("Calculate"):

			knn_regressor = load_prediction_model("Models/knn_regression_viewers.pkl")
			expected_viewers_reshaped = np.array(expected_viewers).reshape(-1,1)

			predict_players = knn_regressor.predict(expected_viewers_reshaped)

			st.success("With {} viewers on your game you should expect to have {} players per month".format(expected_viewers,(predict_players[0].round(1))))

# Prediction with knn (k=5) and data since covid-19
	if choice == 'Prediction based on data since covid':

		st.subheader("Average of viewers expected per month")

		expected_viewers_2020 = st.slider("Number of viewers do you expect to have", 0, 1000000)

		if st.button("Calculate"):

			knn_regressor_2020 = load_prediction_model("Models/knn_regression_viewers_2020.pkl")
			expected_viewers_reshaped_2020 = np.array(expected_viewers_2020).reshape(-1,1)

			predict_players_2020 = knn_regressor_2020.predict(expected_viewers_reshaped_2020)

			st.success("With {} viewers on your game you should expect to have {} players per month".format(expected_viewers_2020,(predict_players_2020[0].round(1))))

if __name__ == '__main__':
	main()