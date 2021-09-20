# Import Section
import streamlit as st
import pandas as pd
import plotly.express as px
import cufflinks as cf
import math
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)
cf.go_offline()

# Container Section
header = st.beta_container()
data = st.beta_container()
visualization = st.beta_container()
model_training = st.beta_container()
model_evaluation = st.beta_container()
prediction = st.beta_container()
footer = st.beta_container()

# Header Section
with header:
	st.title('Pizza Price Prediction - Knightbearr') # Set the title

	st.image('./picture/pizza.jpg') # Image
	
	st.header("**Welcome!**") # Header

	st.markdown("In this project, I'll make a **Machine Learning** program.\
				Well, here we will predict the price from the data that I have prepared.\
				Would you like to know where? and who created this data?")
	
	st.subheader('**The First**') # Subheader

	st.markdown("So, where do I get this data from? I got this data from **Kaggle** for those of you who don't know, let me explain, What is **Kaggle**?")
	st.markdown("**Kaggle** is one of the most famous sites in the world of **Data Science** and **Machine Learning** which consists of more than **6000 datasets**\
				which can be downloaded in CSV format. **Kaggle** is very useful for those of you who are studying **Data Science**. **This dataset** helps a lot\
				scientists around the world to create models.")
	st.markdown("**Kaggle** is not just a dataset but comprises the **largest community of data scientists**. Not a little\
				companies that have analytical problems, but they don't have the skilled **Data Scientist** resources. **For you beginners \
				and Data Science students**, **Kaggle** is very useful as a place to practice and sharpen your skills.")


	st.subheader('**Second**')
	st.markdown("Who created this Dataset? yes, you are right, I made this Dataset, \
				If you want to download this dataset or curious about what this data looks like, you can click this link\
				[Pizza Price Prediction](https://www.kaggle.com/knightbearr/pizza-price-prediction)")
	
	st.markdown("And first of all, before we predict the price of **Pizza**, let me show you how what form of data or description of data that we want to use for this Project.\
				 So, without further ado, let's just take a look at all of this data.")
	st.markdown("Let's look at the overall data first.")

# Data Section
with data:
	# Title
	st.subheader('**Overall Data:**')

	# Load Data
	data = pd.read_csv('data/pizza.csv')

	data['price_rupiah'] = data['price_rupiah'].str.replace('Rp', '').str.replace(',', '').astype('int64')
	data['diameter'] = data['diameter'].str.replace('inch', '').astype('float64')

	# Prepare Data
	head_data = data.head() 
	tail_data = data.tail()
	type_data = data.dtypes.to_frame()
	null_data = data.isna().mean().to_frame()
	clean_data = data.loc[:, ['price_rupiah', 'diameter']]
	type_clean_data = data[['price_rupiah', 'diameter']].dtypes.to_frame()

	# Display Data
	st.write(data)

	# Markdown
	st.markdown("Do you know what the shape of the data above is? or do you have an idea? Which technique, and which **Machine Learning** algorithm, will we use to predict the price from the above data?")
	st.markdown("If you don't know, that's fine, that's fine, but if you do, congratulations! you already know one of the techniques and algorithms in **Machine Learning!**")
	st.markdown("Before that, let's see how the first and last 5 rows of data look like.")

	st.subheader('**First 5 rows of Data:**')
	st.write(head_data)

	st.subheader('**Last 5 rows of Data:**')
	st.write(tail_data)

	st.markdown("Okay, before we proceed to the prediction stage, first of all, let's see if there is missing data or Nah.")
	st.write(type_data)
	st.markdown('Okay mostly the data is an Object Type.')
	st.write(null_data)
	st.markdown("Great! There's no null value!")

	st.markdown("Okay, next let's tidy up this data, before we visualize it.")
	st.write(clean_data)
	st.write(type_clean_data)
	st.subheader('Conclusion:')
	st.markdown('* There are **129 rows** and **9 columns** in the dataset.')
	st.markdown('* This data is included in **Categorical Data** due to the large number of objects in the data.')
	st.markdown('* There is no **Missing Value** in the data.')

	st.markdown('Great! now the data is ready for us to visualize.')

# Visualizatio Section
with visualization:
	st.header('**Data Visualization**') # Header
	st.markdown("Data Visualization is the most important thing in the process of making a **Machine Learning**, why? \
				because from **Data Visualization** we can know, correlation or how the shape of the data itself.")

	# Figure 1
	figure_1 = px.scatter_3d(
	            data, # Data
	            title='3D Visualization of Company, Price, and Diameter',
	            x='company', # X data 
	            y='price_rupiah', # Y data
	            z='diameter', # Z data
	            color='company', # Color or Hue
	            opacity=0.7 # Opacity
        		)

	# Figure 2
	figure_2 = px.box(
			    data, # Data
				x='company', # X data
				y='price_rupiah', # Y data
				color='extra_sauce', # Color or Hue
				title='Does Pizza Get Extra Sauce or Nah' # Title
				)

	# Figure 3
	figure_3 = px.box(
			    data, # Data
				x='company', # X data
				y='price_rupiah', # Y data
				color='extra_cheese', # Color or Hue
				title='Does Pizza Get Extra Cheese or Nah' # Title
				)

	# Figure 4
	figure_4 = px.box(
			    data, # Data
				x='company', # X data
				y='price_rupiah', # Y data
				color='extra_mushrooms', # Color or Hue
				title='Does Pizza Get Extra Sauce or Nah' # Title
				)

	# Figure 5
	figure_5 = px.box(
			    data, # Data
				x='company', # X data
				y='price_rupiah', # Y data
				color='variant', # Color or Hue
				title='Boxplot Visualization Pizza Variant' # Title
				)

	# Figure 6
	figure_6 = px.box(
			    data, # Data
				x='company', # X data
				y='price_rupiah', # Y data
				color='variant',
				title='Boxplot Visualization Pizza Topping' # Title
				)

	# Figure 7
	figure_7 = px.box(
			    data, # Data
				x='company', # X data
				y='price_rupiah', # Y data
				color='size',
				title='Boxplot Visualization Pizza Size' # Title
				)

	# Plot Section
	st.plotly_chart(figure_1)
	st.markdown("Isn't this a cool visualization? With data visualization like this, \
				we can understand the shape and description of what data we are examining.")

	st.plotly_chart(figure_2)
	st.markdown('This visualization shows whether Pizza with a certain **price_rupiah** range\
				and from every **company**, can get **extra_sauce** or **Nah**.')

	st.plotly_chart(figure_3)
	st.markdown('This visualization shows whether Pizza with a certain **price_rupiah** range\
				and from every **company**, can get **extra_cheese** or **Nah**.')

	st.plotly_chart(figure_4)
	st.markdown('This visualization shows whether Pizza with a certain **price_rupiah** range\
				and from every **company**, can get **extra_mushrooms** or **Nah**.')

	st.plotly_chart(figure_5)
	st.markdown('This visualization shows whether Pizza with a certain **price_rupiah** range\
				and from every **company**, and seeing the **variant** that every company have.')

	st.plotly_chart(figure_6)
	st.markdown('This visualization shows whether Pizza with a certain **price_rupiah** range\
				and from every **company**, and seeing the **topping** that every company have.')

	st.plotly_chart(figure_7)
	st.markdown('This visualization shows whether Pizza with a certain **price_rupiah** range\
				and from every **company**, and seeing the **size** that every company have.')

# Model Training Section
with model_training:
	encoder = LabelEncoder() # Let's gooo!

	for i in data.columns: # Make a for loops
	    if data[i].dtype == 'object': 
	        encoder.fit_transform(list(data[i].values)) # Fit transform
	        data[i] = encoder.transform(data[i].values) # Transform
	         
	        data = data  

	X = data.drop(columns=['price_rupiah', 'company']) # Our Feature
	y = data['price_rupiah'] # Our Labels

	# Splitting data
	X_train, X_test, y_train, y_test = train_test_split(
			X, y, test_size=0.2, random_state=42
		)

	# Our Model
	model = RandomForestRegressor(
				n_estimators=600, 
				min_samples_leaf=1, 
				random_state=42
			) 

	model = model.fit(X_train, y_train) # Training Model

	pred_test = model.predict(X_test) # Predict Test Data
	pred_train = model.predict(X_train) # Predict Train Data

# Model Evaluation Section
with model_evaluation:
	st.header('**Model Evaluation**') # Title

	st.markdown("Before that, I'll give you some information what is **R2_Score**, **Mean Square Error**, and **Root Mean Square Error** are.")

	st.markdown("* **R2_Score:** R-squared (R2) is a statistical measure that represents the proportion\
				of the variance for a dependent variable that's explained by an independent variable or variables in a regression model.\
				R2 is a statistic that will give some information about the goodness of fit of a model.")

	st.markdown("* **Mean Squared Error:** mean squared error (MSE) or mean squared deviation (MSD) of an estimator \
            	(of a procedure for estimating an unobserved quantity) measures the average of the squares of the errors—that \
           		is, the average squared difference between the estimated values and the actual value.")

	st.markdown("* **Root Mean Square Error or Root Mean Squared Deviation: ** Root Mean Square Error (RMSE) is the standard deviation of the residuals \
				(prediction errors). Residuals are a measure of how far from the regression line data points are RMSE \
				is a measure of how spread out these residuals are. In other words, it tells you how concentrated the data \
				is around the line of best fit.")

	st.markdown("Simple is, If **R2_Score** is close to `1.0`, and If **Mean Squared Error**, and **Root Mean Squared Error**\
				is close to `0.0`, we can say that our model learns well.")

	st.subheader('Train Data Score:') # Subheader

	train_r2_score = r2_score(y_train, pred_train) # R2_score
	train_mse = mean_squared_error(y_train, pred_train) # MSE Score
	train_rmse = math.sqrt(mean_squared_error(y_train, pred_train)) # SQRT MSE Score

	st.write("R2 Score:", train_r2_score) # Displaying 
	st.write("Mean Squared Error:", train_mse) # Displaying
	st.write("Square Root Mean Square Error:", train_rmse) # Displaying

	# Data Frame
	train_data = pd.DataFrame({ 
			'Predicted Price': pred_train, # Train Predict Data
			'Actual Price': y_train # Test Actual Data
		})

	st.line_chart(train_data) # Make A Line Chart

	st.subheader('Test Data Score:') # Subheader

	test_r2_score = r2_score(y_test, pred_test) # R2_score
	test_mse = mean_squared_error(y_test, pred_test) # MSE Score
	test_rmse = math.sqrt(mean_squared_error(y_test, pred_test)) # SQRT MSE Score

	st.write("R2 Score:", test_r2_score) # Displaying
	st.write("Mean Squared Error:", test_mse) # Displaying
	st.write("Square Root Mean Square Error:", test_rmse) # Displaying

	# Data Frame
	test_data = pd.DataFrame({
			'Predicted Price': pred_test, # Test Predict Data
			'Actual Price': y_test # Test Actual Data
		})

	st.line_chart(test_data) # Make A Line Chart

# Prediction Section
with prediction:
	st.header('**Time to see our prediction:**')

	st.subheader("I'll use this new Data to predict the prices.")

	new_examples = ([{
		'diameter': 14,
		'topping': 9,
		'variant': 17,
		'size': 3,
		'extra_sauce': 1,
		'extra_cheese': 1,
		'extra_mushrooms': 1,
		}])

	st.write(new_examples)

	input_data = (14, 9, 17, 3, 1, 1, 1)

	# Changing the data type to numpy array
	change_input = np.asarray(input_data)

	# Reshape the numpy array as we are predicting for one instance
	reshape_array = change_input.reshape(1, -1)

	predict_data = model.predict(reshape_array)

	st.write('The Price of our Pizza is:', int(predict_data), 'Rupiah')

# Footer Section
with footer:
	st.title("Thank you for visiting my web application!")

	st.markdown("Hope you guys like my First Project :D")
	st.markdown("You guys can reach me on:")
	st.markdown("* [Email](azmimuis3312@gmail.com)")
	st.markdown("* [Instagram](https://www.instagram.com/knightbearr/)")
	st.markdown("* [Twitter](https://twitter.com/Knightbearr)")
	st.markdown("* [Kaggle](https://www.kaggle.com/knightbearr)")
	st.markdown("* [Github](https://github.com/knightbearr)")

	st.markdown("You can download this code on my Github: [download code](https://github.com/knightbearr/Pizza-Price-Prediction-Web-Application)")