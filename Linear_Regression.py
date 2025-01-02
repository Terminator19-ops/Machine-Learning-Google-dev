## LOADING DEPENDENCIES

#general
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import io

#Data Handling
import numpy as np
import pandas as pd

#Machine Learning
import keras

#Data Visualisation
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

#Time
import time
t1=time.time()



##LOADING AND READING THE DATASET

#reading the csv file
chicago_taxi_dataset = pd.read_csv("chicago_taxi_train.csv")

#creating a dataframe from a csv file
training_df=chicago_taxi_dataset[['TRIP_SECONDS', 'TRIP_MILES', 'TRIP_SPEED', 'FARE', 'TIPS', 'TIP_RATE', 'TRIP_TOTAL', 'PAYMENT_TYPE', 'COMPANY']]
print('Read dataset completed successfully.')
print('Total number of rows: {0}\n\n'.format(len(training_df.index)))
#{training_df.index}=> it means row labels in data frame [row(-),column(|)]
#(len(training_df.index)) calculates the length of number of indices
#format(len(training_df.index) is used for string formatting, {0} acts as a placeholder, like %d 
training_df.head(200)
#head(n) functrion is used to display the n first lines of the dataframe
#head() returns the first 5 lines of the dataframe



training_df.describe()
#By default, describe() returns statistics for numerical columns:
'''
#Count: Number of non-null (non-missing) entries in each column.
#Mean: Average of the values in the column.
#Std: Standard deviation, showing the spread of data.
#Min: Minimum value in the column.
#25%: The 25th percentile (first quartile) of the column values.
#50%: The 50th percentile (median) of the column values.
#75%: The 75th percentile (third quartile) of the column values.
#Max: Maximum value in the column.
training_df.describe(include='all')
'''

"""
What is the maximum fare? 				              
What is the mean distance across all trips? 		
How many cab companies are in the dataset? 		  
What is the most frequent payment type? 		    
Are any features missing data? 				          
"""
##to find the answers we use the following functions:

#max Fare
max_fare = training_df['FARE'].max()
print("What is the maximum fare? \nAnswer: ${fare:.2f}".format(fare = max_fare))


#mean distance
mean_distance = training_df['TRIP_MILES'].mean()
print("What is the mean distance across all trips? \nAnswer: {mean:.4f} miles".format(mean = mean_distance))


# cab companies 
num_unique_companies =  training_df['COMPANY'].nunique()
print("How many cab companies are in the dataset? \nAnswer: {number}".format(number = num_unique_companies))
#['COMPANY']picks the company column
#.nunique counts the number of distinct values present in the column

# What is the most frequent payment type?
most_freq_payment_type = training_df['PAYMENT_TYPE'].value_counts().idxmax()
print("what is the Most used payment type? \nAnswer: {type}".format(type=most_freq_payment_type))
#.value_counts calculates the number of each type of unique value in the parameter "payment_type"
#.idmax finds the higest of them and gets that as output


# Are any features missing data?
missing_values = training_df.isnull().sum().sum()
print("Are any features missing data? \nAnswer:", "No" if missing_values == 0 else "Yes")
#.isnull funtion identifies all the null or NaN spaces in the dataset
#the first .sum, adds up all the values in all column,(example: so if there are 2 blank spaces in 1st column and 3 in 2nd, and 4 in 3rd, ans will be 2,3 and 4 )
#the second .sum addsup all the values in the summed column into a single value (Example: taking the previous, the answer will be 9)
#in the print there s an if else stetement


#CORRELATION MATRIX

#@title Code - View correlation matrix
answer=training_df.corr(numeric_only = True)

#//print(answer)

#.corr command makes a correaltion matrix with the features as parameters on both sides(x and y axes)
#numeric_only:: only takes in the coulumn which has numeric values   



#A correlation matrix is a table that displays the correlation coefficients between multiple variables in a dataset. 
#machine learning is determining which features correlation with the label.


#1.0: perfect positive correlation; that is, when one attribute rises, the other attribute rises.
#-1.0: perfect negative correlation; that is, when one attribute rises, the other attribute falls.
#0.0: no correlation; the two columns are not linearly related.

#to view pairplot
sns.pairplot(training_df, x_vars=["FARE", "TRIP_MILES", "TRIP_SECONDS"], y_vars=["FARE", "TRIP_MILES", "TRIP_SECONDS"])
#creates plot

#plt.show()

#shows plot
#PLOTTING FUNCTIONS

def make_plots(df, feature_names, label_name, model_output, sample_size=200):

  random_sample = df.sample(n=sample_size).copy()
  random_sample.reset_index()
  weights, bias, epochs, rmse = model_output

  is_2d_plot = len(feature_names) == 1
  model_plot_type = "scatter" if is_2d_plot else "surface"
  fig = make_subplots(rows=1, cols=2,
                      subplot_titles=("Loss Curve", "Model Plot"),
                      specs=[[{"type": "scatter"}, {"type": model_plot_type}]])

  plot_data(random_sample, feature_names, label_name, fig)
  plot_model(random_sample, feature_names, weights, bias, fig)
  plot_loss_curve(epochs, rmse, fig)

  fig.show()
  return

def plot_loss_curve(epochs, rmse, fig):
  curve = px.line(x=epochs, y=rmse)
  curve.update_traces(line_color='#ff0000', line_width=3)

  fig.append_trace(curve.data[0], row=1, col=1)
  fig.update_xaxes(title_text="Epoch", row=1, col=1)
  fig.update_yaxes(title_text="Root Mean Squared Error", row=1, col=1, range=[rmse.min()*0.8, rmse.max()])

  return

def plot_data(df, features, label, fig):
  if len(features) == 1:
    scatter = px.scatter(df, x=features[0], y=label)
  else:
    scatter = px.scatter_3d(df, x=features[0], y=features[1], z=label)

  fig.append_trace(scatter.data[0], row=1, col=2)
  if len(features) == 1:
    fig.update_xaxes(title_text=features[0], row=1, col=2)
    fig.update_yaxes(title_text=label, row=1, col=2)
  else:
    fig.update_layout(scene1=dict(xaxis_title=features[0], yaxis_title=features[1], zaxis_title=label))

  return

def plot_model(df, features, weights, bias, fig):
  df['FARE_PREDICTED'] = bias[0]

  for index, feature in enumerate(features):
    df['FARE_PREDICTED'] = df['FARE_PREDICTED'] + weights[index][0] * df[feature]

  if len(features) == 1:
    model = px.line(df, x=features[0], y='FARE_PREDICTED')
    model.update_traces(line_color='#ff0000', line_width=3)
  else:
    z_name, y_name = "FARE_PREDICTED", features[1]
    z = [df[z_name].min(), (df[z_name].max() - df[z_name].min()) / 2, df[z_name].max()]
    y = [df[y_name].min(), (df[y_name].max() - df[y_name].min()) / 2, df[y_name].max()]
    x = []
    for i in range(len(y)):
      x.append((z[i] - weights[1][0] * y[i] - bias[0]) / weights[0][0])

    plane=pd.DataFrame({'x':x, 'y':y, 'z':[z] * 3})

    light_yellow = [[0, '#89CFF0'], [1, '#FFDB58']]
    model = go.Figure(data=go.Surface(x=plane['x'], y=plane['y'], z=plane['z'],
                                      colorscale=light_yellow))

  fig.add_trace(model.data[0], row=1, col=2)

  return

def model_info(feature_names, label_name, model_output):
  weights = model_output[0]
  bias = model_output[1]

  nl = "\n"
  header = "-" * 80
  banner = header + nl + "|" + "MODEL INFO".center(78) + "|" + nl + header

  info = ""
  equation = label_name + " = "

  for index, feature in enumerate(feature_names):
    info = info + "Weight for feature[{}]: {:.3f}\n".format(feature, weights[index][0])
    equation = equation + "{:.3f} * {} + ".format(weights[index][0], feature)

  info = info + "Bias: {:.3f}\n".format(bias[0])
  equation = equation + "{:.3f}\n".format(bias[0])

  return banner + nl + info + nl + equation

print("SUCCESS: defining plotting functions complete.")

#MODEL FUNCTIONS


def build_model(my_learning_rate, num_features):
  """Create and compile a simple linear regression model."""
  # Describe the topography of the model.
  # The topography of a simple linear regression model
  # is a single node in a single layer.
  inputs = keras.Input(shape=(num_features,))
  outputs = keras.layers.Dense(units=1)(inputs)
  model = keras.Model(inputs=inputs, outputs=outputs)

  # Compile the model topography into code that Keras can efficiently
  # execute. Configure training to minimize the model's mean squared error.
  model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=my_learning_rate),
                loss="mean_squared_error",
                metrics=[keras.metrics.RootMeanSquaredError()])

  return model

def train_model(model, df, features, label, epochs, batch_size):
  """Train the model by feeding it data."""

  # Feed the model the feature and the label.
  # The model will train for the specified number of epochs.
  # input_x = df.iloc[:,1:3].values
  # df[feature]
  history = model.fit(x=features,
                      y=label,
                      batch_size=batch_size,
                      epochs=epochs)

  # Gather the trained model's weight and bias.
  trained_weight = model.get_weights()[0]
  trained_bias = model.get_weights()[1]

  # The list of epochs is stored separately from the rest of history.
  epochs = history.epoch

  # Isolate the error for each epoch.
  hist = pd.DataFrame(history.history)

  # To track the progression of training, we're going to take a snapshot
  # of the model's root mean squared error at each epoch.
  rmse = hist["root_mean_squared_error"]

  return trained_weight, trained_bias, epochs, rmse


def run_experiment(df, feature_names, label_name, learning_rate, epochs, batch_size):

  import plotly.io as pio
  pio.renderers.default = 'browser'
  
  print('INFO: starting training experiment with features={} and label={}\n'.format(feature_names, label_name))

  num_features = len(feature_names)

  features = df.loc[:, feature_names].values
  label = df[label_name].values

  model = build_model(learning_rate, num_features)
  model_output = train_model(model, df, features, label, epochs, batch_size)

  print('\nSUCCESS: training experiment complete\n')
  print('{}'.format(model_info(feature_names, label_name, model_output)))
  make_plots(df, feature_names, label_name, model_output)

  return model

print("SUCCESS: defining linear regression functions complete.")




#EXPERIMENT-1(1 feature)

# The following variables are the hyperparameters.
learning_rate = 0.001
epochs = 50
batch_size = 50

# Specify the feature and the label.
features = ['TRIP_MILES']
label = 'FARE'

model_1 = run_experiment(training_df, features, label, learning_rate, epochs, batch_size)

#Question/Answers

# How many epochs did it take to converge on the final model?
# -----------------------------------------------------------------------------
answer = """
Use the loss curve to see where the loss begins to level off during training.

With this set of hyperparameters:

  learning_rate = 0.001
  epochs = 20
  batch_size = 50

"""
print(answer)

# How well does the model fit the sample data?
# -----------------------------------------------------------------------------
answer = '''
It appears from the model plot that the model fits the sample data fairly well.
'''
print(answer)

# How did raising the learning rate impact your ability to train the model?
# -----------------------------------------------------------------------------
answer = """
When the learning rate is too high, the loss curve bounces around and does not
appear to be moving towards convergence with each iteration. Also, notice that
the predicted model does not fit the data very well. With a learning rate that
is too high, it is unlikely that you will be able to train a model with good
results.
"""
print(answer)

# How did lowering the learning rate impact your ability to train the model?
# -----------------------------------------------------------------------------
answer = '''
When the learning rate is too small, it may take longer for the loss curve to
converge. With a small learning rate the loss curve decreases slowly, but does
not show a dramatic drop or leveling off. With a small learning rate you could
increase the number of epochs so that your model will eventually converge, but
it will take longer.
'''
print(answer)

# Did changing the batch size effect your training results?
# -----------------------------------------------------------------------------
answer = '''
Increasing the batch size makes each epoch run faster, but as with the smaller
learning rate, the model does not converge with just 20 epochs. If you have
time, try increasing the number of epochs and eventually you should see the
model converge.
'''
print(answer)

t2=time.time() #time  
print("time elapsed = ",t2-t1)


#@EXPERIMENT 2

learning_rate = 0.001
epochs = 50
batch_size = 50

training_df.loc[:, 'TRIP_MINUTES'] = training_df['TRIP_SECONDS']/60

features = ['TRIP_MILES', 'TRIP_SECONDS']
label = 'FARE'

model_2 = run_experiment(training_df, features, label, learning_rate, epochs, batch_size)


#TIME 
t3=time.time()
print("the total time taken is ",t3-t1)
print("the total time for the second model is ",t3-t2)

#@title Code - Define functions to make predictions
def format_currency(x):
  return "${:.2f}".format(x)

def build_batch(df, batch_size):
  batch = df.sample(n=batch_size).copy()
  batch.set_index(np.arange(batch_size), inplace=True)
  return batch

def predict_fare(model, df, features, label, batch_size=50):
  batch = build_batch(df, batch_size)
  predicted_values = model.predict_on_batch(x=batch.loc[:, features].values)

  data = {"PREDICTED_FARE": [], "OBSERVED_FARE": [], "L1_LOSS": [],
          features[0]: [], features[1]: []}
  for i in range(batch_size):
    predicted = predicted_values[i][0]
    observed = batch.at[i, label]
    data["PREDICTED_FARE"].append(format_currency(predicted))
    data["OBSERVED_FARE"].append(format_currency(observed))
    data["L1_LOSS"].append(format_currency(abs(observed - predicted)))
    data[features[0]].append(batch.at[i, features[0]])
    data[features[1]].append("{:.2f}".format(batch.at[i, features[1]]))

  output_df = pd.DataFrame(data)
  return output_df

def show_predictions(output):
  header = "-" * 80
  banner = header + "\n" + "|" + "PREDICTIONS".center(78) + "|" + "\n" + header
  print(banner)
  print(output)
  return

output = predict_fare(model_2, training_df, features, label)
show_predictions(output)
