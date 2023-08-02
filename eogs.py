import eogslib
import pandas

eogs = eogslib.EOGS()

# load dataset
data = pandas.read_csv('parkinsons.csv')

# set the entire dataset as training data except the last row
training_data = data.to_numpy()

# train with whole dataset but last value
prediction = eogs.train_many(training_data)

print(prediction)

