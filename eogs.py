import eogslib
import pandas

eogs = eogslib.EOGS(0.5, eogslib.MODE_MANUAL)

# load dataset
data = pandas.read_csv('data.csv')

# set the entire dataset as training data except the last row
training_data = data.iloc[:-1].to_numpy()
test_data = data.iloc[-1:].to_numpy()

# train with whole dataset but last value
eogs.train_many(training_data)

# train and predict next value
prediction = eogs.train(test_data)

print(prediction)

