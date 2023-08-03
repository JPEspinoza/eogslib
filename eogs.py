import eogslib
import pandas

eogs = eogslib.EOGS()

# load dataset
data = pandas.read_csv('parkinsons.csv')

training_data = data[['Jitter(%)', 'Shimmer', 'NHR', 'HNR', 'RPDE', 'DFA']]
test_data = data[['PPE']]

training_data = training_data.to_numpy()
test_data = test_data.to_numpy()

# train with whole dataset but last value
prediction = eogs.train_many(training_data,test_data)

print(prediction)