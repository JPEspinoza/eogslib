import eogslib
import pandas

# load dataset
data = pandas.read_csv('parkinsons.csv')

training_data = data[['Jitter(%)', 'Shimmer', 'NHR', 'HNR', 'RPDE', 'DFA']]
test_data = data[['PPE']]

training_data = training_data
test_data = test_data

# train twice to see the second input fall into the first granule
eogs = eogslib.EOGS()
eogs.train_many(training_data, test_data)