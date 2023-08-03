import eogslib
import pandas

eogs = eogslib.EOGS()

# load dataset
data = pandas.read_csv('parkinsons.csv')

training_data = data[['Jitter(%)', 'Shimmer', 'NHR', 'HNR', 'RPDE', 'DFA']]
test_data = data[['PPE']]

training_data = training_data.head(2)
test_data = test_data.head(2)

# train twice to see the second input fall into the first granule
eogs.train_many(training_data, test_data)