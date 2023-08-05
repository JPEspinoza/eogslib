import eogslib
import pandas

# load dataset
data = pandas.read_csv('parkinsons.csv')

x = data[['Jitter(%)', 'Shimmer', 'NHR', 'HNR', 'RPDE', 'DFA']]
y = data[['PPE']]

x = x.head(100)
y = y.head(100)

# train twice to see the second input fall into the first granule
eogs = eogslib.EOGS()
eogs.train_many(x, y)

# predict
print(eogs.predict_interval(x.tail(1)))
