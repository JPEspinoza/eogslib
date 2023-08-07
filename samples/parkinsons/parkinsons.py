import eogslib
import pandas

# Load the data
data = pandas.read_csv('parkinsons.csv')

x = data[["Jitter(%)", "Shimmer", "NHR", "HNR", "RPDE", "DFA"]]
y = data[["PPE"]]

eogs = eogslib.EOGS()

eogs.fit(x, y)

# Predict
print(y.tail(1))
print(eogs.predict(x.tail(1)))