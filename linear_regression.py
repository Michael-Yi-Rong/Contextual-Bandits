import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

correlations = data.iloc[:, 1:].corr().iloc[0, 1:]
print(correlations)
high_correlations = correlations[correlations.abs() > 0.02]
print("Features with absolute correlation > 0.02:", high_correlations)
X = data.iloc[:, 2:].values
y = data['Payoff'].values
model = LinearRegression()
model.fit(X, y)
r2 = r2_score(y, model.predict(X))
print(f"R² score: {r2}")
