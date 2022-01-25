from sklearn.datasets import load_diabetes
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import VotingRegressor
import matplotlib.pyplot as plt


X, y = load_diabetes(return_X_y=True)
 
# Train classifiers
model_1 = GradientBoostingRegressor(random_state=1)
model_2 = RandomForestRegressor(random_state=1)
model_3 = LinearRegression()

model_1.fit(X, y)
model_2.fit(X, y)
model_3.fit(X, y)

final_model = VotingRegressor([("gb", model_1), ("rf", model_2), ("lr", model_3)])
final_model.fit(X, y)
 
xt = X[:20]
pred1 = model_1.predict(xt)
pred2 = model_2.predict(xt)
pred3 = model_3.predict(xt)
pred4 = final_model.predict(xt)

plt.figure()
plt.plot(pred1, "gd", label="GradientBoostingRegressor")
plt.plot(pred2, "b^", label="RandomForestRegressor")
plt.plot(pred3, "ys", label="LinearRegression")
plt.plot(pred4, "r*", ms=10, label="VotingRegressor")

plt.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
plt.ylabel("predicted")
plt.xlabel("training samples")
plt.legend(loc="best")
plt.title("Regressor predictions and their average")

plt.show()
 

 