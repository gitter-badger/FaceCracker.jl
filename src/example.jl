export example


############################################################
using ScikitLearn
using DecisionTree
using PyPlot

function example()
	X = sort(5 * rand(80))
	XX = reshape(X, 80, 1)
	y = [sin(_) for _ in X]
	y[1:5:end] += 3 * (0.5 - rand(16))
	regr = DecisionTreeRegressor(pruning_purity_threshold=0.05)
	fit!(regr, XX, y)
	# predict
	X_test = 0:0.01:5.0
	y_test = predict(regr, hcat(X_test))
	X_test, y_test
end

