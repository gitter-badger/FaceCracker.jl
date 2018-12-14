using ScikitLearn
using DecisionTree
using PyPlot


"""
	decision_tree()
"""
function decision_tree()
	X = sort(5 * rand(80))
	XX = reshape(X, 80, 1)
	y = sin.(X)
	y[1:5:end] .+= 3 * (0.5 .- rand(16))
	regr = DecisionTreeRegressor(pruning_purity_threshold=0.05)
	fit!(regr, XX, y)
	# predict
	X_test = 0:0.01:5.0
	y_test = predict(regr, hcat(X_test))
	scatter(X, y, c="k", label="data")
	plot(X_test, y_test, c="r", label="pruning purity threshold=0.05", linewidth=2)
	xlabel("data")
	ylabel("target")
	title("Decision Tree Regression")
	legend(prop=Dict("size" => 10))
end

