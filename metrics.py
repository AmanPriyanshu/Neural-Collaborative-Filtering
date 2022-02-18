import numpy as np

def hit_ratio(y, pred, N=10):
	mask = np.zeros_like(y)
	mask[y>0] = 1
	pred_masked = pred*mask
	best_index = np.argmax(y)
	pred_masked_indexes = np.argsort(pred_masked, reverse=True)[:N]
	if best_index in pred_masked_indexes:
		return 1
	else:
		return 0

def ndcg(y, pred, N=10):
	actual_recommendation_best_10indexes = torch.argsort(client_true_vals[0], descending=True)[:10]
    actual_recommendation_best_10 = client_true_vals[0][actual_recommendation_best_10indexes]
    predicted_recommendation_best_10 = predict[0][actual_recommendation_best_10indexes]
    predicted_recommendation_best_10 = torch.round(predicted_recommendation_best_10)
    predicted_recommendation_best_10[predicted_recommendation_best_10<0] = 0
    dcg_numerator = torch.pow(2, predicted_recommendation_best_10) - 1
    denomimator = torch.log2(torch.arange(start=2, end=K+2))
    idcg_numerator = torch.pow(2, actual_recommendation_best_10) - 1
    dcg = torch.sum(dcg_numerator/denomimator)
    idcg = torch.sum(idcg_numerator/denomimator)
    ndcg = dcg/idcg

def compute_metrics(y, pred, metric_functions=None):
	if metric_functions is None:
		metric_functions = [hit_ratio, ndcg]
	return [fun(y, pred) for fun in metric_functions]