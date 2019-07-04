def evaluation(method,prediction,real):
	if method == "jaccard":
		inter,union = 0,0
		for p in prediction:
			if p in real:
				inter+=1
			else:
				union+=1
		return inter / union
