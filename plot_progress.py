from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv("progress.csv")
columns = list(df.columns)
df = df.values
for index, feature_name in enumerate(columns[1:]):
	feature = df.T[index+1]
	plt.cla()
	plt.plot(df.T[0], feature)
	plt.xlabel("Epochs")
	plt.ylabel(feature_name)
	if feature_name != "loss":
		plt.ylim([-0.01, 1.01])
	plt.savefig("./images/"+feature_name.replace("@", "")+".png")