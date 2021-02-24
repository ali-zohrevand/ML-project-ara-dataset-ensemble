import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
columnsString = "daySection,Ph1,Ph2,Ir1,Fo1,Fo2,Di3,Di4,Ph3,Ph4,Ph5,Ph6,Co1,Co2,Co3,So1,So2,Di1,Di2,Te1,Fo3,activity1,activity2"
col_names = columnsString.split(",")
dataset = pd.read_csv(f"DAY_1.csv", header=None, names=col_names)
print(dataset.corr()['activity1'].sort_values())
plot=sns.scatterplot(x="activity1",y="Di4",data=dataset)
print(plot)

#
# df = pd.read_csv("DAY_1.csv")
# print()
# # plot=sns.pairplot(pima)
# # plot.savefig("output.png")
#
