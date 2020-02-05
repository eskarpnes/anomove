from etl import ETL
import matplotlib.pyplot as plt

etl = ETL("/home/login/Dataset/", window_sizes=[128, 256, 512, 1024])
etl.load("CIMA", tiny=True)

print(etl.cima.keys())
infant = etl.cima["UoC_001"]["data"]


fig, axs = plt.subplots(3, 3, sharex=True, sharey=True, figsize=(15, 15))

for i in range(1, 15, 2):
    plot_index = (i-1)//2
    columns = infant.iloc[:500, i:i+2]
    axs[plot_index // 3][plot_index % 3].scatter(
        columns.iloc[:, 0],
        columns.iloc[:, 1],
        c="red",
        alpha=1,
        s=2
    )
    column_name = columns.columns[0][:-2]
    axs[plot_index // 3][plot_index % 3].title.set_text(column_name)
    axs[plot_index // 3][plot_index % 3].grid()


plt.gca().invert_yaxis()
plt.show()
