from etl.etl import ETL
from matplotlib import pyplot as plt


etl = ETL("/home/erlend/datasets", [128, 256, 512, 1024], size=16, random_seed=42)
etl.cache = False
etl.load("CIMA")


infant = etl.cima["077"]
infant = etl.resample(infant)
before_sma = infant["data"]["right_wrist_x"][:250]

etl.preprocess_pooled()

after_sma = etl.cima["077"]["data"]["right_wrist_x"][:250]

fig = plt.Figure()

plt.plot(before_sma, color="red", alpha=0.5)
plt.plot(after_sma, color="green", alpha=0.5)

plt.xlabel("Frame")
plt.ylabel("right_wrist_x")
plt.legend(["Raw data", "SMA=3"])

plt.savefig("sma.png")

