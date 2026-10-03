import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

epochs, train, dev = [], [], []
with open("results/train_log.csv") as f:
    for row in csv.DictReader(f):
        epochs.append(int(row["epoch"]))
        train.append(float(row["train_loss"]))
        dev.append(float(row["dev_loss"]))

best = dev.index(min(dev))
plt.figure(figsize=(7, 4))
plt.plot(epochs, train, marker="o", label="train")
plt.plot(epochs, dev, marker="o", label="dev")
plt.axvline(epochs[best], ls="--", color="gray", label=f"best dev (epoch {epochs[best]})")
plt.xlabel("epoch")
plt.ylabel("loss per token (label-smoothed)")
plt.title("Training and dev loss")
plt.legend()
plt.savefig("results/fig2_loss_curves.png", dpi=150, bbox_inches="tight")
print("best epoch:", epochs[best], "dev:", dev[best])