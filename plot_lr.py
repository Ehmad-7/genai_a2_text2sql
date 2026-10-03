import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from train import noam_lr

steps = list(range(1, 20001))
lrs = [noam_lr(s) for s in steps]
print("peak lr:", max(lrs), "at step", steps[lrs.index(max(lrs))])

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for a, log in zip(ax, [False, True]):
    a.plot(steps, lrs)
    a.axvline(4000, ls="--", color="gray")
    a.set_xlabel("step")
    a.set_ylabel("learning rate")
    if log:
        a.set_xscale("log"); a.set_yscale("log")
        a.set_title("log-log (decay ~ step^-0.5)")
    else:
        a.set_title("Warmup 4000 steps, then inverse-sqrt decay")
fig.savefig("results/fig3_lr_schedule.png", dpi=150, bbox_inches="tight")