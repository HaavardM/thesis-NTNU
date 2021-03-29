import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
import numpy as np

plt.style.use('ggplot')

x = np.linspace(0, 10)
y = np.sin(x)

gp = GaussianProcessRegressor(alpha=0.1)



f, std = gp.predict(x.reshape(-1, 1), return_std=True)
mean = f
ci_high = mean + 2*std
ci_low = mean - 2*std
plt.figure()
plt.plot(x, mean, color="red")
plt.fill_between(x, ci_high, ci_low, alpha=0.3)
plt.xlim((min(x), max(x)))
plt.ylim((-10, 10))
plt.title("Prior f(x)")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.savefig("prior.png")


x_obs = np.array([2.0])
y_obs = np.array([2.0])
gp.fit(x_obs[..., np.newaxis], y_obs)
f, std = gp.predict(x.reshape(-1, 1), return_std=True)
mean = f
ci_high = mean + 2*std
ci_low = mean - 2*std
plt.figure()
plt.plot(x, mean, color="red")
plt.fill_between(x, ci_high, ci_low, alpha=0.3)
plt.scatter(x_obs, y_obs, marker="o", color="black")
plt.xlim((min(x), max(x)))
plt.ylim((-10, 10))
plt.title("Posterior f(x)")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.savefig("posterior.png")
