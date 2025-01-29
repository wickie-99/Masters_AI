import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import qmc
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score, mean_absolute_error
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.termination.default import DefaultMultiObjectiveTermination

# -----------------------------
# Step 1: Latin Hypercube Sampling (LHS)
# -----------------------------

# Define design variable ranges
valve_timing_range = np.array([-20, 20])  # Degrees
injection_timing_range = np.array([100, 400])  # Degrees
intake_pressure_range = np.array([1, 3])  # Bar

num_samples = 300  # Limited simulation budget

# Generate LHS samples
sampler = qmc.LatinHypercube(d=3)
lhs_samples = sampler.random(n=num_samples)

# Scale samples to engine parameter ranges
lhs_samples[:, 0] = valve_timing_range[0] + lhs_samples[:, 0] * (valve_timing_range[1] - valve_timing_range[0])
lhs_samples[:, 1] = injection_timing_range[0] + lhs_samples[:, 1] * (injection_timing_range[1] - injection_timing_range[0])
lhs_samples[:, 2] = intake_pressure_range[0] + lhs_samples[:, 2] * (intake_pressure_range[1] - intake_pressure_range[0])

# Simulated responses (for demonstration purposes)
np.random.seed(42)
backfire_risk = np.exp(-((lhs_samples[:, 0] + 10) ** 2) / 50) + np.random.normal(0, 0.05, num_samples)  # Lower is better
performance = np.sin(lhs_samples[:, 1] * np.pi / 400) + np.random.normal(0, 0.05, num_samples)  # Higher is better

# Store in DataFrame
data = pd.DataFrame(lhs_samples, columns=['Valve Timing', 'Injection Timing', 'Intake Pressure'])
data['Backfire Risk'] = backfire_risk
data['Performance'] = performance

# -----------------------------
# Step 2: Neural Network Training with K-Fold Cross-Validation
# -----------------------------

X = lhs_samples  # Inputs
y1 = backfire_risk  # Output 1: Backfire Risk
y2 = performance  # Output 2: Engine Performance

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Define Neural Network Model
nn1 = MLPRegressor(hidden_layer_sizes=(50, 50), activation='relu', solver='adam', max_iter=500, random_state=42)
nn2 = MLPRegressor(hidden_layer_sizes=(50, 50), activation='relu', solver='adam', max_iter=500, random_state=42)

# Train and Validate using Cross-Validation
cv_r2_1 = cross_val_score(nn1, X, y1, cv=kf, scoring='r2')
cv_r2_2 = cross_val_score(nn2, X, y2, cv=kf, scoring='r2')

# Train on full dataset
nn1.fit(X, y1)
nn2.fit(X, y2)

# Predict on training data
y1_pred = nn1.predict(X)
y2_pred = nn2.predict(X)

# Calculate Metrics
r2_1 = r2_score(y1, y1_pred)
r2_2 = r2_score(y2, y2_pred)
mae_1 = mean_absolute_error(y1, y1_pred)
mae_2 = mean_absolute_error(y2, y2_pred)

print(f"Neural Network Model Performance:")
print(f"Backfire Risk - R²: {r2_1:.3f}, MAE: {mae_1:.3f}")
print(f"Engine Performance - R²: {r2_2:.3f}, MAE: {mae_2:.3f}")

# -----------------------------
# Step 3: Multi-Objective Optimization Using pymoo NSGA-II
# -----------------------------

class EngineOptimizationProblem(Problem):
    def __init__(self):
        super().__init__(n_var=3, n_obj=2, n_constr=0, xl=np.array([-20, 100, 1]), xu=np.array([20, 400, 3]))
    
    def _evaluate(self, X, out, *args, **kwargs):
        out["F"] = np.column_stack([
            nn1.predict(X),  # Backfire Risk (Minimize)
            -nn2.predict(X)  # Engine Performance (Maximize, so we minimize negative value)
        ])

problem = EngineOptimizationProblem()
algorithm = NSGA2(
    pop_size=50,
    sampling=FloatRandomSampling(),
    crossover=SBX(prob=0.9, eta=15),
    mutation=PM(eta=20),
    eliminate_duplicates=True
)

termination = DefaultMultiObjectiveTermination(n_max_gen=100)

res = minimize(problem, algorithm, termination=termination, verbose=True)

# Extract Pareto front solutions
pareto_front = res.F

# -----------------------------
# Step 4: Visualization
# -----------------------------

# Plot 3D Scatter Plot to Show All Variables
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(lhs_samples[:, 0], lhs_samples[:, 1], lhs_samples[:, 2], c=backfire_risk, cmap="coolwarm", edgecolor="k", label="LHS Samples")
ax.set_xlabel("Valve Timing (Degrees)")
ax.set_ylabel("Injection Timing (Degrees)")
ax.set_zlabel("Intake Pressure (Bar)")
ax.set_title("Latin Hypercube Sampling (LHS)")
plt.legend()
plt.show()

# Plot Pareto Front
plt.figure(figsize=(6, 5))
plt.scatter(pareto_front[:, 0], -pareto_front[:, 1], color="blue", label="Pareto Front Solutions")
plt.xlabel("Backfire Risk (Minimize)")
plt.ylabel("Engine Performance (Maximize)")
plt.title("Pareto Front - pymoo NSGA-II Optimization")
plt.legend()
plt.grid()
plt.show()
