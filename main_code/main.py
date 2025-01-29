import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import qmc
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.optimizers import Adam
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
backfire_intensity = np.exp(-((lhs_samples[:, 0] + 10) ** 2) / 50) + np.random.normal(0, 0.05, num_samples)  # Lower is better
performance = np.sin(lhs_samples[:, 1] * np.pi / 400) + np.random.normal(0, 0.05, num_samples)  # Higher is better

# In-cylinder parameters for backfire prediction
hydrogen_concentration = np.random.uniform(0, 1, num_samples)  # Normalized max H2 concentration
crank_angle = np.random.uniform(0, 720, num_samples)  # Crank angle at max H2 concentration
turbulence_intensity = np.random.uniform(0, 1, num_samples)  # Normalized turbulence intensity
velocity = np.random.uniform(0, 1, num_samples)  # Normalized velocity

# Store in DataFrame
data = pd.DataFrame(lhs_samples, columns=['Valve Timing', 'Injection Timing', 'Intake Pressure'])
data['Backfire Intensity'] = backfire_intensity
data['Performance'] = performance
data['Hydrogen Concentration'] = hydrogen_concentration
data['Crank Angle'] = crank_angle
data['Turbulence Intensity'] = turbulence_intensity
data['Velocity'] = velocity

# -----------------------------
# Step 2: Neural Network Training with Intermediate Hidden Layer
# -----------------------------

# Input features
X_base = lhs_samples  # Common inputs: Valve Timing, Injection Timing, Intake Pressure
X_in_cylinder = np.column_stack([hydrogen_concentration, crank_angle, turbulence_intensity, velocity])

# Outputs
y_backfire = backfire_intensity  # Backfire Intensity
y_performance = performance  # Engine Performance

# Split dataset for training
X_train_base, X_test_base, X_train_cylinder, X_test_cylinder, y_train, y_test = train_test_split(
    X_base, X_in_cylinder, y_backfire, test_size=0.2, random_state=42)

# Define Input Layers
input_base = Input(shape=(3,), name="Base_Input")  # VT, IT, IP
hidden_base = Dense(64, activation="relu")(input_base)
hidden_base = Dense(32, activation="relu")(hidden_base)

# Intermediate Hidden Layer (Processing In-Cylinder Parameters)
input_in_cylinder = Input(shape=(4,), name="In_Cylinder_Input")  # H2 Concentration, Crank Angle, Turbulence, Velocity
hidden_in_cylinder = Dense(32, activation="relu")(input_in_cylinder)

# Merge the two hidden layers
merged = Concatenate()([hidden_base, hidden_in_cylinder])
merged = Dense(32, activation="relu")(merged)
output = Dense(1, activation="linear", name="Backfire_Intensity_Output")(merged)

# Define Model
nn_backfire = Model(inputs=[input_base, input_in_cylinder], outputs=output)
nn_backfire.compile(optimizer=Adam(learning_rate=0.001), loss="mse", metrics=["mae"])

# Train the model
nn_backfire.fit([X_train_base, X_train_cylinder], y_train, epochs=100, batch_size=16, validation_split=0.2, verbose=1)

# -----------------------------
# Step 3: Multi-Objective Optimization Using pymoo NSGA-II
# -----------------------------

class EngineOptimizationProblem(Problem):
    def __init__(self):
        super().__init__(n_var=3, n_obj=2, n_constr=0, xl=np.array([-20, 100, 1]), xu=np.array([20, 400, 3]))
    
    def _evaluate(self, X, out, *args, **kwargs):
        estimated_hydrogen_concentration = np.full((X.shape[0], 1), np.mean(hydrogen_concentration))
        estimated_crank_angle = np.full((X.shape[0], 1), np.mean(crank_angle))
        estimated_turbulence_intensity = np.full((X.shape[0], 1), np.mean(turbulence_intensity))
        estimated_velocity = np.full((X.shape[0], 1), np.mean(velocity))

        X_in_cylinder = np.column_stack([estimated_hydrogen_concentration, estimated_crank_angle,
                                         estimated_turbulence_intensity, estimated_velocity])
        
        out["F"] = np.column_stack([
            nn_backfire.predict([X, X_in_cylinder]),  # Backfire Intensity (Minimize)
            -nn_performance.predict(X)  # Engine Performance (Maximize, so we minimize negative value)
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

# Plot Latin Hypercube Sampling Distribution
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(lhs_samples[:, 0], lhs_samples[:, 1], lhs_samples[:, 2], c=backfire_intensity, cmap="coolwarm", edgecolor="k", label="LHS Samples")
ax.set_xlabel("Valve Timing (Degrees)")
ax.set_ylabel("Injection Timing (Degrees)")
ax.set_zlabel("Intake Pressure (Bar)")
ax.set_title("Latin Hypercube Sampling (LHS)")
plt.legend()
plt.show()

# Plot Pareto Front
plt.figure(figsize=(6, 5))
plt.scatter(pareto_front[:, 0], -pareto_front[:, 1], color="blue", label="Pareto Front Solutions")
plt.xlabel("Backfire Intensity (Minimize)")
plt.ylabel("Engine Performance (Maximize)")
plt.title("Pareto Front - NSGA-II Optimization")
plt.legend()
plt.grid()
plt.show()

# Display optimization results
print("Optimization Complete. Extracted Pareto Front Solutions:")
print(pareto_front)