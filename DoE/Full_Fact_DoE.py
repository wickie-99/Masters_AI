from pyDOE2 import fullfact
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import math
from statsmodels.formula.api import ols
import statsmodels.api as sm
from mpl_toolkits.mplot3d import Axes3D

def create_full_factorial_design(factors, randomize=False):
    # Create a 2-level full factorial design
    design = fullfact([2]*len(factors))
    
    # Convert levels from 0/1 to -1/+1
    design = 2*design - 1
    
    # Convert design matrix to a DataFrame
    df = pd.DataFrame(design, columns=factors)
    
    # Randomize the design if needed
    if randomize:
        df = df.sample(frac=1).reset_index(drop=True)
   
    return df

def main_effects_plot(excel_file, result_column):
    # Load data from excel
    df = pd.read_excel(excel_file)
    
    # Identify factors by excluding the result column
    factors = [col for col in df.columns if col != result_column]
        
    # Calculate main effects
    main_effects = {}
    for factor in factors:
        mean_plus = df[df[factor] == 1][result_column].mean()
        mean_minus = df[df[factor] == -1][result_column].mean()
        main_effects[factor] = mean_plus - mean_minus

    # Plot main effects
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.bar(main_effects.keys(), main_effects.values())

    # Annotate the bars with their values
    for factor, value in main_effects.items():
        ax.text(factor, value + 0.01 * abs(value), '{:.2f}'.format(value), ha='center', va='bottom')

    ax.set_ylabel('Main Effect')
    ax.set_title('Main Effects Plot')
    plt.xticks(rotation=45, ha="right")
    plt.show()

def interaction_effects_plot(excel_file, result_column):
    df = pd.read_excel(excel_file)
    factors = [col for col in df.columns if col!= result_column]
    interactions = list(itertools.combinations(factors, 2))

    interaction_effects = []
    for interaction in interactions:
        grouped = df.groupby(list(interaction))[result_column].mean()
        interaction_effect = (grouped[1,1] - grouped[1,-1] - grouped[-1,1] + grouped[-1,-1])
        interaction_effects.append(interaction_effect)
    
    interaction_labels = [f"{x[0]} x {x[1]}" for x in interactions]
    fig, ax = plt.subplots(figsize = (5,5))
    bars = ax.bar(interaction_labels, interaction_effects)

    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.01*abs(yval),
                round(yval,2), ha='center', va='bottom')
    
    ax.set_ylabel("Interaction Effect")
    ax.set_title("Two-way Interaction Effects")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


def interaction_point_plot(excel_file, result_column):
    df = pd.read_excel(excel_file)
    factors = [col for col in df.columns if col != result_column]
    interactions = list(itertools.combinations(factors, 2))

    cols = 3
    rows = math.ceil(len(interactions) / cols)
    fig, axs = plt.subplots(rows, cols, figsize=(10, 10*rows/3))

    for idx, interaction in enumerate(interactions):
        row = idx // cols
        col = idx % cols

        ax = axs[row, col] if rows > 1 else axs[col]

        for level in [-1, 1]:
            subset = df[df[interaction[0]] == level]
            ax.plot(subset[interaction[1]].unique(), subset.groupby(interaction[1])[result_column].mean(), 'o-', label=f'{interaction[0]} = {level}')

        ax.set_title(f'{interaction[0]} x {interaction[1]}')
        ax.legend()
        ax.grid(True)

    for idx in range(len(interactions), rows*cols):
        row = idx // cols
        col = idx % cols
        axs[row, col].axis('off')

    plt.tight_layout()
    plt.show()

def diagnostic_plots(model):
    residuals = model.resid
    predicted = model.fittedvalues
    
    fig, axs = plt.subplots(1, 3, figsize=(10, 5))
    
    axs[0].scatter(predicted, residuals, edgecolors='k', facecolors='none')
    axs[0].axhline(y=0, color='k', linestyle='dashed', linewidth=1)
    axs[0].set_title('Residuals vs. Predicted')
    axs[0].set_xlabel('Predicted values')
    axs[0].set_ylabel('Residuals')
    
    axs[1].scatter(range(len(residuals)), residuals, edgecolors='k', facecolors='none')
    axs[1].axhline(y=0, color='k', linestyle='dashed', linewidth=1)
    axs[1].set_title('Residuals vs. Run')
    axs[1].set_xlabel('Run')
    axs[1].set_ylabel('Residuals')
    
    sm.qqplot(residuals, line='45', fit=True, ax=axs[2])
    axs[2].set_title('Q-Q Plot')
    
    plt.tight_layout()
    plt.show()

def plot_3D_surface(model, data, x_name, y_name, z_name, held_factor, held_value, title):
    x_range = np.linspace(-1, 1, 100)
    y_range = np.linspace(-1, 1, 100)
    x_grid, y_grid = np.meshgrid(x_range, y_range)

    df_pred = pd.DataFrame({
        x_name: x_grid.ravel(),
        y_name: y_grid.ravel(),
        held_factor: [held_value] * x_grid.size
    })
    df_pred = sm.add_constant(df_pred)
    Z = model.predict(df_pred).values.reshape(x_grid.shape)
    
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot_surface(x_grid, y_grid, Z, cmap='viridis', alpha=0.6)
    
    mask = data[held_factor] == held_value
    ax.scatter(data[mask][x_name], data[mask][y_name], data[mask][z_name], color='r', marker='o')
    
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.set_zlabel(z_name)
    ax.set_title(title)
    
    plt.show()

def plot_contour(x_name, y_name, held_factor, held_value, title, model):
    x_range = np.linspace(-1, 1, 100)
    y_range = np.linspace(-1, 1, 100)
    x_grid, y_grid = np.meshgrid(x_range, y_range)
    
    predictions = model.predict(pd.DataFrame({
        x_name: x_grid.ravel(),
        y_name: y_grid.ravel(),
        held_factor: held_value,
    }))
    
    Z = predictions.values.reshape(x_grid.shape)
    
    plt.figure(figsize=(7, 5))
    contour = plt.contourf(x_grid, y_grid, Z, 20, cmap='viridis')
    plt.colorbar(contour)
    plt.title(title)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.show()


print('Hello DoE')

factors = ['T', 'P', 'CoF', 'RPM']
df = create_full_factorial_design(factors, randomize=False)
df.to_excel('Full_Factorial_Design_2.xlsx', index=False)

excel_file = 'Full_Factorial_Design.xlsx'
results_column = 'Results'
main_effects_plot(excel_file, results_column)
interaction_effects_plot(excel_file, results_column)
interaction_point_plot(excel_file, results_column)

df = pd.read_excel(excel_file)

formula = 'Results ~ T + CoF + RPM + T:CoF + T:RPM + T:RPM:CoF'
model = ols(formula, data=df).fit()

anova_table = sm.stats.anova_lm(model, typ=1)
print(anova_table)

diagnostic_plots(model)
