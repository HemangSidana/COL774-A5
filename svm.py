import sys
import numpy as np
import cvxpy as cp
import pandas as pd
import json
from sklearn.preprocessing import StandardScaler

def main():
    if len(sys.argv) != 2:
        print("Usage: python svm.py <train_filename.csv>")
        sys.exit(1)

    # Parse filename
    train_file = sys.argv[1]
    anystring = train_file.split('.')[0].replace("train_", "")
    
    # Define output file names
    weight_file = f"weight_{anystring}.json"
    sv_file = f"sv_{anystring}.json"

    data = pd.read_csv(train_file)
    X = data.drop('target', axis=1).values
    Y = data['target'].values
    Y = np.where(Y==0,-1,1)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    n, d = X.shape

    w = cp.Variable(d)        
    b = cp.Variable()        
    slack = cp.Variable(n)   

    C = 1  
    objective = cp.Minimize(cp.norm1(w)/2 + C * cp.sum(slack))

    constraints = []
    for i in range(n):
        constraints.append(Y[i] * (X[i] @ w + b) >= 1 - slack[i]) 
        constraints.append(slack[i] >= 0) 

    problem = cp.Problem(objective, constraints)
    problem.solve()


    output = {
        "weights": w.value.tolist(),  
        "bias": b.value.item()  
    }

    with open(weight_file, "w") as json_file:
        json.dump(output, json_file, indent=2)
    

    separable = 0
    support_vectors= []
    if max(slack.value)<1:
        separable = 1
        for i in range(n):
            margin_distance = Y[i] * (X[i] @ w.value + b.value)
            if np.isclose(margin_distance, 1, atol=1e-4):  
                support_vectors.append(i)

    output = {
        "seperable": separable,  
        "support_vectors": support_vectors
    }

    with open(sv_file, "w") as json_file:
        json.dump(output, json_file, indent=2)

if __name__ == "__main__":
    main()
