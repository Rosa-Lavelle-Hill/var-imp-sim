from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

def define_model(pred_model, fixed_seed):
    """
    # Defines model and corresponding parameter grid for training
    :param pred_model: string, either: "enet" for elastic net regression, "lasso" for lasso regression,
    "tree" for a decision tree, or "rf" for a random forest
    :param fixed_seed: integer defining random seed
    :return: untrained sklearn model, a corresponding parameter grid containing parameter values which will used to train
    """
    from Params.Grids import enet_param_grid, lasso_param_grid, rf_param_grid, tree_param_grid
    if pred_model == "enet":
        model = ElasticNet(random_state=fixed_seed)
        param_grid = enet_param_grid
    elif pred_model == "lasso":
        model = Lasso(random_state=fixed_seed)
        param_grid = lasso_param_grid
    elif pred_model == "rf":
        model = RandomForestRegressor(random_state=fixed_seed)
        param_grid = rf_param_grid
    elif pred_model == "tree":
        model = DecisionTreeRegressor(random_state=fixed_seed)
        param_grid = tree_param_grid
    else:
        print("Error: please define pred_model param as one of 'enet', 'lasso', 'tree', or 'rf'")
        breakpoint()
    return model, param_grid