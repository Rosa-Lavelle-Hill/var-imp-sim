from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor

def define_model(pred_model, fixed_seed):
    # Define model
    from Params.Grids import enet_param_grid, lasso_param_grid, rf_param_grid
    if pred_model == "enet":
        model = ElasticNet(random_state=fixed_seed)
        param_grid = enet_param_grid
    elif pred_model == "lasso":
        model = Lasso(alpha=1, max_iter=500, random_state=fixed_seed)
        param_grid = lasso_param_grid
    elif pred_model == "rf":
        model = RandomForestRegressor(random_state=fixed_seed)
        param_grid = rf_param_grid
    else:
        print("Error: please define pred_model param as one of 'enet', 'lasso', or 'rf'")
        breakpoint()
    return model, param_grid