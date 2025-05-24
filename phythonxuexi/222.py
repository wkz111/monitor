import numpy as np
from graphviz import Digraph
from sklearn.tree import DecisionTreeClassifier
from geneticalgorithm import geneticalgorithm as ga
from sklearn.model_selection import cross_val_score


def ga_optimize_dt(X, y):
    """
    遗传算法驱动决策树参数优化
    :param X: 特征矩阵（含AE RMS、温度梯度等）
    :param y: 标签（0-正常,1-轻度,2-重度）
    :return: 最优参数（熵阈值, AE权重）
    """
    # 参数搜索空间（网页30阈值设定）
    varbound =np.array([[0.3, 0.5],  # 熵阈值（信息增益分裂下限）
                         [0.5, 0.7]])  # AE特征权重占比（网页30中AE占60%）

    def fitness(params):
        entropy_thresh = params[0]
        ae_weight = params[1]
        # 特征加权融合（网页34多模态融合策略）
        X_weighted = X * [ae_weight, 1 - ae_weight]  # 假设X为[AE特征,温度特征]
        clf = DecisionTreeClassifier(
            criterion='entropy',
            max_depth=5,
            min_impurity_decrease=entropy_thresh
        )
        # 5折交叉验证准确率作为适应度（网页30验证方法）
        accuracy = cross_val_score(clf, X_weighted, y, cv=5).mean()
        return -accuracy  # 最小化负准确率

    # 遗传算法配置（网页21参数）
    algorithm_param = {
        'max_num_iteration': 200,
        'population_size': 20,
        'mutation_probability': 0.1,
        'elit_ratio': 0.01,
        'parents_portion': 0.3
    }
    model = ga(
        function=fitness,
        dimension=2,
        variable_type='real',
        variable_boundaries=varbound,
        algorithm_parameters=algorithm_param
    )
    model.run()
    return model.output_dict['variable']