import joblib


class MultiParamDecisionTree:
    def __init__(self, model_path='ga_ddt_model.pkl'):
        self.model = joblib.load(model_path)
        self.ae_weight = 0.6  # 默认AE权重（网页30优化结果）
        self.temp_weight = 0.4

    import joblib

    class MultiParamDecisionTree:
        def __init__(self, model_path='ga_ddt_model.pkl'):
            self.model = joblib.load(model_path)
            self.ae_weight = 0.6  # 默认AE权重（网页30优化结果）
            self.temp_weight = 0.4

        def predict(self, features):
            """
            加权特征决策树推理（网页34融合策略）
            :param features: 输入特征[AE_RMS, Temp_Gradient]
            :return: 磨损等级
            """
            weighted_features = np.array(features) * [self.ae_weight, self.temp_weight]
            return self.model.predict(weighted_features.reshape(1, -1))[0]
    def predict(self, features):
        """
        加权特征决策树推理（网页34融合策略）
        :param features: 输入特征[AE_RMS, Temp_Gradient]
        :return: 磨损等级
        """
        weighted_features = np.array(features) * [self.ae_weight, self.temp_weight]
        return self.model.predict(weighted_features.reshape(1, -1))[0]