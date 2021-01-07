package ml.shifu.shifu.core.stability.algorithm;

import ml.shifu.shifu.container.obj.ColumnConfig;

/**
 * Base implementation of ChaosAlgorithm, all sub class need to implement generate chaos value both for category
 * and numeric data.
 *
 * @author Wu Devin (haifwu@paypal.com)
 */
public abstract class BaseChaosAlgorithm implements ChaosAlgorithm {
    @Override
    public String generateChaosValue(String originValue, ColumnConfig config) {
        if(config.isCategorical()) {
            return generateCategoryChaosValue(originValue, config);
        }
        return generateNumericChaosValue(originValue, config);
    }

    abstract String generateCategoryChaosValue(String originValue, ColumnConfig config);
    abstract String generateNumericChaosValue(String originValue, ColumnConfig config);
}
