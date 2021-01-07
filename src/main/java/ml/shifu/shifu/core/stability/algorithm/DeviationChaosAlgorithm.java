package ml.shifu.shifu.core.stability.algorithm;

import ml.shifu.shifu.container.obj.ColumnConfig;

/**
 * DeviationChaosAlgorithm stand for all the value being deviated.
 *
 * @author Wu Devin (haifwu@paypal.com)
 */
public class DeviationChaosAlgorithm extends BaseChaosAlgorithm {
    @Override
    String generateCategoryChaosValue(String originValue, ColumnConfig config) {
        if(config.getBinCategory().size() == 0) {
            return originValue;
        }
        return config.getBinCategory().get(config.getBinCategory().size() - 1);
    }

    @Override
    String generateNumericChaosValue(String originValue, ColumnConfig config) {
        return String.valueOf(Double.parseDouble(originValue) + config.getColumnStats().getMax());
    }
}
