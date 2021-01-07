package ml.shifu.shifu.core.stability.algorithm;

import ml.shifu.shifu.container.obj.ColumnConfig;

/**
 * NullValueChaosAlgorithm return null for category value and 0 for numeric value.
 *
 * @author Wu Devin (haifwu@paypal.com)
 */
public class NullValueChaosAlgorithm extends BaseChaosAlgorithm {

    @Override
    String generateCategoryChaosValue(String originValue, ColumnConfig config) {
        return null;
    }

    @Override
    String generateNumericChaosValue(String originValue, ColumnConfig config) {
        return "0";
    }
}
