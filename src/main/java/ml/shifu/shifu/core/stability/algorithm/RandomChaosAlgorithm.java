package ml.shifu.shifu.core.stability.algorithm;

import ml.shifu.shifu.container.obj.ColumnConfig;

import java.util.Random;

/**
 * Generate random value for both category and numeric.
 *
 * @author Wu Devin (haifwu@paypal.com)
 */
public class RandomChaosAlgorithm extends BaseChaosAlgorithm {
    private static Random random = new Random(System.currentTimeMillis());

    @Override
    String generateCategoryChaosValue(String originValue, ColumnConfig config) {
        int cnt = config.getBinCategory().size();
        return config.getBinCategory().get(random.nextInt(cnt));
    }

    @Override
    String generateNumericChaosValue(String originValue, ColumnConfig config) {
        return String.valueOf(random.nextDouble());
    }
}
