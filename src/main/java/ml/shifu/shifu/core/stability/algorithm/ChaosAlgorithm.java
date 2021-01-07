package ml.shifu.shifu.core.stability.algorithm;


import ml.shifu.shifu.container.obj.ColumnConfig;

/**
 * ChaosAlgorithm interface.
 *
 * @author Wu Devin (haifwu@paypal.com)
 */
public interface ChaosAlgorithm {

    /**
     * Generate chaos value from origin value.
     *
     * @param originValue  the origin value
     * @param config the column config contain stats info
     * @return chaos value
     */
    String generateChaosValue(String originValue, ColumnConfig config);
}
