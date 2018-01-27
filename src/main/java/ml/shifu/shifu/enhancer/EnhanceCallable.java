package ml.shifu.shifu.enhancer;

/**
 * EnhanceCallable is used to define basic interface for enhance data.
 *
 * <p>
 * Author: Wu Devin (haifwu@paypal.com)
 * Date: 20/01/2018
 */
public interface EnhanceCallable<Double> {
    Double enhance(Double originValue);
}
