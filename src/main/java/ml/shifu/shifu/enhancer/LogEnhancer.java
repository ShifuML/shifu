package ml.shifu.shifu.enhancer;

/**
 * LogEnhancer will be act as Math.log(x).
 * <p>
 * Author: Wu Devin (haifwu@paypal.com)
 * Date: 20/01/2018
 */
public class LogEnhancer implements EnhanceCallable<Double>{
    @Override
    public Double enhance(Double originValue) {
        return Math.log(originValue);
    }
}
