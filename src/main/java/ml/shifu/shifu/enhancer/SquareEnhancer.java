package ml.shifu.shifu.enhancer;

/**
 * SquareEnhancer will act as method Math.power(x,2).
 * <p>
 * Author: Wu Devin (haifwu@paypal.com)
 * Date: 20/01/2018
 */
public class SquareEnhancer implements EnhanceCallable<Double>{
    public Double enhance(Double originValue) {
        return Math.pow(originValue, 2);
    }
}
