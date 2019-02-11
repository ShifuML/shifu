package ml.shifu.shifu.core.dtrain.wnd;

/**
 * Class implement this interface should have a method initWeight.
 * <p>
 * @author : Wu Devin (haifwu@paypal.com)
 */
public interface WeightInitializable {
    /**
     * Init weight according to the policy
     * @param policy, the init policy
     */
    void initWeight(String policy);
}

