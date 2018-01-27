package ml.shifu.shifu.enhancer;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;

/**
 * EnhancerFactory is used to generate Enhancer according to the name.
 * <p>
 * Author: Wu Devin (haifwu@paypal.com)
 * Date: 20/01/2018
 */
public class EnhancerFactory {
    private static Map<EnhanceType, EnhanceCallable> enhancerTypeToEnhancerMap = new HashMap<EnhanceType, EnhanceCallable>(10);
    private static EnhancerFactory instance = new EnhancerFactory();

    private EnhancerFactory(){
    }

    public enum EnhanceType {
        LOG("log"),
        SQRT("sqrt"),
        SQUARE("square");

        private String enhanceType;

        EnhanceType(String enhanceType) {
            this.enhanceType = enhanceType;
        }

        public String getEnhanceType() {
            return this.enhanceType;
        }

        public static EnhanceType fromString(String enhanceType) {
            enhanceType = enhanceType.toLowerCase();
            for (EnhanceType type : EnhanceType.values()) {
                if (type.enhanceType.equalsIgnoreCase(enhanceType)) {
                    return type;
                }
            }
            return null;
        }
    }

    static {
        /**
         * currently only support three type of enhancer: log, sqrt, square
         */
        enhancerTypeToEnhancerMap.put(EnhanceType.LOG, new LogEnhancer());
        enhancerTypeToEnhancerMap.put(EnhanceType.SQRT, new SqrtEnhancer());
        enhancerTypeToEnhancerMap.put(EnhanceType.SQUARE, new SquareEnhancer());
    }

    public static EnhancerFactory getInstance(){
        return instance;
    }

    /**
     * Get current supported enhancer type.
     * @return A set contain all supported enhancer type.
     */
    public static Set<EnhanceType> getSupportedEnhancerType(){
        return enhancerTypeToEnhancerMap.keySet();
    }

    /**
     * Get enhancer by enhance type.
     *
     * @param enhancerType, the enhance type
     * @return enhancer instance
     */
    public EnhanceCallable getEnhancer(EnhanceType enhancerType) {
        if(enhancerType == null){
            throw new IllegalArgumentException("Try to get enhancer while enhancer type is null!");
        }
        if(! enhancerTypeToEnhancerMap.containsKey(enhancerType)){
            throw new IllegalArgumentException("Unsupported enhancer type: " + enhancerType);
        }
        return enhancerTypeToEnhancerMap.get(enhancerType);
    }

    /**
     * Get enhancer by enhance type.
     *
     * @param enhancerTypeStr, the String format enhance type
     * @return enhancer instance
     */
    public EnhanceCallable getEnhancer(String enhancerTypeStr) {
        EnhanceType enhanceType = EnhanceType.fromString(enhancerTypeStr);
        if(enhanceType == null){
            throw new IllegalArgumentException("enhance type not find " + enhancerTypeStr);
        }
        return getEnhancer(enhanceType);
    }
}
