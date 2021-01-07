package ml.shifu.shifu.core.stability;

import ml.shifu.shifu.core.stability.algorithm.ChaosAlgorithm;
import ml.shifu.shifu.core.stability.algorithm.DeviationChaosAlgorithm;
import ml.shifu.shifu.core.stability.algorithm.NullValueChaosAlgorithm;
import ml.shifu.shifu.core.stability.algorithm.RandomChaosAlgorithm;

/**
 * Enum define different chaos type. For each chaos type, we should consider different logical for both category and
 * numeric value transfer behaviours.
 *
 * @author Wu Devin (haifwu@paypal.com)
 */
public enum ChaosType {
    NULL_VALUE("null", new NullValueChaosAlgorithm(), "return null for category value and return"),
    RANDOM_VALUE("random", new RandomChaosAlgorithm(), "return random value for numeric, random choose category value from existing category type"),
    DEVIATION_VALUE("deviation", new DeviationChaosAlgorithm(), "Return deviation value for both category and numeric");

    private String name;
    private String description;
    private ChaosAlgorithm chaosAlgorithm;

    ChaosType(String name, ChaosAlgorithm chaosAlgorithm, String description) {
        this.name = name;
        this.chaosAlgorithm = chaosAlgorithm;
        this.description = description;
    }

    public String getName() {
        return name;
    }

    public String getDescription() {
        return description;
    }

    public ChaosAlgorithm getChaosAlgorithm() {
        return chaosAlgorithm;
    }

    public static ChaosType fromName(String name) {
        for(ChaosType chaosType: ChaosType.values()) {
            if(chaosType.getName().equalsIgnoreCase(name)) {
                return chaosType;
            }
        }
        return null;
    }
}
