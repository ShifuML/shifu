package ml.shifu.plugin.spark.stats;

import java.io.Serializable;
import java.util.Comparator;

import ml.shifu.core.container.NumericalValueObject;
/*
 * Serialized version of ml.shifu.core.container.NumericalValueObject
 */
public class SerializedNumericalValueObject
        implements Serializable {

    private static final long serialVersionUID = 1L;

    private Boolean isPositive;
    private Double value;
    private Double weight;
    public SerializedNumericalValueObject(Double value, Boolean isPositive, Double weight) {
        this.isPositive= isPositive;
        this.value= value;
        this.weight= weight;
    }

    public SerializedNumericalValueObject() {
        this.weight = 1.0;
    }

    public Boolean getIsPositive() {
        return isPositive;
    }

    public void setIsPositive(Boolean isPositive) {
        this.isPositive = isPositive;
    }

    public Double getValue() {
        return value;
    }

    public void setValue(Double value) {
        this.value = value;
    }

    public Double getWeight() {
        return weight;
    }

    public void setWeight(Double weight) {
        this.weight = weight;
    }

    public static class NumericalValueObjectComparator implements Comparator<NumericalValueObject> {

        public int compare(NumericalValueObject a, NumericalValueObject b) {
            int d = a.getValue().compareTo(b.getValue());
            if (d == 0) {
                return a.getIsPositive().compareTo(b.getIsPositive());
            } else {
                return d;
            }
        }
    }


}
