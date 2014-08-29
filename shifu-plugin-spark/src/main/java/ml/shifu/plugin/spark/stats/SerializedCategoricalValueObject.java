package ml.shifu.plugin.spark.stats;

import java.io.Serializable;
import java.util.Comparator;

import ml.shifu.core.container.CategoricalValueObject;

/*
 * Serialized version of ml.shifu.core.container.CategoricalValueObject 
 */
public class SerializedCategoricalValueObject
        implements Serializable {

    private static final long serialVersionUID = 1L;

    private Boolean isPositive;
    private String value;
    private Double weight;
    
    public SerializedCategoricalValueObject(String value, Double weight, Boolean isPositive) {
        this.isPositive= isPositive;
        this.weight= weight;
        this.value= value;
    }
    
    public SerializedCategoricalValueObject() {
        this.weight = 1.0;
    }
    
    public Boolean getIsPositive() {
        return isPositive;
    }

    public void setIsPositive(Boolean isPositive) {
        this.isPositive = isPositive;
    }

    public String getValue() {
        return value;
    }

    public void setValue(String value) {
        this.value = value;
    }

    public Double getWeight() {
        return weight;
    }

    public void setWeight(Double weight) {
        this.weight = weight;
    }

    public static class CategoricalValueObjectComparator implements Comparator<CategoricalValueObject> {

        public int compare(CategoricalValueObject a, CategoricalValueObject b) {
            int d = a.getValue().compareTo(b.getValue());
            if (d == 0) {
                return a.getIsPositive().compareTo(b.getIsPositive());
            } else {
                return d;
            }
        }
    }

}
