package ml.shifu.core.container;


import com.fasterxml.jackson.annotation.JsonIgnore;

public class BinaryConfusionMatrix {

    private Double tp;
    private Double tn;
    private Double fp;
    private Double fn;
    private Double weightedTp;
    private Double weightedTn;
    private Double weightedFp;
    private Double weightedFn;

    private Double precision;
    private Double recall;
    private Double falsePositiveRate;
    private Double actionRate;


    private Double weightedPrecision;
    private Double weightedRecall;
    private Double weightedFalsePositiveRate;
    private Double weightedActionRate;


    @JsonIgnore
    public void calculatePrecision() {
        this.precision = tp / (tp + fp);
        this.weightedPrecision = weightedTp / (weightedTp + weightedFp);
    }

    @JsonIgnore
    public void calculateRecall() {
        this.recall = tp / (tp + fn);
        this.weightedRecall = weightedTp / (weightedTp + weightedFn);
    }

    @JsonIgnore
    public void calculateFalsePositiveRate() {
        this.falsePositiveRate = fp / (tn + fp);
        this.weightedFalsePositiveRate = weightedFp / (weightedTn + weightedFp);
    }

    @JsonIgnore
    public void calculateActionRate() {
        this.actionRate = (tp + fp) / (tp + fp + tn + fn);
        this.weightedActionRate = (weightedTp + weightedFp) / (weightedTp + weightedFp + weightedTn + weightedFn);
    }


    public Double getWeightedPrecision() {
        return weightedPrecision;
    }

    public void setWeightedPrecision(Double weightedPrecision) {
        this.weightedPrecision = weightedPrecision;
    }

    public Double getWeightedRecall() {
        return weightedRecall;
    }

    public void setWeightedRecall(Double weightedRecall) {
        this.weightedRecall = weightedRecall;
    }

    public Double getWeightedFalsePositiveRate() {
        return weightedFalsePositiveRate;
    }

    public void setWeightedFalsePositiveRate(Double weightedFalsePositiveRate) {
        this.weightedFalsePositiveRate = weightedFalsePositiveRate;
    }

    public Double getWeightedActionRate() {
        return weightedActionRate;
    }

    public void setWeightedActionRate(Double weightedActionRate) {
        this.weightedActionRate = weightedActionRate;
    }


    public Double getRecall() {
        return recall;
    }

    public void setRecall(Double recall) {
        this.recall = recall;
    }

    public Double getFalsePositiveRate() {
        return falsePositiveRate;
    }

    public void setFalsePositiveRate(Double falsePositiveRate) {
        this.falsePositiveRate = falsePositiveRate;
    }

    public Double getActionRate() {
        return actionRate;
    }

    public void setActionRate(Double actionRate) {
        this.actionRate = actionRate;
    }

    public Double getPrecision() {
        return precision;
    }

    public void setPrecision(Double precision) {
        this.precision = precision;
    }


    public Double getWeightedFn() {
        return weightedFn;
    }

    public void setWeightedFn(Double weightedFn) {
        this.weightedFn = weightedFn;
    }

    public Double getTp() {
        return tp;
    }

    public void setTp(Double tp) {
        this.tp = tp;
    }

    public Double getTn() {
        return tn;
    }

    public void setTn(Double tn) {
        this.tn = tn;
    }

    public Double getFp() {
        return fp;
    }

    public void setFp(Double fp) {
        this.fp = fp;
    }

    public Double getFn() {
        return fn;
    }

    public void setFn(Double fn) {
        this.fn = fn;
    }

    public Double getWeightedTp() {
        return weightedTp;
    }

    public void setWeightedTp(Double weightedTp) {
        this.weightedTp = weightedTp;
    }

    public Double getWeightedTn() {
        return weightedTn;
    }

    public void setWeightedTn(Double weightedTn) {
        this.weightedTn = weightedTn;
    }

    public Double getWeightedFp() {
        return weightedFp;
    }

    public void setWeightedFp(Double weightedFp) {
        this.weightedFp = weightedFp;
    }


}
