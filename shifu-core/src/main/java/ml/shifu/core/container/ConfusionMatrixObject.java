package ml.shifu.core.container;

public class ConfusionMatrixObject {

    private double tp, fp, tn, fn, weightedTp, weightedFp, weightedTn, weightedFn;
    private double score;

    public ConfusionMatrixObject() {
        this.tp = 0.0;
        this.fn = 0.0;
        this.fp = 0.0;
        this.tn = 0.0;
        this.weightedFn = 0.0;
        this.weightedFp = 0.0;
        this.weightedTn = 0.0;
        this.weightedTp = 0.0;
    }

    public ConfusionMatrixObject(ConfusionMatrixObject cmo) {
        this.tp = cmo.tp;
        this.fn = cmo.fn;
        this.fp = cmo.fp;
        this.tn = cmo.tn;
        this.weightedFn = cmo.weightedFn;
        this.weightedFp = cmo.weightedFp;
        this.weightedTn = cmo.weightedTn;
        this.weightedTp = cmo.weightedTp;
    }

    public double getTp() {
        return tp;
    }

    public void setTp(double tp) {
        this.tp = tp;
    }

    public double getFp() {
        return fp;
    }

    public void setFp(double fp) {
        this.fp = fp;
    }

    public double getTn() {
        return tn;
    }

    public void setTn(double tn) {
        this.tn = tn;
    }

    public double getFn() {
        return fn;
    }

    public void setFn(double fn) {
        this.fn = fn;
    }

    public double getWeightedTp() {
        return weightedTp;
    }

    public void setWeightedTp(double weightedTp) {
        this.weightedTp = weightedTp;
    }

    public double getWeightedFp() {
        return weightedFp;
    }

    public void setWeightedFp(double weightedFp) {
        this.weightedFp = weightedFp;
    }

    public double getWeightedTn() {
        return weightedTn;
    }

    public void setWeightedTn(double weightedTn) {
        this.weightedTn = weightedTn;
    }

    public double getWeightedFn() {
        return weightedFn;
    }

    public void setWeightedFn(double weightedFn) {
        this.weightedFn = weightedFn;
    }

    public double getTotal() {
        return this.tp + this.tn + this.fn + this.fp;
    }

    public double getWeightedTotal() {
        return this.weightedTp + this.weightedTn + this.weightedFn + this.weightedFp;
    }

    public double getPosTotal() {
        return this.getTp() + this.getFn();
    }

    public double getWeightPosTotal() {
        return this.getWeightedTp() + this.getWeightedFn();
    }

    public double getScore() {
        return score;
    }

    public void setScore(double score) {
        this.score = score;
    }

}
