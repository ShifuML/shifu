package ml.shifu.shifu.core.binning;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by zhanhu on 4/18/17.
 */
public class CategoricalBinInfo implements Comparable<CategoricalBinInfo>{
    private List<String> values;
    private long negativeCnt;
    private long positiveCnt;
    private double weightPos;
    private double weightNeg;

    public List<String> getValues() {
        return values;
    }

    public void setValues(List<String> values) {
        this.values = values;
    }

    public long getNegativeCnt() {
        return negativeCnt;
    }

    public void setNegativeCnt(long negativeCnt) {
        this.negativeCnt = negativeCnt;
    }

    public long getPositiveCnt() {
        return positiveCnt;
    }

    public void setPositiveCnt(long positiveCnt) {
        this.positiveCnt = positiveCnt;
    }

    public double getWeightPos() {
        return weightPos;
    }

    public void setWeightPos(double weightPos) {
        this.weightPos = weightPos;
    }

    public double getWeightNeg() {
        return weightNeg;
    }

    public void setWeightNeg(double weightNeg) {
        this.weightNeg = weightNeg;
    }

    public double getPositiveRate() {
        assert getTotalInstCnt() != 0;
        return ((double) positiveCnt) / getTotalInstCnt();
    }

    public long getTotalInstCnt() {
        return getNegativeCnt() + getPositiveCnt();
    }

    @Override
    public int compareTo(CategoricalBinInfo other) {
        return Double.compare(getPositiveRate(), other.getPositiveRate());
    }

    @Override
    public CategoricalBinInfo clone() {
        CategoricalBinInfo other = new CategoricalBinInfo();
        other.setNegativeCnt(this.getNegativeCnt());
        other.setPositiveCnt(this.getPositiveCnt());
        other.setWeightNeg(this.getWeightNeg());
        other.setWeightPos(this.getWeightPos());
        other.setValues(new ArrayList<String>(this.getValues()));
        return other;
    }

    public void mergeRight(CategoricalBinInfo next) {
        this.values.addAll(next.getValues());
        this.positiveCnt = this.positiveCnt + next.getPositiveCnt();
        this.negativeCnt = this.negativeCnt + next.getNegativeCnt();
        this.weightPos = this.weightPos + next.getWeightPos();
        this.weightNeg = this.weightNeg + next.getWeightNeg();
    }
}
