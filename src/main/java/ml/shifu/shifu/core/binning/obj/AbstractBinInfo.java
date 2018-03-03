package ml.shifu.shifu.core.binning.obj;

/**
 * Created by zhanhu on 5/11/17.
 */
public abstract class AbstractBinInfo implements Comparable<AbstractBinInfo> {

    protected long negativeCnt;
    protected long positiveCnt;
    protected double weightPos;
    protected double weightNeg;

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

    public abstract void mergeRight(AbstractBinInfo binInfo);

    public abstract AbstractBinInfo clone();
}
