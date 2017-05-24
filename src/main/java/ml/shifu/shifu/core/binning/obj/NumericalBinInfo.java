package ml.shifu.shifu.core.binning.obj;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Created by zhanhu on 5/11/17.
 */
public class NumericalBinInfo extends AbstractBinInfo implements Comparable<AbstractBinInfo> {

    private static final Logger LOG = LoggerFactory.getLogger(NumericalBinInfo.class);

    private double leftThreshold;
    private double rightThreshold;

    public double getLeftThreshold() {
        return leftThreshold;
    }

    public void setLeftThreshold(double leftThreshold) {
        this.leftThreshold = leftThreshold;
    }

    public double getRightThreshold() {
        return rightThreshold;
    }

    public void setRightThreshold(double rightThreshold) {
        this.rightThreshold = rightThreshold;
    }

    @Override
    public NumericalBinInfo clone() {
        NumericalBinInfo other = new NumericalBinInfo();
        other.setLeftThreshold(this.leftThreshold);
        other.setRightThreshold(this.rightThreshold);
        other.setNegativeCnt(this.getNegativeCnt());
        other.setPositiveCnt(this.getPositiveCnt());
        other.setWeightNeg(this.getWeightNeg());
        other.setWeightPos(this.getWeightPos());
        return other;
    }

    @Override
    public int compareTo(AbstractBinInfo binInfo) {
        if ( binInfo instanceof NumericalBinInfo ) {
            NumericalBinInfo numBinInfo = (NumericalBinInfo) binInfo;
            return Double.compare(leftThreshold, numBinInfo.getLeftThreshold());
        } else {
            return 0;
        }
    }

    @Override
    public void mergeRight(AbstractBinInfo binInfo) {
        if ( binInfo instanceof NumericalBinInfo ) {
            NumericalBinInfo numBinInfo = (NumericalBinInfo) binInfo;
            this.setRightThreshold(numBinInfo.getRightThreshold());
            this.positiveCnt = this.positiveCnt + numBinInfo.getPositiveCnt();
            this.negativeCnt = this.negativeCnt + numBinInfo.getNegativeCnt();
            this.weightPos = this.weightPos + numBinInfo.getWeightPos();
            this.weightNeg = this.weightNeg + numBinInfo.getWeightNeg();
        } else {
            LOG.warn("CategoricalBinInfo could only be merged with CategoricalBinInfo. Skip Merge");
        }
    }
}
