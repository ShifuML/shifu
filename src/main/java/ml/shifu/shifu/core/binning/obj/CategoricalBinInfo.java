package ml.shifu.shifu.core.binning.obj;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by zhanhu on 4/18/17.
 */
public class CategoricalBinInfo extends AbstractBinInfo implements Comparable<AbstractBinInfo>{
    private static final Logger LOG = LoggerFactory.getLogger(CategoricalBinInfo.class);

    private List<String> values;

    public List<String> getValues() {
        return values;
    }
    public void setValues(List<String> values) {
        this.values = values;
    }

    @Override
    public int compareTo(AbstractBinInfo other) {
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

    @Override
    public void mergeRight(AbstractBinInfo next) {
        if ( next instanceof  CategoricalBinInfo ) {
            CategoricalBinInfo cbinInfo = (CategoricalBinInfo) next;
            this.values.addAll(cbinInfo.getValues());
            this.positiveCnt = this.positiveCnt + cbinInfo.getPositiveCnt();
            this.negativeCnt = this.negativeCnt + cbinInfo.getNegativeCnt();
            this.weightPos = this.weightPos + cbinInfo.getWeightPos();
            this.weightNeg = this.weightNeg + cbinInfo.getWeightNeg();
        } else {
            LOG.warn("CategoricalBinInfo could only be merged with CategoricalBinInfo. Skip Merge");
        }
    }
}
