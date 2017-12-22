package ml.shifu.shifu.core.binning;

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.core.ColumnStatsCalculator;
import ml.shifu.shifu.core.binning.obj.AbstractBinInfo;
import ml.shifu.shifu.core.binning.obj.CategoricalBinInfo;
import ml.shifu.shifu.core.binning.obj.NumericalBinInfo;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Created by zhanhu on 5/8/17.
 */
public class ColumnConfigDynamicBinning {

    public static Logger LOG = LoggerFactory.getLogger(ColumnConfigDynamicBinning.class);

    private ColumnConfig columnConfig;
    private double ivKeepRatio;
    private long minimumInstCnt;
    private int expectMaxBinNum;

    public ColumnConfigDynamicBinning(ColumnConfig columnConfig, int expectMaxBinNum, double ivKeepRatio,
            long minimumInstCnt) {
        this.columnConfig = columnConfig;
        this.expectMaxBinNum = expectMaxBinNum;
        this.ivKeepRatio = ivKeepRatio;
        this.minimumInstCnt = minimumInstCnt;
    }

    public List<AbstractBinInfo> run() {
        List<AbstractBinInfo> binInfos = genBinInfos(this.columnConfig);
        Collections.sort(binInfos);

        // reduce bin number to not exceed expectMaxBinNum
        if(this.expectMaxBinNum > 0) {
            AutoDynamicBinning autoDynamicBinning = new AutoDynamicBinning(this.expectMaxBinNum);
            binInfos = autoDynamicBinning.merge(binInfos);
        }

        // filter and merge bi
        if(this.minimumInstCnt > 0) {
            binInfos = mergeSmallBinInfos(binInfos);
        }

        AbstractBinInfo missingBinInfo = genMissingBinInfo();
        double maxVarIv = calculateIv(binInfos, missingBinInfo);
        boolean isToContinue = true;

        while(isToContinue) {
            int nextBinNum = binInfos.size() - 1;
            AutoDynamicBinning autoDynamicBinning = new AutoDynamicBinning(nextBinNum);
            List<AbstractBinInfo> newBinInfos = autoDynamicBinning.merge(cloneBinInfoList(binInfos));

            double currentVarIv = calculateIv(newBinInfos, missingBinInfo);
            if(newBinInfos.size() == binInfos.size() // bin number is not decreased
                    || currentVarIv < maxVarIv * this.ivKeepRatio) { // current is less than (keepRatio * maxIv)
                isToContinue = false;
            } else {
                binInfos = newBinInfos;
            }
        }

        return binInfos;
    }

    private List<AbstractBinInfo> cloneBinInfoList(List<AbstractBinInfo> binInfos) {
        List<AbstractBinInfo> copyBinInfos = new ArrayList<AbstractBinInfo>();
        for(AbstractBinInfo binInfo: binInfos) {
            copyBinInfos.add(binInfo.clone());
        }
        return copyBinInfos;
    }

    private double calculateIv(List<AbstractBinInfo> binInfos, AbstractBinInfo missingBinInfo) {
        long[] binCountNeg = new long[binInfos.size() + 1];
        long[] binCountPos = new long[binInfos.size() + 1];
        for(int i = 0; i < binInfos.size(); i++) {
            AbstractBinInfo binInfo = binInfos.get(i);
            binCountNeg[i] = binInfo.getNegativeCnt();
            binCountPos[i] = binInfo.getPositiveCnt();
        }
        binCountNeg[binCountNeg.length - 1] = missingBinInfo.getNegativeCnt();
        binCountPos[binCountPos.length - 1] = missingBinInfo.getPositiveCnt();
        ColumnStatsCalculator.ColumnMetrics columnMetrics = ColumnStatsCalculator.calculateColumnMetrics(binCountNeg,
                binCountPos);

        return columnMetrics.getIv();
    }

    private List<AbstractBinInfo> mergeSmallBinInfos(List<AbstractBinInfo> binInfos) {
        int i = 0;
        while(i < binInfos.size()) {
            AbstractBinInfo binInfo = binInfos.get(i);
            if(this.minimumInstCnt > 0 && binInfo.getTotalInstCnt() < this.minimumInstCnt && binInfos.size() > 1) {
                if(i == 0) {
                    AbstractBinInfo nextBinInfo = binInfos.get(i + 1);
                    binInfo.mergeRight(nextBinInfo);
                    binInfos.remove(i + 1);
                } else if(i == binInfos.size() - 1) {
                    AbstractBinInfo prevBinInfo = binInfos.get(i - 1);
                    prevBinInfo.mergeRight(binInfo);
                    binInfos.remove(i);
                } else {
                    AbstractBinInfo prevBinInfo = binInfos.get(i - 1);
                    AbstractBinInfo nextBinInfo = binInfos.get(i + 1);
                    double prDeltaLeft = Math.abs(prevBinInfo.getPositiveRate() - binInfo.getPositiveRate());
                    double prDeltaRight = Math.abs(binInfo.getPositiveRate() - nextBinInfo.getPositiveRate());
                    if(prDeltaLeft < prDeltaRight) {
                        prevBinInfo.mergeRight(binInfo);
                        binInfos.remove(i);
                    } else {
                        binInfo.mergeRight(nextBinInfo);
                        binInfos.remove(i + 1);
                    }
                }
            } else {
                i++;
            }
        }
        return binInfos;
    }

    private List<AbstractBinInfo> genBinInfos(ColumnConfig columnConfig) {
        if(columnConfig.isCategorical()) {
            return genCategoricalBinInfos(columnConfig);
        } else {
            return genNumericalBinInfos(columnConfig);
        }
    }

    private List<AbstractBinInfo> genCategoricalBinInfos(ColumnConfig columnConfig) {
        List<AbstractBinInfo> categoricalBinInfos = new ArrayList<AbstractBinInfo>();
        for(int i = 0; i < columnConfig.getBinCategory().size(); i++) {
            CategoricalBinInfo binInfo = new CategoricalBinInfo();
            List<String> values = new ArrayList<String>();
            values.add(columnConfig.getBinCategory().get(i));
            binInfo.setValues(values);
            binInfo.setPositiveCnt(columnConfig.getBinCountPos().get(i));
            binInfo.setNegativeCnt(columnConfig.getBinCountNeg().get(i));
            binInfo.setWeightPos(columnConfig.getBinWeightedPos().get(i));
            binInfo.setWeightNeg(columnConfig.getBinWeightedNeg().get(i));

            categoricalBinInfos.add(binInfo);
        }

        return categoricalBinInfos;
    }

    private List<AbstractBinInfo> genNumericalBinInfos(ColumnConfig columnConfig) {
        List<AbstractBinInfo> numericalBinInfos = new ArrayList<AbstractBinInfo>();
        if(columnConfig.getBinBoundary() == null) {
            LOG.info("column bin boundary is null, index {}, name {}.", columnConfig.getColumnNum(),
                    columnConfig.getColumnName());
        }
        for(int i = 0; i < columnConfig.getBinBoundary().size(); i++) {
            NumericalBinInfo binInfo = new NumericalBinInfo();
            binInfo.setLeftThreshold(columnConfig.getBinBoundary().get(i));
            if(i == columnConfig.getBinBoundary().size() - 1) {
                binInfo.setRightThreshold(Double.POSITIVE_INFINITY);
            } else {
                binInfo.setRightThreshold(columnConfig.getBinBoundary().get(i + 1));
            }

            binInfo.setPositiveCnt(columnConfig.getBinCountPos().get(i));
            binInfo.setNegativeCnt(columnConfig.getBinCountNeg().get(i));
            binInfo.setWeightPos(columnConfig.getBinWeightedPos().get(i));
            binInfo.setWeightNeg(columnConfig.getBinWeightedNeg().get(i));

            numericalBinInfos.add(binInfo);
        }

        return numericalBinInfos;
    }

    private AbstractBinInfo genMissingBinInfo() {
        AbstractBinInfo binInfo = null;
        // add missing binning
        if(this.columnConfig.isCategorical()) {
            binInfo = new CategoricalBinInfo();
        } else {
            binInfo = new NumericalBinInfo();
        }
        binInfo.setPositiveCnt(columnConfig.getBinCountPos().get(columnConfig.getBinCountPos().size() - 1));
        binInfo.setNegativeCnt(columnConfig.getBinCountNeg().get(columnConfig.getBinCountNeg().size() - 1));
        binInfo.setWeightPos(columnConfig.getBinWeightedPos().get(columnConfig.getBinWeightedPos().size() - 1));
        binInfo.setWeightNeg(columnConfig.getBinWeightedNeg().get(columnConfig.getBinWeightedNeg().size() - 1));
        return binInfo;
    }
}
