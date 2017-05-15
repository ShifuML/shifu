package ml.shifu.shifu.core.binning;

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.core.ColumnStatsCalculator;
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

    public ColumnConfigDynamicBinning(ColumnConfig columnConfig,
                                      int expectMaxBinNum, double ivKeepRatio, long minimumInstCnt) {
        this.columnConfig = columnConfig;
        this.expectMaxBinNum = expectMaxBinNum;
        this.ivKeepRatio = ivKeepRatio;
        this.minimumInstCnt = minimumInstCnt;
    }

    public List<CategoricalBinInfo> run() {
        List<CategoricalBinInfo> binInfos = genCategoricalBinInfos(this.columnConfig);
        Collections.sort(binInfos);

        // reduce bin number to not exceed expectMaxBinNum
        if (this.expectMaxBinNum > 0) {
            CateDynamicBinning cateDynamicBinning = new CateDynamicBinning(this.expectMaxBinNum);
            binInfos = cateDynamicBinning.merge(binInfos);
        }

        // filter and merge bi
        if ( this.minimumInstCnt > 0 ) {
            binInfos = mergeSmallBinInfos(binInfos);
        }

        CategoricalBinInfo missingBinInfo = genMissingBinInfo();
        double maxVarIv = calculateIv(binInfos, missingBinInfo);
        boolean isToContinue = true;

        while (isToContinue) {
            int nextBinNum = binInfos.size() - 1;
            CateDynamicBinning cateDynamicBinning = new CateDynamicBinning(nextBinNum);
            List<CategoricalBinInfo> newBinInfos = cateDynamicBinning.merge(cloneBinInfoList(binInfos));

            double currentVarIv = calculateIv(newBinInfos, missingBinInfo);
            LOG.info("New bin number is : {} with IV : {}, while maxVarIv is {}",
                    newBinInfos.size(), currentVarIv, maxVarIv);
            if (currentVarIv > maxVarIv * this.ivKeepRatio) {
                binInfos = newBinInfos;
            } else {
                isToContinue = false;
            }
        }

        return binInfos;
    }

    private List<CategoricalBinInfo> cloneBinInfoList(List<CategoricalBinInfo> binInfos) {
        List<CategoricalBinInfo> copyBinInfos = new ArrayList<CategoricalBinInfo>();
        for (CategoricalBinInfo binInfo : binInfos) {
            copyBinInfos.add(binInfo.clone());
        }
        return copyBinInfos;
    }

    private double calculateIv(List<CategoricalBinInfo> binInfos, CategoricalBinInfo missingBinInfo) {
        long[] binCountNeg = new long[binInfos.size() + 1];
        long[] binCountPos = new long[binInfos.size() + 1];
        for (int i = 0; i < binInfos.size(); i++) {
            CategoricalBinInfo binInfo = binInfos.get(i);
            binCountNeg[i] = binInfo.getNegativeCnt();
            binCountPos[i] = binInfo.getPositiveCnt();
        }
        binCountNeg[binCountNeg.length - 1] = missingBinInfo.getNegativeCnt();
        binCountPos[binCountPos.length - 1] = missingBinInfo.getPositiveCnt();
        ColumnStatsCalculator.ColumnMetrics columnMetrics =
                ColumnStatsCalculator.calculateColumnMetrics(binCountNeg, binCountPos);

        return columnMetrics.getIv();
    }

    private List<CategoricalBinInfo> mergeSmallBinInfos(List<CategoricalBinInfo> binInfos) {
        int i = 0;
        while (i < binInfos.size()) {
            CategoricalBinInfo binInfo = binInfos.get(i);
            if (this.minimumInstCnt > 0 && binInfo.getTotalInstCnt() < this.minimumInstCnt && binInfos.size() > 1) {
                if (i == 0) {
                    CategoricalBinInfo nextBinInfo = binInfos.get(i + 1);
                    binInfo.mergeRight(nextBinInfo);
                    binInfos.remove(i + 1);
                } else if (i == binInfos.size() - 1) {
                    CategoricalBinInfo prevBinInfo = binInfos.get(i - 1);
                    prevBinInfo.mergeRight(binInfo);
                    binInfos.remove(i);
                } else {
                    CategoricalBinInfo prevBinInfo = binInfos.get(i - 1);
                    CategoricalBinInfo nextBinInfo = binInfos.get(i + 1);
                    double prDeltaLeft = Math.abs(prevBinInfo.getPositiveRate() - binInfo.getPositiveRate());
                    double prDeltaRight = Math.abs(binInfo.getPositiveRate() - nextBinInfo.getPositiveRate());
                    if (prDeltaLeft < prDeltaRight) {
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

    private List<CategoricalBinInfo> genCategoricalBinInfos(ColumnConfig columnConfig) {
        List<CategoricalBinInfo> categoricalBinInfos = new ArrayList<CategoricalBinInfo>();
        for (int i = 0; i < columnConfig.getBinCategory().size(); i++) {
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

    private CategoricalBinInfo genMissingBinInfo() {
        // add missing binning
        CategoricalBinInfo binInfo = new CategoricalBinInfo();
        binInfo.setPositiveCnt(columnConfig.getBinCountPos().get(columnConfig.getBinCountPos().size() - 1));
        binInfo.setNegativeCnt(columnConfig.getBinCountNeg().get(columnConfig.getBinCountNeg().size() - 1));
        binInfo.setWeightPos(columnConfig.getBinWeightedPos().get(columnConfig.getBinWeightedPos().size() - 1));
        binInfo.setWeightNeg(columnConfig.getBinWeightedNeg().get(columnConfig.getBinWeightedNeg().size() - 1));
        return binInfo;
    }
}
