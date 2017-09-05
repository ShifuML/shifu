package ml.shifu.shifu.core.binning;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import ml.shifu.shifu.core.binning.obj.AbstractBinInfo;

/**
 * Created by zhanhu on 4/18/17.
 */
public class AutoDynamicBinning {

    private final static double EPS = 1e-6;

    private int expectedBinningNum;
    public AutoDynamicBinning(int expectedBinNum) {
        this.expectedBinningNum = expectedBinNum;
    }

    public List<AbstractBinInfo> merge(List<AbstractBinInfo> binInfos) {
        List<AbstractBinInfo> mergedBinInfos = new ArrayList<AbstractBinInfo>(binInfos);
        if (mergedBinInfos.size() > this.expectedBinningNum) {
            double totalInstCnt = getTotalInstCount(mergedBinInfos);
            mergedBinInfos = adjustBinInfos(mergedBinInfos, this.expectedBinningNum, totalInstCnt);
        }

        return mergedBinInfos;
    }

    private double getTotalInstCount(List<AbstractBinInfo> binInfos) {
        double total = 0.0;
        for (AbstractBinInfo binInfo : binInfos) {
            total += (binInfo.getNegativeCnt() + binInfo.getPositiveCnt());
        }
        return total;
    }

    private List<AbstractBinInfo> adjustBinInfos(List<AbstractBinInfo> mergedBinInfos,
                                                    int expectedBinningNum, double totalInstCnt) {
        while (mergedBinInfos.size() > expectedBinningNum) {
            int pos = getBestMergeNode(mergedBinInfos, totalInstCnt);
            if (pos > 0) {
                mergedBinInfos.get(pos - 1).mergeRight(mergedBinInfos.get(pos));
                mergedBinInfos.remove(pos);
            } else {
                break;
            }
        }

        return mergedBinInfos;
    }

    private int getBestMergeNode(List<AbstractBinInfo> mergedBinInfos, double totalInstCnt) {
        double entropy = calculateEntropy(mergedBinInfos, totalInstCnt);
        double entryReduction = Double.MAX_VALUE;
        int nodeIndexToMerge = 0;
        int pos = -1;

        AbstractBinInfo current = null;

        Iterator<AbstractBinInfo> iterator = mergedBinInfos.iterator();
        if (iterator.hasNext()) {
            pos = 0;
            current = iterator.next();
        }

        while (iterator.hasNext()) {
            pos++;
            AbstractBinInfo next = iterator.next();

            AbstractBinInfo temp = current.clone();
            temp.mergeRight(next);

            double entropyMerging = entropy
                    - getInfoValue(current, totalInstCnt)
                    - getInfoValue(next, totalInstCnt)
                    + getInfoValue(temp, totalInstCnt);

            double reduction = entropyMerging - entropy;
            if (reduction < entryReduction) {
                nodeIndexToMerge = pos;
                entryReduction = reduction;
            }

            current = next;
        }

        return nodeIndexToMerge;
    }

    private double calculateEntropy(List<AbstractBinInfo> mergedBinInfos, double totalInstCnt) {
        double entropy = 0.0;

        for (AbstractBinInfo binInfo : mergedBinInfos) {
            entropy += getInfoValue(binInfo, totalInstCnt);
        }

        return entropy;
    }

    private double getInfoValue(AbstractBinInfo cateBinInfo, double totalInstCnt) {
        double percent = cateBinInfo.getTotalInstCnt() / totalInstCnt;
        double positiveRate = (cateBinInfo.getPositiveCnt() + EPS) / cateBinInfo.getTotalInstCnt();
        double negativeRate = (cateBinInfo.getNegativeCnt() + EPS) / cateBinInfo.getTotalInstCnt();
        return -1 * percent * (positiveRate * log2(positiveRate) + negativeRate * log2(negativeRate));
    }

    private double log2(double ratio) {
        return Math.log(ratio) / Math.log(2.0d);
    }

}
