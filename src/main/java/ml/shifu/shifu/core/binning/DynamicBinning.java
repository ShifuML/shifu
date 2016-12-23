/*
 * Copyright [2013-2015] PayPal Software Foundation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package ml.shifu.shifu.core.binning;

import ml.shifu.shifu.core.binning.obj.NumBinInfo;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;

/**
 * Created by zhanhu on 7/6/16.
 */
public class DynamicBinning extends AbstractBinning<Double> {

    private final static double EPS = 1e-6;

    private List<NumBinInfo> binInfos;

    public DynamicBinning(List<NumBinInfo> binInfos, int expectedBinNum) {
        super(expectedBinNum);
        this.binInfos = binInfos;
    }

    @Override
    public void addData(String val) {
        // Do nothing
    }

    @Override
    public List<Double> getDataBin() {
        List<NumBinInfo> mergedBinInfos = combineEmptyBin(binInfos);

        if ( mergedBinInfos.size() > super.expectedBinningNum ) {
            double totalInstCnt = getTotalInstCount(mergedBinInfos);
            mergedBinInfos = adjustBinInfos(mergedBinInfos, super.expectedBinningNum, totalInstCnt);
        }

        List<Double> retBins = new ArrayList<Double>();
        for ( NumBinInfo numBinInfo : mergedBinInfos) {
            retBins.add(numBinInfo.getLeftThreshold());
        }

        return retBins;
    }

    private double getTotalInstCount(List<NumBinInfo> mergedBinInfos) {
        double total = 0.0;
        for ( NumBinInfo binInfo : mergedBinInfos ) {
            total += binInfo.getTotalInstCnt();
        }
        return total;
    }

    private List<NumBinInfo> adjustBinInfos(List<NumBinInfo> mergedBinInfos, int expectedBinningNum, double totalInstCnt) {
        while ( mergedBinInfos.size() > expectedBinningNum ) {
            int pos = getBestMergeNode(mergedBinInfos, totalInstCnt);
            if ( pos > 0 ) {
                mergedBinInfos.get(pos - 1).mergeRight(mergedBinInfos.get(pos));
                mergedBinInfos.remove(pos);
            } else {
                break;
            }
        }

        return mergedBinInfos;
    }

    private int getBestMergeNode(List<NumBinInfo> mergedBinInfos, double totalInstCnt) {
        double entropy = calculateEntropy(mergedBinInfos, totalInstCnt);
        double entryReduction = Double.MAX_VALUE;
        int nodeIndexToMerge = 0;
        int pos = -1;

        NumBinInfo current = null;

        Iterator<NumBinInfo> iterator = mergedBinInfos.iterator();
        if ( iterator.hasNext() ) {
            pos = 0;
            current = iterator.next();
        }


        while (iterator.hasNext()) {
            pos ++;
            NumBinInfo next = iterator.next();

            NumBinInfo temp = current.clone();
            temp.mergeRight(next);

            double entropyMerging = entropy
                    - getInfoValue(current, totalInstCnt)
                    - getInfoValue(next, totalInstCnt)
                    + getInfoValue(temp, totalInstCnt);

            double reduction = entropyMerging - entropy;
            if ( reduction < entryReduction ) {
                nodeIndexToMerge = pos;
                entryReduction = reduction;
            }

            current = next;
        }

        return  nodeIndexToMerge;
    }

    private double calculateEntropy(List<NumBinInfo> mergedBinInfos, double totalInstCnt) {
        double entropy = 0.0;

        for ( NumBinInfo binInfo : mergedBinInfos ) {
            entropy += getInfoValue(binInfo, totalInstCnt);
        }

        return entropy;
    }

    private double getInfoValue(NumBinInfo numBinInfo, double totalInstCnt) {
        double percent = numBinInfo.getTotalInstCnt() / totalInstCnt;
        double positiveRate = (numBinInfo.getPositiveInstCnt() + EPS) / numBinInfo.getTotalInstCnt();
        double negativeRate = (numBinInfo.getTotalInstCnt() - numBinInfo.getPositiveInstCnt() + EPS)
                / numBinInfo.getTotalInstCnt();

        return -1 * percent * (positiveRate * log2(positiveRate) + negativeRate * log2(negativeRate));
    }

    private double log2(double ratio) {
        return Math.log(ratio) / Math.log(2.0d);
    }

    private List<NumBinInfo> combineEmptyBin(List<NumBinInfo> binInfos) {
        int[] mergeIndicator = new int[binInfos.size()];

        for ( int i = 0; i < binInfos.size(); i ++ ) {
            NumBinInfo binInfo = binInfos.get(i);
            if ( binInfo.getTotalInstCnt() > 0 ) {
                mergeIndicator[i] = i;
            } else {
                int pos = findNearestNonEmptyBinInfo(binInfos, i);
                if (pos >= 0) {
                    mergeIndicator[i] = pos;
                } else {
                    // usually it won't happen here
                    mergeIndicator[i] = i;
                }
            }
        }

        List<NumBinInfo> mergedBinInfos = new LinkedList<NumBinInfo>();
        for ( int i = 0; i < mergeIndicator.length; i ++ ) {
            if ( mergeIndicator[i] == i ) {
                NumBinInfo binInfo = binInfos.get(i).clone();

                // merge left bin info
                int j = i - 1;
                while ( j >= 0 && mergeIndicator[j] == i ) {
                    binInfo.setLeftThreshold(binInfos.get(j).getLeftThreshold());
                    j --;
                }

                j = i + 1;
                while ( j < mergeIndicator.length && mergeIndicator[j] == i ) {
                    binInfo.setRightThreshold(binInfos.get(j).getRightThreshold());
                    j ++;
                }

                mergedBinInfos.add(binInfo);
            }
        }

        return mergedBinInfos;
    }

    private int findNearestNonEmptyBinInfo(List<NumBinInfo> binInfos, int i) {
        int lpos = -1;
        int rpos = -1;
        double dl = Double.MAX_VALUE;
        double dr = Double.MAX_VALUE;

        int j = i - 1;
        while ( j >= 0 ) {
            if ( binInfos.get(j).getTotalInstCnt() > 0 ) {
                lpos = j;
                dl = binInfos.get(i).getLeftThreshold() - binInfos.get(j).getRightThreshold();
                break;
            }

            j--;
        }

        j = i + 1;
        while ( j < binInfos.size() ) {
            if ( binInfos.get(j).getTotalInstCnt() > 0 ) {
                rpos = j;
                dr = binInfos.get(j).getLeftThreshold() - binInfos.get(i).getRightThreshold();
                break;
            }

            j++;
        }

        return ((dl < dr) ? lpos : rpos);
    }
}
