/**
 * Copyright [2012-2014] eBay Software Foundation
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

package ml.shifu.core.di.builtin;


import ml.shifu.core.container.ColumnBinningResult;
import ml.shifu.core.container.NumericalValueObject;
import ml.shifu.core.di.spi.ColumnNumBinningCalculator;
import ml.shifu.core.util.QuickSort;

import java.util.ArrayList;
import java.util.List;

public class EqualPositiveColumnNumBinningCalculator implements ColumnNumBinningCalculator {

    private final static Double EPS = 1e-6;

    public ColumnBinningResult calculate(List<NumericalValueObject> voList, int maxNumBins) {

        ColumnBinningResult columnBinningResult = new ColumnBinningResult();

        QuickSort.sort(voList, new NumericalValueObject.NumericalValueObjectComparator());


        int sumPositive = 0, voSize = voList.size();

        for (NumericalValueObject vo : voList) {
            sumPositive += vo.getIsPositive() ? 1 : 0;
        }


        int binSize = (int) Math.ceil((double) sumPositive / (double) maxNumBins);
        int currBin = 0;

        // double currBinSumScore = 0;

        Integer[] countNeg = new Integer[maxNumBins];
        Integer[] countPos = new Integer[maxNumBins];
        Double[] countWeightedNeg = new Double[maxNumBins];
        Double[] countWeightedPos = new Double[maxNumBins];


        countNeg[0] = 0;
        countPos[0] = 0;
        countWeightedNeg[0] = 0.0;
        countWeightedPos[0] = 0.0;

        List<Double> binBoundary = new ArrayList<Double>();
        // add first bin (from negative infinite)
        binBoundary.add(Double.NEGATIVE_INFINITY);

        NumericalValueObject vo;

        double prevData = voList.get(0).getValue();
        // For each Variable
        for (int i = 0; i < voSize; i++) {

            vo = voList.get(i);

            double currData = vo.getValue();
            // currBinSumScore += vo.getScore();

            // current bin is full
            if (countPos[currBin] >= binSize) { // vo.getTag() != 0 &&
                // still have some negative leftover
                if (currBin == maxNumBins - 1 && i != voList.size() - 1) {
                    continue;
                }
                // and data is different from the previous pair
                if (i == 0 || Math.abs(currData - prevData) > EPS) {

                    // MOVE to the new bin, if not the last vo
                    if (i == voList.size() - 1) {
                        break;
                    }
                    currBin++;
                    binBoundary.add(currData);

                    // AFTER move to the new bin
                    // currBinSumScore = 0;
                    countNeg[currBin] = 0;
                    countPos[currBin] = 0;
                    countWeightedNeg[currBin] = 0.0;
                    countWeightedPos[currBin] = 0.0;
                }
            }

            // increment the counter of the current bin
            if (vo.getIsPositive()) {
                countPos[currBin]++;
                countWeightedPos[currBin] += vo.getWeight();

            } else {
                countNeg[currBin]++;
                countWeightedNeg[currBin] += vo.getWeight();
            }
            prevData = currData;
        }

        // Finishing...
        // this.binBoundary.add(vo.getNumericalData());
        // this.binAvgScore.add(currBinSumScore / (countNeg[currBin] +
        // countPos[currBin]));

        int actualNumBins = currBin + 1;
        columnBinningResult.setLength(actualNumBins);

        List<Integer> binCountNeg = new ArrayList<Integer>();
        List<Integer> binCountPos = new ArrayList<Integer>();
        List<Double> binPosRate = new ArrayList<Double>();
        List<Double> binWeightedNeg = new ArrayList<Double>();
        List<Double> binWeightedPos = new ArrayList<Double>();

        for (int i = 0; i < actualNumBins; i++) {

            binCountNeg.add(countNeg[i]);
            binCountPos.add(countPos[i]);
            binPosRate.add((double) countPos[i] / (countPos[i] + countNeg[i]));
            binWeightedNeg.add(countWeightedNeg[i]);
            binWeightedPos.add(countWeightedPos[i]);
        }


        columnBinningResult.setBinBoundary(binBoundary);
        columnBinningResult.setBinCountNeg(binCountNeg);
        columnBinningResult.setBinCountPos(binCountPos);
        columnBinningResult.setBinPosRate(binPosRate);
        columnBinningResult.setBinWeightedNeg(binWeightedNeg);
        columnBinningResult.setBinWeightedPos(binWeightedPos);

        return columnBinningResult;
    }

}
