package ml.shifu.shifu.di.builtin;


import ml.shifu.shifu.di.spi.ColumnNumBinningCalculator;
import ml.shifu.shifu.container.NumericalValueObject;
import ml.shifu.shifu.container.obj.ColumnBinningResult;
import ml.shifu.shifu.util.QuickSort;

import java.util.ArrayList;
import java.util.List;

public class TotalPercentileColumnNumBinningCalculator implements ColumnNumBinningCalculator {

    private final static Double EPS = 1e-6;

    public ColumnBinningResult calculate(List<NumericalValueObject> voList, int maxNumBins) {

        ColumnBinningResult columnBinningResult = new ColumnBinningResult();


        List<Double> binBoundary = new ArrayList<Double>();
        List<Integer> binCountNeg = new ArrayList<Integer>();
        List<Integer> binCountPos = new ArrayList<Integer>();
        List<Double> binPosRate = new ArrayList<Double>();
        List<Double> binWeightedNeg = new ArrayList<Double>();
        List<Double> binWeightedPos = new ArrayList<Double>();

        QuickSort.sort(voList, new NumericalValueObject.NumericalValueObjectComparator());


        int bin = 0, cntTotal = 0, cntValidValue = 0, cntPos = 0, cntNeg = 0;
        double cntWeightedPos = 0.0, cntWeightedNeg = 0.0;

        boolean isFull = false;

        // Add initial bin left boundary: -infinity
        binBoundary.add(Double.NEGATIVE_INFINITY);

        cntValidValue = voList.size();


        int cntCumTotal = 0;
        for (NumericalValueObject vo : voList) {

            // Pre-processing: if bin is full, add binBoundary
            if (isFull) {
                binBoundary.add(vo.getValue());
                isFull = false;
            }

            // Core: push into bin
            if (vo.getIsPositive()) {
                cntPos++;
                cntWeightedPos += vo.getWeight();
                cntTotal += 1;
                cntCumTotal += 1;
            } else {
                cntNeg++;
                cntWeightedNeg += vo.getWeight();
                cntTotal += 1;
                cntCumTotal += 1;
            }


            // Post-processing: if bin is full, update related fields
            if ((double) cntCumTotal / (double) cntValidValue >= (double) (bin + 1) / (double) maxNumBins) {
                // Bin is Full
                isFull = true;
                binCountPos.add(cntPos);
                binCountNeg.add(cntNeg);

                binWeightedNeg.add(cntWeightedNeg);
                binWeightedPos.add(cntWeightedPos);
                binPosRate.add((double) binCountPos.get(bin) / (binCountPos.get(bin) + binCountNeg.get(bin)));

                bin++;
                cntTotal = 0;
                cntPos = 0;
                cntNeg = 0;
                cntWeightedNeg = 0.0;
                cntWeightedPos = 0.0;
            }

        }


        columnBinningResult.setLength(binBoundary.size());
        columnBinningResult.setBinBoundary(binBoundary);
        columnBinningResult.setBinCountNeg(binCountNeg);
        columnBinningResult.setBinCountPos(binCountPos);
        columnBinningResult.setBinPosRate(binPosRate);
        columnBinningResult.setBinWeightedNeg(binWeightedNeg);
        columnBinningResult.setBinWeightedPos(binWeightedPos);

        return columnBinningResult;
    }

}
