/*
 * Copyright [2012-2014] PayPal Software Foundation
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
package ml.shifu.shifu.core;

import ml.shifu.shifu.container.ValueObject;
import ml.shifu.shifu.container.ValueObject.ValueObjectComparator;
import ml.shifu.shifu.container.obj.ModelStatsConf.BinningMethod;
import ml.shifu.shifu.util.QuickSort;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.Serializable;
import java.util.*;

/**
 * Binning, it helps to put data input bins
 */
public class Binning {

    /**
     * logger
     */
    private static Logger log = LoggerFactory.getLogger(Binning.class);

    /**
     * epsilon, it helps to judge equation between 2 float number or prevent divide 0 exception
     */
    private final double EPS = 1e-5;

    /**
     * auto type threshold, it help to judge if it's categorical or numerical
     */
    private Integer autoTypeThreshold = 5;

    /**
     * merger flag
     */
    private Boolean mergeEnabled = true;

    /**
     * Data type, Numerical/Categorical/Auto
     */
    public static enum BinningDataType {
        Numerical, Categorical, Auto
    }

    /**
     * positive tags
     */
    private List<String> posTags;

    /**
     * negative tags
     */
    private List<String> negTags;

    /**
     * data type, flag
     */
    private BinningDataType dataType;

    /**
     * Input Data
     */
    private List<ValueObject> voList;

    /**
     * value object size
     */
    private Integer voSize;

    /**
     * Negative Count(Good txn)
     */
    private List<Integer> binCountNeg;

    /**
     * Positive Count(Bad txn)
     */
    private List<Integer> binCountPos;

    private List<Double> binWeightedNeg;

    public List<Double> getBinWeightedNeg() {
        return binWeightedNeg;
    }

    public void setBinWeightedNeg(List<Double> binWeightedNeg) {
        this.binWeightedNeg = binWeightedNeg;
    }

    public List<Double> getBinWeightedPos() {
        return binWeightedPos;
    }

    public void setBinWeightedPos(List<Double> binWeightedPos) {
        this.binWeightedPos = binWeightedPos;
    }

    private List<Double> binWeightedPos;

    /**
     * Bin Boundary for Numerical Variables
     */
    private List<Double> binBoundary = null;

    /**
     * Bin Category for Categorical Variables
     */
    private List<String> binCategory = null;

    /**
     * Bin Average Score
     */
    private List<Integer> binAvgScore = null;

    /**
     * Bin positive rate
     */
    private List<Double> binPosCaseRate = null;

    /**
     * default numBins
     */
    private int expectNumBins = 10;

    /**
     * actual bins, if the data is not larger than expect
     */
    private int actualNumBins = -1;

    /**
     * default binningMethod
     */
    private BinningMethod binningMethod = BinningMethod.EqualPositive;

    /**
     * Constructor
     * 
     * @param posTags
     *            The positive tags list, identify the positive tag in voList
     * @param negTags
     *            The negative tags list, identify the negative tag in volist
     * @param type
     *            The data type
     * @param voList
     *            Value object list
     */
    public Binning(List<String> posTags, List<String> negTags, BinningDataType type, List<ValueObject> voList) {
        this.posTags = posTags;
        this.negTags = negTags;

        this.dataType = type;
        this.voList = voList;
        this.voSize = voList.size();

        binCountNeg = new ArrayList<Integer>();
        binCountPos = new ArrayList<Integer>();
        binBoundary = new ArrayList<Double>();
        binCategory = new ArrayList<String>();
        binAvgScore = new ArrayList<Integer>();
        binPosCaseRate = new ArrayList<Double>();

        this.binWeightedNeg = new ArrayList<Double>();
        this.binWeightedPos = new ArrayList<Double>();

        // voList is sorted!
        // Collections.sort(this.voList, new
        // ValueObject.VariableObjectComparator());
    }

    /**
     * setter, the max bins
     * 
     * @param numBins
     *            the numBins
     */
    public void setMaxNumOfBins(int numBins) {
        this.expectNumBins = numBins;
    }

    /**
     * setter, the binning method
     * 
     * @param binningMethod
     *            the binningMethod
     */
    public void setBinningMethod(BinningMethod binningMethod) {
        this.binningMethod = binningMethod;
    }

    /**
     * Set the the max size of auto type threshold
     * 
     * @param autoTypeThreshold
     *            the autoTypeThreshold
     */
    public void setAutoTypeThreshold(Integer autoTypeThreshold) {
        this.autoTypeThreshold = autoTypeThreshold;
    }

    /**
     * Numerical: Raw to Value Conversion happens before Binning
     * <p>
     * Categorical: Raw to Value Conversion happens after Binning
     * <p>
     * Start binning method
     */
    public void doBinning() {
        // Set DataType by the number of different keys.
        // If it is lower than the threshold, will be treated as Categorical;
        // otherwise as Numerical;
        if(dataType.equals(BinningDataType.Auto)) {
            int cntRaw = 0;
            int cntValue = 0;

            for(ValueObject vo: voList) {
                if(vo.getValue() != null) {
                    cntValue++;
                } else {
                    cntRaw++;
                }
            }

            Set<Object> keySet = new HashSet<Object>();
            for(ValueObject vo: voList) {
                if(vo.getValue() != null) {
                    keySet.add(vo.getValue());
                } else {
                    keySet.add(vo.getRaw());
                }
                if(keySet.size() > this.autoTypeThreshold) {
                    break;
                }
            }

            // log.info("BinningDataType: Auto");
            // log.info("    # Numerical:   " + cntValue);
            // log.info("    # Categorical: " + cntRaw);
            // log.info("    # Different Key: " + keySet.size());

            if(cntRaw > 0 || keySet.size() <= this.autoTypeThreshold) {
                this.dataType = BinningDataType.Categorical;
                if(cntValue > 0) {
                    for(ValueObject vo: voList) {
                        if(vo.getRaw() == null) {
                            vo.setRaw(vo.getValue().toString());
                        }
                    }
                }
                // log.info("FinalType: Categorical");
            } else {
                this.dataType = BinningDataType.Numerical;
                // log.info("FinalType: Numerical");
            }
        }

        if(dataType.equals(BinningDataType.Categorical)) {
            doCategoricalBinning();
        } else if(dataType.equals(BinningDataType.Numerical)) {
            doNumericalBinning();
        }
    }

    /**
     * Start numerical binnning
     * </p>
     * BinBoundary: left, inclusive
     */
    private void doNumericalBinning() {
        // use our in-place quick order
        QuickSort.sort(voList, new ValueObjectComparator(BinningDataType.Numerical));
        // Collections.sort(voList, new ValueObjectComparator(BinningDataType.Numerical));

        if(BinningMethod.EqualPositive.equals(binningMethod)) {
            doEqualPositiveBinning();
        } else if(BinningMethod.EqualTotal.equals(binningMethod)) {
            doEqualTotalBinning();
        }
    }

    /**
     * equal bad binning
     */
    private void doEqualPositiveBinning() {
        int sumBad = 0;
        for(int i = 0; i < voSize; i++) {
            sumBad += (posTags.contains(voList.get(i).getTag()) ? 1 : 0);
        }
        int binSize = (int) Math.ceil((double) sumBad / (double) expectNumBins);
        int currBin = 0;

        // double currBinSumScore = 0;

        Integer[] countNeg = new Integer[expectNumBins];
        Integer[] countPos = new Integer[expectNumBins];
        Double[] countWeightedNeg = new Double[expectNumBins];
        Double[] countWeightedPos = new Double[expectNumBins];

        countNeg[0] = 0;
        countPos[0] = 0;
        countWeightedNeg[0] = 0.0;
        countWeightedPos[0] = 0.0;

        // add first bin (from negative infinite)
        this.binBoundary.add(Double.NEGATIVE_INFINITY);

        ValueObject vo = null;

        double prevData = voList.get(0).getValue();
        // For each Variable
        for(int i = 0; i < voSize; i++) {

            vo = voList.get(i);
            double currData = vo.getValue();
            // currBinSumScore += vo.getScore();

            // current bin is full
            if(countPos[currBin] >= binSize) { // vo.getTag() != 0 &&
                // still have some negative leftover
                if(currBin == expectNumBins - 1 && i != voList.size() - 1) {
                    continue;
                }
                // and data is different from the previous pair
                if(i == 0 || (mergeEnabled == true && Math.abs(currData - prevData) > EPS) || mergeEnabled == false) {
                    // BEFORE move to the new bin
                    // this.binAvgScore.add(currBinSumScore / (countNeg[currBin]
                    // + countPos[currBin]));

                    // MOVE to the new bin, if not the last vo
                    if(i == voList.size() - 1) {
                        break;
                    }
                    currBin++;
                    this.binBoundary.add(currData);

                    // AFTER move to the new bin
                    // currBinSumScore = 0;
                    countNeg[currBin] = 0;
                    countPos[currBin] = 0;
                    countWeightedNeg[currBin] = 0.0;
                    countWeightedPos[currBin] = 0.0;
                }
            }

            // increment the counter of the current bin
            if(negTags.contains(voList.get(i).getTag())) {
                countNeg[currBin]++;
                countWeightedNeg[currBin] += vo.getWeight();
            } else {
                countPos[currBin]++;
                countWeightedPos[currBin] += vo.getWeight();
            }
            prevData = currData;
        }

        // Finishing...
        // this.binBoundary.add(vo.getNumericalData());
        // this.binAvgScore.add(currBinSumScore / (countNeg[currBin] +
        // countPos[currBin]));
        this.actualNumBins = currBin + 1;

        for(int i = 0; i < this.actualNumBins; i++) {
            binCountNeg.add(countNeg[i]);
            binCountPos.add(countPos[i]);
            binAvgScore.add(0);
            binPosCaseRate.add((double) countPos[i] / (countPos[i] + countNeg[i]));
            this.binWeightedNeg.add(countWeightedNeg[i]);
            this.binWeightedPos.add(countWeightedPos[i]);
        }
    }

    /**
     * equal total binning
     */
    private void doEqualTotalBinning() {

        @SuppressWarnings("unused")
        int cntTotal = 0;
        int bin = 0;
        int cntValidValue = 0;
        int cntPos = 0;
        int cntNeg = 0;
        double cntWeightedPos = 0.0, cntWeightedNeg = 0.0;

        boolean isFull = false;

        // Add initial bin left boundary: -infinity
        binBoundary.add(Double.NEGATIVE_INFINITY);

        for(ValueObject vo: voList) {
            if(posTags.contains(vo.getTag()) || negTags.contains(vo.getTag())) {
                cntValidValue += 1;
            }
        }

        int cntCumTotal = 0;
        for(ValueObject vo: voList) {

            // Pre-processing: if bin is full, add binBoundary
            if(isFull) {
                binBoundary.add(vo.getValue());
                isFull = false;
            }

            // Core: push into bin or skip
            if(posTags.contains(vo.getTag())) {
                cntPos++;
                cntWeightedPos += vo.getWeight();
                cntTotal += 1;
                cntCumTotal += 1;
            } else if(negTags.contains(vo.getTag())) {
                cntNeg++;
                cntWeightedNeg += vo.getWeight();
                cntTotal += 1;
                cntCumTotal += 1;
            } else {
                // skip
            }

            // Post-processing: if bin is full, update related fields
            if((double) cntCumTotal / (double) cntValidValue >= (double) (bin + 1) / (double) expectNumBins) {
                // Bin is Full
                isFull = true;
                binCountPos.add(cntPos);
                binCountNeg.add(cntNeg);
                binAvgScore.add(0);
                binWeightedNeg.add(cntWeightedNeg);
                binWeightedPos.add(cntWeightedPos);
                binPosCaseRate.add((double) binCountPos.get(bin) / (binCountPos.get(bin) + binCountNeg.get(bin)));

                bin++;
                cntTotal = 0;
                cntPos = 0;
                cntNeg = 0;
                cntWeightedNeg = 0.0;
                cntWeightedPos = 0.0;
            }

        }
    }

    /**
     * if map contain key, the value increase 1
     * 
     * @param map
     * @param key
     */
    private void incMapCnt(Map<String, Integer> map, String key) {
        int cnt = map.containsKey(key) ? map.get(key) : 0;
        map.put(key, cnt + 1);
    }

    private void incMapWithValue(Map<String, Double> map, String key, Double value) {
        double num = map.containsKey(key) ? map.get(key) : 0.0;
        map.put(key, num + value);
    }

    /**
     * categorical binning
     */
    private void doCategoricalBinning() {
        // In JDK1.6, the sort action will copy the whole array. That's memory consuming
        // For categorical variable, it's not necessary to sort the data
        // Collections.sort(voList, new ValueObjectComparator(BinningDataType.Categorical));

        Map<String, Integer> categoryHistNeg = new HashMap<String, Integer>();
        Map<String, Integer> categoryHistPos = new HashMap<String, Integer>();
        Map<String, Double> categoryWeightedNeg = new HashMap<String, Double>();
        Map<String, Double> categoryWeightedPos = new HashMap<String, Double>();

        Set<String> categorySet = new HashSet<String>();
        // Map<String, Double> categoryScoreMap = new HashMap<String, Double>();

        for(int i = 0; i < voSize; i++) {
            String category = voList.get(i).getRaw();
            categorySet.add(category);
            // Double score = categoryScoreMap.containsKey(category) ? categoryScoreMap.get(category) : 0;
            // categoryScoreMap.put(category, score + voList.get(i).getScore());

            if(negTags.contains(voList.get(i).getTag())) {
                incMapCnt(categoryHistNeg, category);
                incMapWithValue(categoryWeightedNeg, category, voList.get(i).getWeight());
            } else {
                incMapCnt(categoryHistPos, category);
                incMapWithValue(categoryWeightedPos, category, voList.get(i).getWeight());
            }
        }

        Map<String, Double> categoryFraudRateMap = new HashMap<String, Double>();

        for(String key: categorySet) {
            double cnt0 = categoryHistNeg.containsKey(key) ? categoryHistNeg.get(key) : 0;
            double cnt1 = categoryHistPos.containsKey(key) ? categoryHistPos.get(key) : 0;
            double rate;
            if(Double.compare(cnt0 + cnt1, 0) == 0) {
                rate = 0;
            } else {
                rate = cnt1 / (cnt0 + cnt1);
            }
            categoryFraudRateMap.put(key, rate);
        }

        // Sort map
        MapComparator cmp = new MapComparator(categoryFraudRateMap);
        Map<String, Double> sortedCategoryFraudRateMap = new TreeMap<String, Double>(cmp);
        sortedCategoryFraudRateMap.putAll(categoryFraudRateMap);

        for(Map.Entry<String, Double> entry: sortedCategoryFraudRateMap.entrySet()) {
            String key = entry.getKey();
            Integer countNeg = categoryHistNeg.containsKey(key) ? categoryHistNeg.get(key) : 0;
            binCountNeg.add(countNeg);
            Integer countPos = categoryHistPos.containsKey(key) ? categoryHistPos.get(key) : 0;
            binCountPos.add(countPos);

            Double weightedNeg = categoryWeightedNeg.containsKey(key) ? categoryWeightedNeg.get(key) : 0.0;
            this.binWeightedNeg.add(weightedNeg);

            Double weightedPos = categoryWeightedPos.containsKey(key) ? categoryWeightedPos.get(key) : 0.0;
            this.binWeightedPos.add(weightedPos);

            // use zero, the average score is calculate in post-process
            binAvgScore.add(0);
            binCategory.add(key);
            binPosCaseRate.add(entry.getValue());
        }

        this.actualNumBins = binCategory.size();

        for(ValueObject vo: voList) {
            String key = vo.getRaw();

            // TODO: Delete this after categorical data is correctly labeled.
            if(binCategory.indexOf(key) == -1) {
                vo.setValue(0.0);
            } else {
                // --- end deletion ---
                vo.setValue(binPosCaseRate.get(binCategory.indexOf(key)));
            }
        }
    }

    /**
     * getter the number of bins
     * 
     * @return the actual number of bins
     */
    public int getNumBins() {
        return this.actualNumBins;
    }

    /**
     * get the bin boundary, from negative infinite to the max of value object
     * 
     * @return the bins boundary
     */
    public List<Double> getBinBoundary() {
        return this.binBoundary;
    }

    /**
     * get the bin category
     * 
     * @return bin category
     */
    public List<String> getBinCategory() {
        return this.binCategory;
    }

    /**
     * @return get the negative bin count
     */
    public List<Integer> getBinCountNeg() {
        return binCountNeg;
    }

    /**
     * @return get the positive bin count
     */
    public List<Integer> getBinCountPos() {
        return binCountPos;
    }

    /**
     * @return get the average score list
     */
    public List<Integer> getBinAvgScore() {
        return binAvgScore;
    }

    /**
     * @return get the positive rate for kv
     */
    public List<Double> getBinPosCaseRate() {
        return binPosCaseRate;
    }

    /**
     * @return get the volist
     */
    public List<ValueObject> getUpdatedVoList() {
        for(ValueObject vo: voList) {
            if(vo.getValue() == null) {
                log.error("Not Updated yet.");
                return null;
            }
        }
        return voList;
    }

    /**
     * comparator for map
     */
    private static class MapComparator implements Comparator<String>, Serializable {

        private static final long serialVersionUID = 2178035954425107063L;

        Map<String, Double> base;

        public MapComparator(Map<String, Double> base) {
            this.base = base;
        }

        public int compare(String a, String b) {
            return base.get(a).compareTo(base.get(b));
        }
    }

    /**
     * @return get the data type
     */
    public BinningDataType getUpdatedDataType() {
        return dataType;
    }

    /*
     * set the merge flag
     */
    public void setMergeEnabled(Boolean mergeEnabled) {
        this.mergeEnabled = mergeEnabled;
    }

}
