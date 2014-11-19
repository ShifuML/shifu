/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package ml.shifu.shifu.core.binning;

import java.util.ArrayList;
import java.util.List;

import ml.shifu.shifu.util.QuickSort;

import org.apache.commons.lang.StringUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * EqualPopulationBinning class
 * 
 * @Oct 22, 2014
 *
 */
public class EqualPopulationBinning extends AbstractBinning<Double> {
    
    private final static Logger log = LoggerFactory.getLogger(EqualPopulationBinning.class);
    
    /**
     * The default scale for generating histogram is for keep accuracy. 
     * General speaking, larger scale will guarantee better accuracy. But it will also cause worse efficiency  
     */
    public static final int HIST_SCALE = 100;
    
    /**
     * The maximum histogram unit count that could be hold
     */
    private int maxHistogramUnitCnt;
    
    /**
     * Current histogram
     */
    private List<HistogramUnit> histogram;
    
    /**
     * Empty constructor : it is just for bin merging
     */
    protected EqualPopulationBinning() {}
    
    /**
     * Construct @EqualPopulationBinning with expected bin number
     * @param binningNum
     */
    public EqualPopulationBinning(int binningNum) {
        this(binningNum, null);
    }
    
    /**
     * Construct @@EqualPopulationBinning with expected bin number and 
     *      values list that would be treated as missing value
     * @param binningNum
     * @param missingValList
     */
    public EqualPopulationBinning(int binningNum, List<String> missingValList) {
        super(binningNum);
        this.maxHistogramUnitCnt = super.expectedBinningNum * HIST_SCALE;
        this.histogram = new ArrayList<HistogramUnit>(this.maxHistogramUnitCnt + 1);
    }

    
    /* 
     * Add the value (in format of text) into histogram with frequency 1. 
     * First of all the input string will be trimmed and check whether it is missing value or not
     * If it is missing value, the missing value count will +1
     * After that, the input string will be parsed into double. If it is not a double, invalid value count will +1 
     * @see ml.shifu.shifu.core.binning.AbstractBinning#addData(java.lang.String)
     */
    @Override
    public void addData(String val) {
        String fval = StringUtils.trimToEmpty(val);
        if ( !isMissingVal(fval) ) {
            double dval = 0;
            
            try {
                dval = Double.parseDouble(fval);
            } catch (NumberFormatException e) {
                // not a number? just ignore
                super.incInvalidValCnt();
                return;
            }

            process(dval, 1);
        } else {
            super.incMissingValCnt();
        }
        
    }

    /**
     * Add a value into histogram with frequency 1.
     * @param val
     */
    public void addData(double val) {
        process(val, 1);
    }
    
    /**
     * Add a value into histogram with frequency.
     * @param val
     * @param frequency
     */
    public void addData(double val, int frequency) {
        process(val, frequency);
    }
    
    /* 
     * Generate data bin by expected bin number
     * @see ml.shifu.shifu.core.binning.AbstractBinning#getDataBin()
     */
    @Override
    public List<Double> getDataBin() {
        return getDataBin(super.expectedBinningNum);
    }

    /**
     * Get the median value in the histogram
     * @return
     */
    public Double getMedian() {
        List<Double> dataBinning = getDataBin(2);
        if ( dataBinning.size() > 1 ) {
            return dataBinning.get(1);
        } else {
            return null;
        }
    }
    
    /**
     * Generate data bin by expected bin number
     * @param toBinningNum
     * @return
     */
    private List<Double> getDataBin(int toBinningNum) {        
        List<Double> binBorders = new ArrayList<Double>();
        binBorders.add(Double.NEGATIVE_INFINITY);
        
        if ( histogram.size() <= toBinningNum ) {
            for ( int i = 0; i < histogram.size() - 1; i ++ ) {
                HistogramUnit chu = histogram.get(i);
                HistogramUnit nhu = histogram.get(i + 1);
                
                binBorders.add((chu.getHval() + nhu.getHval()) / 2);
            }
            
            return binBorders;
        }
        
        int totalCnt = 0;
        for ( HistogramUnit hu: this.histogram ) {
            totalCnt += hu.getHcnt();
        }
        
        int currentPos = 0;
        
        for(int j = 1; j < toBinningNum; j++) {
            double s = (double) (j * totalCnt) / toBinningNum;
            int pos = locateHistogram(s, currentPos);

            if ( pos < 0 || pos == currentPos ) {
                continue;
            } else {
                // System.out.println(s);
                HistogramUnit chu = histogram.get(pos);
                HistogramUnit nhu = histogram.get(pos + 1);

                double d = s - sum(histogram.get(pos).getHval());

                double a = nhu.getHcnt() - chu.getHcnt();
                double b = 2 * chu.getHcnt();
                double c = -2 * d;

                double z = 0.0;
                if(Double.compare(a, 0) == 0) {
                    z = -1 * c / b;
                } else {
                    z = (-1 * b + Math.sqrt(b * b - 4 * a * c)) / (2 * a);
                }

                double u = chu.getHval() + (nhu.getHval() - chu.getHval()) * z;
                binBorders.add(u);

                currentPos = pos;
            }
        }
            
        // binBorders.add(Double.POSITIVE_INFINITY);
        return binBorders;
    }
    
    /**
     * Locate histogram unit with just less than s, from some histogram unit
     * @param s
     * @param startPos
     * @return
     */
    private int locateHistogram(double s, int startPos) {
        for ( int i = startPos; i < histogram.size() - 1 ; i ++ ) {
            HistogramUnit chu = histogram.get(i);
            HistogramUnit nhu = histogram.get(i + 1);
            
            double sc = sum(chu.getHval());
            double sn = sum(nhu.getHval());
            
            // System.out.println("s=" + s + ",chu=" + chu.toString() + ",nhu=" + nhu.toString());
            if ( sc < s && s <= sn ) {
                return i;
            }
        }
        
        return -1;
    }

    /**
     * Sum the histogram's frequency whose value less than or equal some value 
     * @param hval
     * @return
     */
    private double sum(double hval) {
        int i = -1;
        for (int k = 0; k < histogram.size() - 1; k ++ ) {
            if ( histogram.get(k).getHval() <= hval && hval < histogram.get(k + 1).getHval() ) {
                i = k;
                break;
            }
        }
        
        if ( i >= 0 ) {
            HistogramUnit chu = histogram.get(i);
            HistogramUnit nhu = histogram.get(i + 1);
            double mb = chu.getHcnt() + (nhu.getHcnt() - nhu.getHcnt()) * (hval - chu.getHval()) / (nhu.getHval() - chu.getHval());
            double s = (chu.getHcnt() + mb) * (hval - chu.getHval()) / (nhu.getHval() - chu.getHval());
            s = s  / 2;
            
            for ( int j = 0 ; j < i; j ++ ) {
                HistogramUnit hu = histogram.get(j);
                s = s + hu.getHcnt();
            }
            
            return s + chu.getHcnt() / 2;
        } 
        
        return -1.0;
    }

    /**
     * Process the histogram with value and frequency
     * @param dval
     * @param frequency
     */
    private void process(double dval, int frequency) {
        HistogramUnit hu = getHistogramUnitIndex(dval);
        if ( hu != null ) {
            hu.setHcnt(hu.getHcnt() + frequency);
        } else {
            hu = new HistogramUnit(dval, frequency);
            histogram.add(hu);
            
            if ( histogram.size() > this.maxHistogramUnitCnt ) {
                QuickSort.sort(histogram);
                mergeHistogram();
            }
        }
    }
    
    /**
     * Merge the histogram to reduce histogram unit
     */
    private void mergeHistogram() {
        double minInterval = Double.MAX_VALUE;
        int pos = -1;
        for ( int i = 0; i < histogram.size() - 1; i ++ ) {
            double interval = histogram.get(i + 1).getHval() - histogram.get(i).getHval();
            if ( interval < minInterval ) {
                minInterval = interval;
                pos = i;
            }
        } 
        
        if ( pos >= 0  ) {
            HistogramUnit chu = histogram.get(pos);
            HistogramUnit nhu = histogram.get(pos + 1);
            
            chu.setHval((chu.getHval() * chu.getHcnt() + nhu.getHval() * nhu.getHcnt()) / (chu.getHcnt() + nhu.getHcnt()));
            chu.setHcnt(chu.getHcnt() + nhu.getHcnt());
            
            histogram.remove(pos + 1);
        }
    }

    /**
     * Get histogram unit by value, if not found return null
     * @param dval
     * @return
     */
    private HistogramUnit getHistogramUnitIndex(double dval) {
        for ( HistogramUnit hu : histogram ) {
            if ( Double.compare(hu.getHval(), dval) == 0 ) {
                return hu;
            }
        }
        return null;
    }


    /* (non-Javadoc)
     * @see ml.shifu.shifu.core.binning.AbstractBinning#mergeBin(ml.shifu.shifu.core.binning.AbstractBinning)
     */
    @Override
    public void mergeBin(AbstractBinning<?> another) {
        EqualPopulationBinning binning = (EqualPopulationBinning) another;
        
        super.mergeBin(another);
        
        this.histogram.addAll(binning.histogram);
        QuickSort.sort(histogram);
        
        int size = this.histogram.size();
        while ( size > this.maxHistogramUnitCnt ) {
            this.mergeHistogram();
            size = this.histogram.size();
        }
    }
    
    /**
     * convert @EqualIntervalBinning to String
     * @return
     */
    protected void stringToObj(String objValStr) {
        super.stringToObj(objValStr);

        if ( histogram == null ) {
            histogram = new ArrayList<HistogramUnit>();
        } else {
            histogram.clear();
        }
        
        String[] objStrArr = objValStr.split(Character.toString(FIELD_SEPARATOR), -1);
        maxHistogramUnitCnt = Integer.parseInt(objStrArr[4]);
        
        if ( objStrArr.length > 5 &&  StringUtils.isNotBlank(objStrArr[5]) ) {
            String[] histogramStrArr = objStrArr[5].split(Character.toString(SETLIST_SEPARATOR), -1);
            for ( String histogramStr : histogramStrArr ) {
                histogram.add(HistogramUnit.stringToObj(histogramStr));
            }
        } else {
            log.warn("Empty categorical bin - " + objValStr);
        }
    }
    
    /**
     * convert @EqualIntervalBinning to String
     * @return
     */
    public String objToString() {
        List<String> histogramStrList = new ArrayList<String>();
        for (HistogramUnit hu : this.histogram ) {
            histogramStrList.add(hu.objToString());
        }
        
        return super.objToString() 
                + Character.toString(FIELD_SEPARATOR) 
                + Integer.toString(maxHistogramUnitCnt)
                + Character.toString(FIELD_SEPARATOR) 
                + StringUtils.join(histogramStrList, SETLIST_SEPARATOR);
    }
    
    /**
     * 
     * HistogramUnit class is the unit for histogram
     * @Nov 19, 2014
     *
     */
    public static class HistogramUnit implements Comparable<HistogramUnit> {
        private double hval;
        private int hcnt;
        
        public HistogramUnit(double hval, int hcnt) {
            this.hval = hval;
            this.hcnt = hcnt;
        }

        public double getHval() {
            return hval;
        }

        public void setHval(double hval) {
            this.hval = hval;
        }

        public int getHcnt() {
            return hcnt;
        }

        public void setHcnt(int hcnt) {
            this.hcnt = hcnt;
        }

        /* (non-Javadoc)
         * @see java.lang.Comparable#compareTo(java.lang.Object)
         */
        @Override
        public int compareTo(HistogramUnit another) {
            return Double.compare(hval, another.getHval());
        }
        
        @Override
        public String toString() {
            return "[" + hval + ", " + hcnt + "]";
        }
        
        /**
         * convert @HistogramUnit object to String
         * @return
         */
        public String objToString() {
            return Double.toString(hval) + Character.toString(PAIR_SEPARATOR) + Integer.toString(hcnt);
        }
        
        /**
         * Constructor @HistogramUnit from String
         * @param histogramStr
         * @return
         */
        public static HistogramUnit stringToObj(String histogramStr) {
            String[] fields = StringUtils.split(histogramStr, PAIR_SEPARATOR);
            return new HistogramUnit(Double.parseDouble(fields[0]), Integer.parseInt(fields[1]));
        }

    }
    
}
