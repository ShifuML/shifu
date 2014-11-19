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

import ml.shifu.shifu.core.binning.obj.LinkNode;

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
     * Current histogram unit count in histogram
     */
    private int currentHistogramUnitCnt;
    
    /**
     * The header and tail of histogram
     */
    private LinkNode<HistogramUnit> header, tail;
    
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
        
        this.currentHistogramUnitCnt = 0;
        this.header = null;
        this.tail = null;
        //this.histogram = new ArrayList<HistogramUnit>(this.maxHistogramUnitCnt + 1);
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

        if ( this.currentHistogramUnitCnt <= toBinningNum ) {
            // if the count of histogram unit is less than expected bin number
            // return each histogram unit as a bin. The boundary will be middle value 
            // of every two histogram unit values
            convertHistogramUnitIntoBin(binBorders);
            return binBorders;
        }
        
        int totalCnt = getTotalInHistogram();
        LinkNode<HistogramUnit> currStartPos = this.header;
        for(int j = 1; j < toBinningNum; j++) {
            double s = (double) (j * totalCnt) / toBinningNum;
            LinkNode<HistogramUnit> pos = locateHistogram(s, currStartPos);
            if ( pos == null || pos == currStartPos  ) {
                continue;
            } else {
                HistogramUnit chu = pos.data();
                HistogramUnit nhu = pos.next().data();

                double d = s - sum(chu.getHval());
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

                currStartPos = pos;
            }
        }
            
        // binBorders.add(Double.POSITIVE_INFINITY);
        return binBorders;
    }
    
    /**
     * @param binBorders
     */
    private void convertHistogramUnitIntoBin(List<Double> binBorders) {
        LinkNode<HistogramUnit> tmp = this.header;
        while(tmp != this.tail) {
            HistogramUnit chu = tmp.data();
            HistogramUnit nhu = tmp.next().data();
            binBorders.add((chu.getHval() + nhu.getHval()) / 2);

            tmp = tmp.next();
        }
    }

    /**
     * Get the total value count in histogram
     * @return
     */
    private int getTotalInHistogram() {
        int total = 0;
        
        LinkNode<HistogramUnit> tmp = this.header;
        while ( tmp != null ) {
            total += tmp.data().getHcnt();
            tmp = tmp.next();
        }
        
        return total;
    }

    /**
     * Locate histogram unit with just less than s, from some histogram unit
     * @param s
     * @param startPos
     * @return
     */
    private LinkNode<HistogramUnit> locateHistogram(double s, LinkNode<HistogramUnit> startPos) {
        while ( startPos != this.tail ) {
            HistogramUnit chu = startPos.data();
            HistogramUnit nhu = startPos.next().data();
            
            double sc = sum(chu.getHval());
            double sn = sum(nhu.getHval());
            
            if ( sc < s && s <= sn ) {
                return startPos;
            }
            
            startPos = startPos.next();
        }
        
        return null;
    }

    /**
     * Sum the histogram's frequency whose value less than or equal some value 
     * @param hval
     * @return
     */
    private double sum(double hval) {
        LinkNode<HistogramUnit> posHistogramUnit = null;
        
        LinkNode<HistogramUnit> tmp = this.header;
        while ( tmp != this.tail ) {
            HistogramUnit chu = tmp.data();
            HistogramUnit nhu = tmp.next().data();
            
            if ( chu.getHval() <= hval && hval < nhu.getHval() ) {
                posHistogramUnit = tmp;
                break;
            }
            
            tmp = tmp.next();
        }
        
        if (  posHistogramUnit != null ) {
            HistogramUnit chu = posHistogramUnit.data();
            HistogramUnit nhu = posHistogramUnit.next().data();
            double mb = chu.getHcnt() + (nhu.getHcnt() - nhu.getHcnt()) * (hval - chu.getHval()) / (nhu.getHval() - chu.getHval());
            double s = (chu.getHcnt() + mb) * (hval - chu.getHval()) / (nhu.getHval() - chu.getHval());
            s = s  / 2;
            
            tmp = this.header;
            while ( tmp != posHistogramUnit ) {
                HistogramUnit hu = tmp.data();
                s = s + hu.getHcnt();
                tmp = tmp.next();
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
        LinkNode<HistogramUnit> node = new LinkNode<HistogramUnit>(new HistogramUnit(dval, frequency));
        if ( this.tail == null && this.maxHistogramUnitCnt > 1) {
            this.header = node;
            this.tail = node;
            this.currentHistogramUnitCnt = 1;
        } else {
            insertWithTrim(node);
        }
    }
    
    private void insertWithTrim(LinkNode<HistogramUnit> node) {
        LinkNode<HistogramUnit> insertOpsUnit = null;
        LinkNode<HistogramUnit> minIntervalOpsUnit = null;
        Double minInterval = Double.MAX_VALUE;
        
        LinkNode<HistogramUnit> tmp = this.tail;
        while ( tmp != null ) {
            if ( insertOpsUnit == null ) {
                // find the right insert position to insert
                if ( tmp.data().getHval() <= node.data().getHval() ) {
                    insertOpsUnit = tmp;
                }
                
                // find the minimum interval 
                double interval = Math.abs(node.data().getHval() - tmp.data().getHval());
                if ( interval < minInterval ) {
                    if ( Double.compare(interval, 0.0) == 0 ) {
                        tmp.data().setHcnt(tmp.data().getHcnt() + node.data().getHcnt());
                        return;
                    }
                    
                    minInterval = interval;
                    minIntervalOpsUnit = (node.data().getHval() - tmp.data().getHval() >= 0) ? tmp : node;
                }
                
            }
            
            
            if ( tmp.next()  != null ) {
                LinkNode<HistogramUnit> next = tmp.next();
                double interval = next.data().getHval() - tmp.data().getHval();
                
                if ( interval < minInterval ) {
                    minInterval = interval;
                    minIntervalOpsUnit = tmp;
                }
            }
            
            tmp = tmp.prev();
        }
        
        // insert node into linked list
        if ( insertOpsUnit == null ) { // insert as the first node
            this.header.setPrev(node);
            node.setNext(this.header);
            this.header = node;
        } else if ( insertOpsUnit == this.tail ) { // insert as the last node
            node.setPrev(insertOpsUnit);
            insertOpsUnit.setNext(node);
            this.tail = node;
        } else { // some intermediate node
            node.setNext(insertOpsUnit.next());
            node.setPrev(insertOpsUnit);
            insertOpsUnit.next().setPrev(node);
            insertOpsUnit.setNext(node);
        }
        
        // merge info into next node
        LinkNode<HistogramUnit> nextNode = minIntervalOpsUnit.next();
        HistogramUnit chu = minIntervalOpsUnit.data();
        HistogramUnit nhu = nextNode.data();

        if(this.currentHistogramUnitCnt == this.maxHistogramUnitCnt) {
            nhu.setHval((chu.getHval() * chu.getHcnt() + nhu.getHval() * nhu.getHcnt()) / (chu.getHcnt() + nhu.getHcnt()));
            nhu.setHcnt(chu.getHcnt() + nhu.getHcnt());
            removeCurrentNode(minIntervalOpsUnit, nextNode);
        } else {
            this.currentHistogramUnitCnt++;
        }
    }
    
    /**
     * @param minIntervalOpsUnit
     */
    private void removeCurrentNode(LinkNode<HistogramUnit> currNode, LinkNode<HistogramUnit> nextNode) {
        // remove current node
        if ( currNode == this.header ) {
            nextNode.setPrev(null);
            this.header = nextNode;
        } else {
            LinkNode<HistogramUnit> prev = currNode.prev();
            prev.setNext(nextNode);
            nextNode.setPrev(prev);
        }
    }

    /* (non-Javadoc)
     * @see ml.shifu.shifu.core.binning.AbstractBinning#mergeBin(ml.shifu.shifu.core.binning.AbstractBinning)
     */
    @Override
    public void mergeBin(AbstractBinning<?> another) {
        EqualPopulationBinning binning = (EqualPopulationBinning) another;
        
        super.mergeBin(binning);
        
        LinkNode<HistogramUnit> tmp = binning.header;
        while ( tmp != null ) {
            this.insertWithTrim(new LinkNode<HistogramUnit>(tmp.data()));
            tmp = tmp.next();
        }
    }
    
    /**
     * convert @EqualIntervalBinning to String
     * @return
     */
    protected void stringToObj(String objValStr) {
        super.stringToObj(objValStr);

        String[] objStrArr = objValStr.split(Character.toString(FIELD_SEPARATOR), -1);
        maxHistogramUnitCnt = Integer.parseInt(objStrArr[4]);
        
        if ( objStrArr.length > 5 &&  StringUtils.isNotBlank(objStrArr[5]) ) {
            String[] histogramStrArr = objStrArr[5].split(Character.toString(SETLIST_SEPARATOR), -1);
            for ( String histogramStr : histogramStrArr ) {
                HistogramUnit hu = HistogramUnit.stringToObj(histogramStr);
                this.insertWithTrim(new LinkNode<HistogramUnit>(hu));
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
        
        if ( this.header != null ) {
            LinkNode<HistogramUnit> tmp = this.header;
            while ( tmp != null ) {
                histogramStrList.add(tmp.data().objToString());
                tmp = tmp.next();
            }
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
