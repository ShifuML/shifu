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
import java.util.HashMap;
import java.util.Map;

import ml.shifu.shifu.core.binning.obj.LinkNode;

import org.apache.commons.lang.StringUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * EqualPopulationBinning class
 */
public class EqualPopulationBinning extends AbstractBinning<Double> {

    private final static Logger log = LoggerFactory.getLogger(EqualPopulationBinning.class);

    /**
     * The default scale for generating histogram is for keep accuracy.
     * General speaking, larger scale will guarantee better accuracy. But it will also cause worse efficiency
     * 
     * TODO here to make it computable with expected bin num, 100 * 10 = 1000, if set bin num to 100, this should not be
     * 100 because of bad performance.
     */
    public static final int HIST_SCALE = 100;

    /**
     * The threshold to define extra small bin.
     *  If (binCount / average binCount) &lt; EXTRA_BIN_COUNT_PERCENTAGE, it will be assume as extra small bin.
     */
    public static final double EXTRA_SMALL_BIN_PERCENTAGE = 0.003d;

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
     * Use to cache frequency sum of HistogramUnit, to improve getDataBin time performance, otherwise there maybe
     * timeout in reducer
     */
    private Map<LinkNode<HistogramUnit>, Double> sumCache = new HashMap<LinkNode<HistogramUnit>, Double>();
    /**
     * Empty constructor : it is just for bin merging
     */
    protected EqualPopulationBinning() {
    }

    /**
     * Construct @EqualPopulationBinning with expected bin number
     * 
     * @param binningNum
     *            the binningNum
     */
    public EqualPopulationBinning(int binningNum) {
        this(binningNum, null);
    }

    /**
     * Construct @EqualPopulationBinning with expected bin number and with histogram scale factor
     * 
     * @param binningNum
     *            the binningNum
     * @param histogramScale
     *            the histogram scale
     */
    public EqualPopulationBinning(int binningNum, int histogramScale) {
        this(binningNum, null);
        this.maxHistogramUnitCnt = super.expectedBinningNum * histogramScale;
        this.maxHistogramUnitCnt = (this.maxHistogramUnitCnt > 10000) ? 10000 : this.maxHistogramUnitCnt;
    }

    /**
     * Construct @@EqualPopulationBinning with expected bin number and
     * values list that would be treated as missing value
     * 
     * @param binningNum
     *            the binningNum
     * @param missingValList
     *            the missingValList
     */
    public EqualPopulationBinning(int binningNum, List<String> missingValList) {
        super(binningNum);
        this.maxHistogramUnitCnt = super.expectedBinningNum * HIST_SCALE;
        this.maxHistogramUnitCnt = (this.maxHistogramUnitCnt > 10000) ? 10000 : this.maxHistogramUnitCnt;
        this.currentHistogramUnitCnt = 0;
        this.header = null;
        this.tail = null;
    }

    /**
     * Add the value (in format of text) into histogram with frequency 1.
     * First of all the input string will be trimmed and check whether it is missing value or not
     * If it is missing value, the missing value count will +1
     * After that, the input string will be parsed into double. If it is not a double, invalid value count will +1
     * 
     * @see ml.shifu.shifu.core.binning.AbstractBinning#addData(java.lang.String)
     */
    @Override
    public void addData(String val) {
        String fval = StringUtils.trimToEmpty(val);
        if(!isMissingVal(fval)) {
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
     * Add the value (in format of text) into histogram with weight.
     * First of all the input string will be trimmed and check whether it is missing value or not
     * If it is missing value, the missing value count will +1
     * After that, the input string will be parsed into double. If it is not a double, invalid value count will +1
     * 
     * @param val
     *            , string type value
     * @param wVal
     *            , frequency or weight of this value
     */
    public void addData(String val, double wVal) {
        String fval = StringUtils.trimToEmpty(val);
        if(!isMissingVal(fval)) {
            double dval = 0;
            try {
                dval = Double.parseDouble(fval);
            } catch (NumberFormatException e) {
                // not a number? just ignore
                super.incInvalidValCnt();
                return;
            }
            process(dval, wVal);
        } else {
            super.incMissingValCnt();
        }
    }

    /**
     * Add a value into histogram with frequency 1.
     * 
     * @param val
     *            the value to be added
     */
    public void addData(double val) {
        process(val, 1);
    }

    /**
     * Add a value into histogram with frequency.
     * 
     * @param val
     *            the value to be added
     * @param frequency
     *            the weight
     */
    public void addData(double val, double frequency) {
        process(val, frequency);
    }

    /*
     * Generate data bin by expected bin number
     * 
     * @see ml.shifu.shifu.core.binning.AbstractBinning#getDataBin()
     */
    @Override
    public List<Double> getDataBin() {
        return getDataBin(super.expectedBinningNum);
    }

    /**
     * Get the median value in the histogram
     * 
     * @return median value
     */
    public Double getMedian() {
        List<Double> dataBinning = getDataBin(2);
        if(dataBinning.size() > 1) {
            return dataBinning.get(1);
        } else {
            return null;
        }
    }

    /**
     * Generate data bin by expected bin number
     * 
     * @param toBinningNum
     *            toBinningNum
     * @return list of data binning
     */
    private List<Double> getDataBin(int toBinningNum) {
        List<Double> binBorders = new ArrayList<Double>();
        binBorders.add(Double.NEGATIVE_INFINITY);

        double totalCnt = getTotalInHistogram();
        // merge extra small bins
        // extra small bin means : binCount less than 3% of average bin count
        // binCount < ( total * (1/toBinningNum) * (3/100))
        mergeExtraSmallBins(totalCnt, toBinningNum);

        if(this.currentHistogramUnitCnt <= toBinningNum) {
            // if the count of histogram unit is less than expected bin number
            // return each histogram unit as a bin. The boundary will be middle value
            // of every two histogram unit values
            convertHistogramUnitIntoBin(binBorders);
            return binBorders;
        }

        LinkNode<HistogramUnit> currStartPos = null;
        //To improve time performance
        sumCacheGen();
        for(int j = 1; j < toBinningNum; j++) {
            double s = (j * totalCnt) / toBinningNum; 
            LinkNode<HistogramUnit> pos = locateHistogram(s, currStartPos);
            if(pos == null || pos == currStartPos) {
                continue;
            } else {
                HistogramUnit chu = pos.data();
                HistogramUnit nhu = pos.next().data();

                //double d = s - sum(chu.getHval());
                double d = s - sumCache.get(pos);
                if(d < 0) {
                    double u = (chu.getHval() + nhu.getHval()) / 2;
                    binBorders.add(u);
                    currStartPos = pos;
                    continue;
                }

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

        return binBorders;
    }

    /**
     * merge extra small bin into its nearest node
     * @param totalCnt - total instance count
     * @param toBinningNum - the expected binning number
     */
    private void mergeExtraSmallBins(double totalCnt, int toBinningNum) {
        if ( this.header == null || this.header == this.tail ) {
            // if no node or just one node, do nothing
            return;
        }

        double minimumBinCnt = ((totalCnt / toBinningNum) * EXTRA_SMALL_BIN_PERCENTAGE);

        // there are two and more nodes
        LinkNode<HistogramUnit> tmp = this.header;
        while (tmp != null) {
            if (tmp.data().getHcnt() < minimumBinCnt) {
                HistogramUnit chu = tmp.data();

                if ( tmp == this.header ) {
                    HistogramUnit nhu = tmp.next().data();
                    // merge to next
                    nhu.setHcnt(chu.getHcnt() + nhu.getHcnt());
                    nhu.setHval((chu.getHval() * chu.getHcnt() + nhu.getHval() * nhu.getHcnt())
                            / (chu.getHcnt() + nhu.getHcnt()));
                    this.header = tmp.next();
                    this.header.setPrev(null);
                } else if (tmp == this.tail) {
                    HistogramUnit phu = tmp.prev().data();
                    phu.setHcnt(chu.getHcnt() + phu.getHcnt());
                    phu.setHval((chu.getHval() * chu.getHcnt() + phu.getHval() * phu.getHcnt())
                            / (chu.getHcnt() + phu.getHcnt()));
                    this.tail = tmp.prev();
                    this.tail.setNext(null);
                } else {
                    HistogramUnit phu = tmp.prev().data();
                    HistogramUnit nhu = tmp.next().data();

                    if (chu.getHval() - phu.getHval() < nhu.getHval() - chu.getHval()) {
                        // closer to previous, so to merge to previous
                        phu.setHcnt(chu.getHcnt() + phu.getHcnt());
                        phu.setHval((chu.getHval() * chu.getHcnt() + phu.getHval() * phu.getHcnt())
                                / (chu.getHcnt() + phu.getHcnt()));
                    } else {
                        // merge to next
                        nhu.setHcnt(chu.getHcnt() + nhu.getHcnt());
                        nhu.setHval((chu.getHval() * chu.getHcnt() + nhu.getHval() * nhu.getHcnt())
                                / (chu.getHcnt() + nhu.getHcnt()));
                    }

                    // remove tmp node from link
                    tmp.prev().setNext(tmp.next());
                    tmp.next().setPrev(tmp.prev());
                }
            }

            tmp = tmp.next();
        }
    }

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
     * 
     * @return total hosto value
     */
    private double getTotalInHistogram() {
        double total = 0;

        LinkNode<HistogramUnit> tmp = this.header;
        while(tmp != null) {
            total += tmp.data().getHcnt();
            tmp = tmp.next();
        }

        return total;
    }

    /**
     * Locate histogram unit with just less than s, from some histogram unit
     * 
     * @param s
     *            the s value
     * @param startPos
     *            start pos
     * @return next node
     */
    private LinkNode<HistogramUnit> locateHistogram(double s, LinkNode<HistogramUnit> startPos) {
        while(startPos != this.tail) {
            if(startPos == null) {
                startPos = this.header;
            }

            double sc = sumCache.get(startPos);
            double sn = sumCache.get(startPos.next());

            if(sc >= s || (sc < s && s <= sn)) {
                return startPos;
            }

            startPos = startPos.next();
        }

        return null;
    }

    /**
     * Generate sum the histogram's frequency at exact histogram pos
     * To improve time performance
     */
    private void sumCacheGen() {
       LinkNode<HistogramUnit> cur = this.header;
       double sum = 0;
       sumCache.clear();
       while(cur != null) {
           sumCache.put(cur, sum + cur.data().getHcnt() / 2d);
           sum += cur.data().getHcnt();
           cur = cur.next();
       }
    }

    /**
     * Sum the histogram's frequency whose value less than or equal some value
     * 
     * @param hval
     *            the h value
     * @return current sum
     */
    @SuppressWarnings("unused")
    private double sum(double hval) {
        LinkNode<HistogramUnit> posHistogramUnit = null;

        LinkNode<HistogramUnit> tmp = this.header;
        while(tmp != this.tail) {
            HistogramUnit chu = tmp.data();
            HistogramUnit nhu = tmp.next().data();

            if(chu.getHval() <= hval && hval < nhu.getHval()) {
                posHistogramUnit = tmp;
                break;
            }

            tmp = tmp.next();
        }

        if(posHistogramUnit != null) {
            HistogramUnit chu = posHistogramUnit.data();
            HistogramUnit nhu = posHistogramUnit.next().data();
            double mb = chu.getHcnt() + (nhu.getHcnt() - chu.getHcnt()) * (hval - chu.getHval())
                    / (nhu.getHval() - chu.getHval());
            double s = (chu.getHcnt() + mb) * (hval - chu.getHval()) / (nhu.getHval() - chu.getHval());
            s = s / 2;

            tmp = this.header;
            while(tmp != posHistogramUnit) {
                HistogramUnit hu = tmp.data();
                s = s + hu.getHcnt();
                tmp = tmp.next();
            }

            return s + chu.getHcnt() / 2d;
        } else if(tmp == this.tail) {
            double sum = 0.0;
            tmp = this.header;
            while(tmp != null) {
                sum += tmp.data().getHcnt();
                tmp = tmp.next();
            }
            return sum;
        }

        return -1.0;
    }

    /**
     * Process the histogram with value and frequency
     * 
     * @param dval
     *            the d value
     * @param frequency
     *            the weight
     */
    private void process(double dval, double frequency) {
        LinkNode<HistogramUnit> node = new LinkNode<HistogramUnit>(new HistogramUnit(dval, frequency));
        if(this.tail == null && this.maxHistogramUnitCnt > 1) {
            this.header = node;
            this.tail = node;
            this.currentHistogramUnitCnt = 1;
        } else {
            insertWithTrim(node);
        }
    }

    /**
     * Insert one @HistogramUnit node into the histogram.
     * Meanwhile it will try to keep the histogram as most @maxHistogramUnitCnt
     * So when inserting one node in, the method will try to find the place to insert as well as minimum interval
     * 
     * @param node
     *            current node
     */
    private void insertWithTrim(LinkNode<HistogramUnit> node) {
        LinkNode<HistogramUnit> insertOpsUnit = null;
        LinkNode<HistogramUnit> minIntervalOpsUnit = null;
        Double minInterval = Double.MAX_VALUE;

        LinkNode<HistogramUnit> tmp = this.tail;
        while(tmp != null) {
            if(insertOpsUnit == null) {
                int res = Double.compare(tmp.data().getHval(), node.data().getHval());
                if(res > 0) {
                    // do nothing
                } else if(res == 0) {
                    tmp.data().setHcnt(tmp.data().getHcnt() + node.data().getHcnt());
                    return;
                } else if(res < 0) {
                    // find the right insert position to insert
                    insertOpsUnit = tmp;

                    double interval = node.data().getHval() - tmp.data().getHval();
                    if(interval < minInterval) {
                        minInterval = interval;
                        minIntervalOpsUnit = tmp;
                    }

                    if(tmp.next() != null) {
                        interval = tmp.next().data().getHval() - node.data().getHval();
                        if(interval < minInterval) {
                            minInterval = interval;
                            minIntervalOpsUnit = node;
                        }
                    }
                }
            }

            if(tmp.next() != null) {
                LinkNode<HistogramUnit> next = tmp.next();
                double interval = next.data().getHval() - tmp.data().getHval();

                if(interval < minInterval) {
                    minInterval = interval;
                    minIntervalOpsUnit = tmp;
                }
            }

            tmp = tmp.prev();
        }

        // insert node into linked list
        if(insertOpsUnit == null) { // insert as the first node
            if(this.header != null) {
                this.header.setPrev(node);
            }
            node.setNext(this.header);
            this.header = node;
            if(this.tail == null) {
                this.tail = node;
            }
        } else if(insertOpsUnit == this.tail) { // insert as the last node
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
        if(this.currentHistogramUnitCnt == this.maxHistogramUnitCnt) {
            LinkNode<HistogramUnit> nextNode = minIntervalOpsUnit.next();
            HistogramUnit chu = minIntervalOpsUnit.data();
            HistogramUnit nhu = nextNode.data();
            nhu.setHval((chu.getHval() * chu.getHcnt() + nhu.getHval() * nhu.getHcnt())
                    / (chu.getHcnt() + nhu.getHcnt()));
            nhu.setHcnt(chu.getHcnt() + nhu.getHcnt());
            removeCurrentNode(minIntervalOpsUnit, nextNode);
        } else {
            this.currentHistogramUnitCnt++;
        }
    }

    private void removeCurrentNode(LinkNode<HistogramUnit> currNode, LinkNode<HistogramUnit> nextNode) {
        // remove current node
        if(currNode == this.header) {
            nextNode.setPrev(null);
            this.header = nextNode;
        } else {
            LinkNode<HistogramUnit> prev = currNode.prev();
            prev.setNext(nextNode);
            nextNode.setPrev(prev);
        }
    }

    @Override
    public void mergeBin(AbstractBinning<?> another) {
        EqualPopulationBinning binning = (EqualPopulationBinning) another;

        super.mergeBin(binning);

        LinkNode<HistogramUnit> tmp = binning.header;
        while(tmp != null) {
            this.insertWithTrim(new LinkNode<HistogramUnit>(tmp.data()));
            tmp = tmp.next();
        }
    }

    public void stringToObj(String objValStr) {
        super.stringToObj(objValStr);

        String[] objStrArr = objValStr.split(Character.toString(FIELD_SEPARATOR), -1);
        maxHistogramUnitCnt = Integer.parseInt(objStrArr[4]);

        if(objStrArr.length > 5 && StringUtils.isNotBlank(objStrArr[5])) {
            String[] histogramStrArr = objStrArr[5].split(Character.toString(SETLIST_SEPARATOR), -1);
            for(String histogramStr: histogramStrArr) {
                HistogramUnit hu = HistogramUnit.stringToObj(histogramStr);
                this.insertWithTrim(new LinkNode<HistogramUnit>(hu));
            }
        } else {
            log.warn("Empty categorical bin - " + objValStr);
        }
    }

    public String objToString() {
        List<String> histogramStrList = new ArrayList<String>();

        if(this.header != null) {
            LinkNode<HistogramUnit> tmp = this.header;
            while(tmp != null) {
                histogramStrList.add(tmp.data().objToString());
                tmp = tmp.next();
            }
        }

        return super.objToString() + Character.toString(FIELD_SEPARATOR) + Integer.toString(maxHistogramUnitCnt)
                + Character.toString(FIELD_SEPARATOR) + StringUtils.join(histogramStrList, SETLIST_SEPARATOR);
    }

    /**
     * HistogramUnit class is the unit for histogram
     */
    public static class HistogramUnit implements Comparable<HistogramUnit> {
        private double hval;
        private double hcnt;

        public HistogramUnit(double hval, double hcnt) {
            this.hval = hval;
            this.hcnt = hcnt;
        }

        public double getHval() {
            return hval;
        }

        public void setHval(double hval) {
            this.hval = hval;
        }

        public double getHcnt() {
            return hcnt;
        }

        public void setHcnt(double hcnt) {
            this.hcnt = hcnt;
        }

        /*
         * (non-Javadoc)
         * 
         * @see java.lang.Comparable#compareTo(java.lang.Object)
         */
        @Override
        public int compareTo(HistogramUnit another) {
            return Double.compare(hval, another.getHval());
        }

        /*
         * (non-Javadoc)
         * 
         * @see java.lang.Object#hashCode()
         */
        @Override
        public int hashCode() {
            final int prime = 31;
            int result = 1;
            long temp;
            temp = Double.doubleToLongBits(hval);
            result = prime * result + (int) (temp ^ (temp >>> 32));
            return result;
        }

        /*
         * (non-Javadoc)
         * 
         * @see java.lang.Object#equals(java.lang.Object)
         */
        @Override
        public boolean equals(Object obj) {
            if(this == obj)
                return true;
            if(obj == null)
                return false;
            if(!(obj instanceof HistogramUnit))
                return false;
            HistogramUnit other = (HistogramUnit) obj;
            return Double.compare(hval, other.hval) == 0;
        }

        @Override
        public String toString() {
            return "[" + hval + ", " + hcnt + "]";
        }

        public String objToString() {
            return Double.toString(hval) + Character.toString(PAIR_SEPARATOR) + Double.toString(hcnt);
        }

        public static HistogramUnit stringToObj(String histogramStr) {
            String[] fields = StringUtils.split(histogramStr, PAIR_SEPARATOR);
            return new HistogramUnit(Double.parseDouble(fields[0]), Double.parseDouble(fields[1]));
        }

    }

}
