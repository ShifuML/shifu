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
import java.util.Collections;
import java.util.List;

import org.apache.commons.lang.StringUtils;

/**
 * EqualPopulationBinning class
 * 
 * @author zhanhu
 * @Oct 22, 2014
 *
 */
public class EqualPopulationBinning extends AbstractBinning<Double> {
    
    public static final int HIST_SCALE = 100;
    
    private int maxHistogramUnitCnt;
    private List<HistogramUnit> histogram;
    
    /**
     * @param binningNum
     */
    public EqualPopulationBinning(int binningNum) {
        this(binningNum, null);
    }
    
    /**
     * @param binningNum
     */
    public EqualPopulationBinning(int binningNum, List<String> missingValList) {
        super(binningNum);
        this.maxHistogramUnitCnt = super.expectedBinningNum * HIST_SCALE;
        this.histogram = new ArrayList<HistogramUnit>();
    }

    /* (non-Javadoc)
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

    public void addData(double val) {
        process(val, 1);
    }
    
    public void addData(double val, int frequency) {
        process(val, frequency);
    }
    
    /* (non-Javadoc)
     * @see ml.shifu.shifu.core.binning.AbstractBinning#getDataBin()
     */
    @Override
    public List<Double> getDataBin() {
        return getDataBin(super.expectedBinningNum);
    }

    public Double getMedian() {
        List<Double> dataBinning = getDataBin(2);
        if ( dataBinning.size() > 1 ) {
            return dataBinning.get(1);
        } else {
            return null;
        }
    }
    
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
     * @param s
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
     * @param dval
     */
    private void process(double dval, int frequency) {
        HistogramUnit hu = getHistogramUnitIndex(dval);
        if ( hu != null ) {
            hu.setHcnt(hu.getHcnt() + frequency);
        } else {
            hu = new HistogramUnit(dval, frequency);
            histogram.add(hu);
            
            Collections.sort(histogram);
            if ( histogram.size() > this.maxHistogramUnitCnt ) {
                mergeHistogram();
            }
        }
    }
    
    /**
     * 
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
        
        if ( pos > 0  ) {
            HistogramUnit chu = histogram.get(pos);
            HistogramUnit nhu = histogram.get(pos + 1);
            
            chu.setHval((chu.getHval() * chu.getHcnt() + nhu.getHval() * nhu.getHcnt()) / (chu.getHcnt() + nhu.getHcnt()));
            chu.setHcnt(chu.getHcnt() + nhu.getHcnt());
            
            histogram.remove(pos + 1);
        }
    }

    /**
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
    }
    
}
