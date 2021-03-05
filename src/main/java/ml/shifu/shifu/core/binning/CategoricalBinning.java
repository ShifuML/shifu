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

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import ml.shifu.shifu.util.Constants;
import org.apache.commons.lang.StringUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.clearspring.analytics.stream.cardinality.CardinalityMergeException;
import com.clearspring.analytics.stream.cardinality.HyperLogLogPlus;

import ml.shifu.shifu.util.Base64Utils;
import ml.shifu.shifu.util.CommonUtils;

/**
 * CategoricalBinning class
 */
public class CategoricalBinning extends AbstractBinning<String> {

    private final static Logger log = LoggerFactory.getLogger(CategoricalBinning.class);

    private boolean isValid = true;
    private Set<String> categoricalVals;
    private int hashSeed = 0;

    private HyperLogLogPlus hyper = new HyperLogLogPlus(8);;

    /**
     * Empty constructor : it is just for bin merging
     */
    public CategoricalBinning() {
    }

    /*
     * Constructor with expected bin number.
     * For categorical variable, the binningNum won't be used
     */
    public CategoricalBinning(int binningNum, int maxCategorySize) {
        this(binningNum, null, maxCategorySize);
    }

    /*
     * Constructor with expected bin number and missing value list
     * For categorical variable, the binningNum won't be used
     */
    public CategoricalBinning(int binningNum, List<String> missingValList, int maxCategorySize) {
        super(binningNum, missingValList, maxCategorySize);
        this.categoricalVals = new HashSet<String>();
    }

    public CategoricalBinning(int binningNum, List<String> missingValList, int maxCategorySize, int hashSeed) {
        this(binningNum, missingValList, maxCategorySize);
        this.hashSeed = hashSeed;
    }

    /*
     * Constructor with expected bin number and missing value list
     * For categorical variable, the binningNum won't be used
     */
    public CategoricalBinning(int binningNum, List<String> missingValList, int maxCategorySize, int hashSeed,
            boolean isPrint) {
        this(binningNum, missingValList, maxCategorySize);
        this.hashSeed = hashSeed;
    }

    /*
     * (non-Javadoc)
     * Add the string into value set
     * First of all the input string will be trimmed and check whether it is missing value or not
     * If it is missing value, the missing value count will +1
     * 
     * @see ml.shifu.shifu.core.binning.AbstractBinning#addData(java.lang.Object)
     */
    @Override
    public void addData(String val) {
        String fval = (val == null ? "" : val);
        log.debug("hash feature test");

        this.hyper.offer(fval);

        if(isMissingVal(fval)) {
            super.incMissingValCnt();
        } else if(fval.length() > Constants.MAX_CATEGORICAL_VAL_LENGTH) {
            super.incInvalidValCnt();
        } else {
            if(isValid && this.hashSeed <= 0) {
                categoricalVals.add(fval);
            } else if(isValid && this.hashSeed > 0) {
                categoricalVals.add(Integer.toString(fval.hashCode() % this.hashSeed));
            }

            if(categoricalVals.size() > maxCategorySize) {
                isValid = false;
                categoricalVals.clear();
            }
        }
    }

    /*
     * (non-Javadoc)
     * 
     * @see ml.shifu.shifu.core.binning.AbstractBinning#getDataBin()
     */
    @Override
    public List<String> getDataBin() {
        List<String> binningVals = new ArrayList<String>();
        binningVals.addAll(categoricalVals);
        return binningVals;
    }

    /*
     * (non-Javadoc)
     * 
     * @see ml.shifu.shifu.core.binning.AbstractBinning#mergeBin(ml.shifu.shifu.core.binning.AbstractBinning)
     */
    @Override
    public void mergeBin(AbstractBinning<?> another) {
        CategoricalBinning binning = (CategoricalBinning) another;
        super.mergeBin(another);

        // merge hyper stats
        try {
            this.hyper.merge(binning.hyper);
        } catch (CardinalityMergeException e) {
            throw new RuntimeException(e);
        }

        this.isValid = (this.isValid && binning.isValid);
        if(this.isValid) {
            for(String cate: binning.categoricalVals) {
                // check if over max category size, skip to copy to avoid OOM
                if(this.categoricalVals.size() <= this.maxCategorySize) {
                    this.categoricalVals.add(cate);
                } else {
                    log.warn("Categorical variables binning merge over max category size ({}).", this.maxCategorySize);
                    this.categoricalVals.clear();
                    this.isValid = false;
                    break;
                }
            }
        } else {
            this.categoricalVals.clear();
        }
    }

    /**
     * convert @CategoricalBinning to String
     */
    public void stringToObj(String objValStr) {
        super.stringToObj(objValStr);

        if(categoricalVals == null) {
            categoricalVals = new HashSet<String>();
        } else {
            categoricalVals.clear();
        }

        String[] objStrArr = CommonUtils.split(objValStr, Character.toString(FIELD_SEPARATOR));
        this.isValid = Boolean.valueOf(objStrArr[4]);
        if(objStrArr.length > 5 && StringUtils.isNotBlank(objStrArr[5])) {
            String[] elements = CommonUtils.split(objStrArr[5], Character.toString(SETLIST_SEPARATOR));
            for(String element: elements) {
                categoricalVals.add(element);
            }
        } else {
            log.warn("Empty categorical bin - " + objValStr);
        }

        if(objStrArr.length > 6 && StringUtils.isNotBlank(objStrArr[6])) {
            try {
                this.hyper = HyperLogLogPlus.Builder.build(Base64Utils.base64DecodeToBytes((objStrArr[6])));
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }
    }

    /*
     * convert @CategoricalBinning to String
     */
    public String objToString() {
        try {
            return super.objToString() + Character.toString(FIELD_SEPARATOR) + Boolean.toString(isValid)
                    + Character.toString(FIELD_SEPARATOR) + StringUtils.join(categoricalVals, SETLIST_SEPARATOR)
                    + Character.toString(FIELD_SEPARATOR) + Base64Utils.base64EncodeFromBytes(this.hyper.getBytes());
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Categorical variable in first 2 stats job to get cardinality, if too large, hash trick can be enabled.
     * 
     * @return cardinality of categories
     */
    public long cardinality() {
        return this.hyper.cardinality();
    }

}
