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
package ml.shifu.shifu.container.obj;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

import ml.shifu.shifu.util.Constants;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/**
 * ModelVarSelectConf class
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public class ModelVarSelectConf {

    // for force select or force remove
    private Boolean forceEnable = Boolean.TRUE;
    private String forceSelectColumnNameFile;
    private String forceRemoveColumnNameFile;

    // settings for variable selection
    private Boolean filterEnable = Boolean.TRUE;
    private Integer filterNum = Integer.valueOf(Constants.SHIFU_DEFAULT_VARSELECT_FILTER_NUM);
    private String filterBy = "KS";
    private Float filterOutRatio = Float.valueOf(Constants.SHIFU_DEFAULT_VARSELECT_FILTEROUT_RATIO);
    /**
     * For filterBy pareto mode, such epsilons are used in pareto sorting
     */
    private double[] epsilons;

    /**
     * If column missing rate is lower than this column, no matter what, whis column will be removed.
     */
    private Float missingRateThreshold = 0.98f;

    private Map<String, Object> params;

    public Boolean getForceEnable() {
        return forceEnable;
    }

    public void setForceEnable(Boolean forceEnable) {
        this.forceEnable = forceEnable;
    }

    public String getForceSelectColumnNameFile() {
        return forceSelectColumnNameFile;
    }

    public void setForceSelectColumnNameFile(String forceSelectColumnConf) {
        this.forceSelectColumnNameFile = forceSelectColumnConf;
    }

    public String getForceRemoveColumnNameFile() {
        return forceRemoveColumnNameFile;
    }

    public void setForceRemoveColumnNameFile(String forceRemoveColumnConf) {
        this.forceRemoveColumnNameFile = forceRemoveColumnConf;
    }

    public Boolean getFilterEnable() {
        return filterEnable;
    }

    public void setFilterEnable(Boolean filterEnable) {
        this.filterEnable = filterEnable;
    }

    public Integer getFilterNum() {
        return filterNum;
    }

    public void setFilterNum(Integer filterNum) {
        this.filterNum = filterNum;
    }

    public String getFilterBy() {
        return filterBy;
    }

    public void setFilterBy(String filterBy) {
        this.filterBy = filterBy;
    }

//    public Boolean getWrapperEnabled() {
//        return wrapperEnabled;
//    }
//
//    public void setWrapperEnabled(Boolean wrapperEnabled) {
//        this.wrapperEnabled = wrapperEnabled;
//    }
//
//    public Integer getWrapperNum() {
//        return wrapperNum;
//    }
//
//    public void setWrapperNum(Integer wrapperNum) {
//        this.wrapperNum = wrapperNum;
//    }
//
//    public String getWrapperBy() {
//        return wrapperBy;
//    }
//
//    public void setWrapperBy(String wrapperBy) {
//        this.wrapperBy = wrapperBy;
//    }

//    /**
//     * @return the wrapperRatio
//     */
//    public Float getWrapperRatio() {
//        return wrapperRatio;
//    }
//
//    /**
//     * @param wrapperRatio
//     *            the wrapperRatio to set
//     */
//    public void setWrapperRatio(Float wrapperRatio) {
//        this.wrapperRatio = wrapperRatio;
//    }
    
    /**
     * @return the filterOutRatio
     */
    public Float getFilterOutRatio() {
        return filterOutRatio;
    }

    /**
     * @param filterOutRatio
     *            the filterOutRatio to set
     */
    public void setFilerOutRatio(Float filterOutRatio) {
        this.filterOutRatio = filterOutRatio;
    }

    public Map<String, Object> getParams() {
        return params;
    }

    public void setParams(Map<String, Object> params) {
        this.params = params;
    }

    /**
     * @return the missingRateThreshold
     */
    public Float getMissingRateThreshold() {
        return missingRateThreshold;
    }

    /**
     * @param missingRateThreshold
     *            the missingRateThreshold to set
     */
    public void setMissingRateThreshold(Float missingRateThreshold) {
        this.missingRateThreshold = missingRateThreshold;
    }

    /**
     * @return the epsilons
     */
    @JsonIgnore
    public double[] getEpsilons() {
        return epsilons;
    }

    /**
     * @param epsilons
     *            the epsilons to set
     */
    @JsonProperty
    public void setEpsilons(double[] epsilons) {
        this.epsilons = epsilons;
    }

    @Override
    public ModelVarSelectConf clone() {
        ModelVarSelectConf other = new ModelVarSelectConf();
        if ( epsilons != null ) {
            other.setEpsilons(Arrays.copyOf(epsilons, epsilons.length));
        }
        other.setFilterBy(filterBy);
        other.setFilterEnable(filterEnable);
        other.setForceEnable(forceEnable);
        other.setForceRemoveColumnNameFile(forceRemoveColumnNameFile);
        other.setForceSelectColumnNameFile(forceSelectColumnNameFile);
        other.setMissingRateThreshold(missingRateThreshold);
        if ( params != null ) {
            other.setParams(new HashMap<String, Object>(params));
        }
        other.setFilerOutRatio(filterOutRatio);
        return other;
    }
}
