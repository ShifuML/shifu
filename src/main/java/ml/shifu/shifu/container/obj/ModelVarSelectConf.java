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
 * {@link ModelVarSelectConf} is 'varselect' part configuration in ModelConfig.json
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public class ModelVarSelectConf {

    /**
     * If enable force select and force remove
     */
    private Boolean forceEnable = Boolean.TRUE;

    /**
     * Force-selected column configuration file
     */
    private String forceSelectColumnNameFile;

    /**
     * Force-remove column configuration file
     */
    private String forceRemoveColumnNameFile;

    /**
     * If enable variable selection
     */
    private Boolean filterEnable = Boolean.TRUE;

    /**
     * How many columns will be selected. This number includes forceSelet columns.
     */
    private Integer filterNum = Constants.SHIFU_DEFAULT_VARSELECT_FILTER_NUM;

    /**
     * Filter by 'KS', 'IV', 'SE', 'SR'
     */
    private String filterBy = "KS";

    /**
     * Filter out ratio, filterNum has higher priority than {@link #filterOutRatio}, if {@link #filterNum} is less than
     * 0. Then filterOutRatio will be effective.
     */
    private Float filterOutRatio = Constants.SHIFU_DEFAULT_VARSELECT_FILTEROUT_RATIO;

    /**
     * For filterBy pareto mode, such epsilons are used in pareto sorting
     */
    private double[] epsilons;

    /**
     * If column missing rate is larger than this value, this column will be removed even it is set as 'FinalSelect'.
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
        if(epsilons != null) {
            other.setEpsilons(Arrays.copyOf(epsilons, epsilons.length));
        }
        other.setFilterBy(filterBy);
        other.setFilterEnable(filterEnable);
        other.setForceEnable(forceEnable);
        other.setFilterNum(filterNum);
        other.setForceRemoveColumnNameFile(forceRemoveColumnNameFile);
        other.setForceSelectColumnNameFile(forceSelectColumnNameFile);
        other.setMissingRateThreshold(missingRateThreshold);
        if(params != null) {
            other.setParams(new HashMap<String, Object>(params));
        }
        other.setFilerOutRatio(filterOutRatio);
        return other;
    }
}
