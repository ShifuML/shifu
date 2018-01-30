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
import com.fasterxml.jackson.databind.annotation.JsonDeserialize;

import ml.shifu.shifu.util.Constants;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/**
 * {@link ModelVarSelectConf} is 'varselect' part configuration in ModelConfig.json
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public class ModelVarSelectConf {

    @JsonDeserialize(using = PostCorrelationMetricDeserializer.class)
    public static enum PostCorrelationMetric {
        IV, KS, SE
    }

    /**
     * If enable force select and force remove
     */
    private Boolean forceEnable = Boolean.TRUE;

    /**
     * Specify all candidate column names for variable selection
     * If the value is null or the file is empty, all variables should be candidates
     */
    private String candidateColumnNameFile;

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
     * Sometimes user may don't want to auto filter variables. Use this option to give user the freedom
     * The default value is true. Some separate command will be provided do auto filter.
     * "shifu varsel -autofilter"
     */
    private Boolean autoFilterEnable = Boolean.TRUE;

    /**
     * If column missing rate is larger than this value, this column will be removed even it is set as 'FinalSelect'.
     */
    private Float missingRateThreshold = 0.98f;

    /**
     * If two features correlation value is larger than {@link #correlationThreshold}, one with larger IV value will be
     * selected. Set it to default 1 or not computed correlation value in norm step, means such threshold has no effect.
     */
    private Float correlationThreshold = 1f;

    /**
     * If IV value of feature is less than this threshold, such feature will be dropped and not selected.
     */
    private Float minIvThreshold = 0f;

    /**
     * If KS value of feature is less than this threshold, such feature will be dropped and not selected.
     */
    private Float minKsThreshold = 0f;

    /**
     * Enable variable selection for correlation value, this is the metric to keep the better feature. If column i and j
     * has higher correlation value than {@link #correlationThreshold}. According to {@link #postCorrelationMetric} to
     * choose a better one to keep, the other one would be dropped. For example, if set it KS, drop column with smaller
     * KS value.
     */
    private PostCorrelationMetric postCorrelationMetric = PostCorrelationMetric.IV;

    private Map<String, Object> params;

    public Boolean getForceEnable() {
        return forceEnable;
    }

    public void setForceEnable(Boolean forceEnable) {
        this.forceEnable = forceEnable;
    }

    public String getCandidateColumnNameFile() {
        return candidateColumnNameFile;
    }

    public void setCandidateColumnNameFile(String candidateColumnNameFile) {
        this.candidateColumnNameFile = candidateColumnNameFile;
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

    public Boolean getAutoFilterEnable() {
        return autoFilterEnable;
    }

    public void setAutoFilterEnable(Boolean autoFilterEnable) {
        this.autoFilterEnable = autoFilterEnable;
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

    /**
     * @return the correlationThreshold
     */
    public Float getCorrelationThreshold() {
        return correlationThreshold;
    }

    /**
     * @param correlationThreshold
     *            the correlationThreshold to set
     */
    public void setCorrelationThreshold(Float correlationThreshold) {
        this.correlationThreshold = correlationThreshold;
    }

    /**
     * @return the postCorrelationMetric
     */
    public PostCorrelationMetric getPostCorrelationMetric() {
        return postCorrelationMetric;
    }

    /**
     * @param postCorrelationMetric
     *            the postCorrelationMetric to set
     */
    public void setPostCorrelationMetric(PostCorrelationMetric postCorrelationMetric) {
        this.postCorrelationMetric = postCorrelationMetric;
    }

    /**
     * @return the minIvThreshold
     */
    public Float getMinIvThreshold() {
        return minIvThreshold;
    }

    /**
     * @return the minKsThreshold
     */
    public Float getMinKsThreshold() {
        return minKsThreshold;
    }

    /**
     * @param minIvThreshold
     *            the minIvThreshold to set
     */
    public void setMinIvThreshold(Float minIvThreshold) {
        this.minIvThreshold = minIvThreshold;
    }

    /**
     * @param minKsThreshold
     *            the minKsThreshold to set
     */
    public void setMinKsThreshold(Float minKsThreshold) {
        this.minKsThreshold = minKsThreshold;
    }

    @Override
    public ModelVarSelectConf clone() {
        ModelVarSelectConf other = new ModelVarSelectConf();
        if(epsilons != null) {
            other.setEpsilons(Arrays.copyOf(epsilons, epsilons.length));
        }

        // parameters for force variables selection
        other.setForceEnable(forceEnable);
        other.setForceRemoveColumnNameFile(forceRemoveColumnNameFile);
        other.setForceSelectColumnNameFile(forceSelectColumnNameFile);
        // parameters for variable filter selection
        other.setFilterEnable(filterEnable);
        other.setFilterBy(filterBy);
        other.setFilterNum(filterNum);
        other.setFilerOutRatio(filterOutRatio);
        // parameters for auto variable filter selection
        other.setAutoFilterEnable(autoFilterEnable);
        other.setMissingRateThreshold(missingRateThreshold);
        other.setMinIvThreshold(minIvThreshold);
        other.setMinKsThreshold(minKsThreshold);
        other.setPostCorrelationMetric(postCorrelationMetric);
        other.setCorrelationThreshold(correlationThreshold);

        if(params != null) {
            other.setParams(new HashMap<String, Object>(params));
        }

        return other;
    }
}
