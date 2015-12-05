/**
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

import java.util.ArrayList;
import java.util.List;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.annotation.JsonDeserialize;
import com.google.common.collect.Lists;

/**
 * SourceData class
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public class RawSourceData implements Cloneable {

    @JsonDeserialize(using = SouceTypeDeserializer.class)
    public static enum SourceType {
        LOCAL, HDFS, S3
    }

    private SourceType source = SourceType.LOCAL;

    private String dataPath;

    private String dataDelimiter = "|";

    private String headerPath;

    private String headerDelimiter = "|";

    private String filterExpressions = "";

    private String weightColumnName = "";

    private String targetColumnName;

    private List<String> posTags;

    private List<String> negTags;

    private List<String> missingOrInvalidValues = Lists.asList("", new String[] { "?" });
    // private List<String> missingOrInvalidValues = Lists.asList("", new String[] { "*", "#", "?", "null", "none" });

    /**
     * Change it to true by default to compute distinct value for later variable selection.
     */
    private Boolean autoType = Boolean.TRUE;

    /**
     * To change it to 0 instead of 250 because 0-1 columns shouldn't be set to categorical. TODO, fix 0-1 bug
     */
    private Integer autoTypeThreshold = 0;

    /**
     * @return the autoTypeThreshold
     */
    @JsonIgnore
    public Integer getAutoTypeThreshold() {
        return autoTypeThreshold;
    }

    /**
     * @param autoTypeThreshold
     *            the autoTypeThreshold to set
     */
    @JsonProperty
    public void setAutoTypeThreshold(Integer autoTypeThreshold) {
        this.autoTypeThreshold = autoTypeThreshold;
    }

    public SourceType getSource() {
        return source;
    }

    public void setSource(SourceType source) {
        this.source = source;
    }

    public String getDataPath() {
        return dataPath;
    }

    public void setDataPath(String dataPath) {
        this.dataPath = dataPath;
    }

    public String getDataDelimiter() {
        return dataDelimiter;
    }

    public void setDataDelimiter(String dataDelimiter) {
        this.dataDelimiter = dataDelimiter;
    }

    public String getHeaderPath() {
        return headerPath;
    }

    public void setHeaderPath(String headerPath) {
        this.headerPath = headerPath;
    }

    public String getHeaderDelimiter() {
        return headerDelimiter;
    }

    public void setHeaderDelimiter(String headerDelimiter) {
        this.headerDelimiter = headerDelimiter;
    }

    public String getFilterExpressions() {
        return filterExpressions;
    }

    public void setFilterExpressions(String filterExpressions) {
        this.filterExpressions = filterExpressions;
    }

    public String getWeightColumnName() {
        return weightColumnName;
    }

    public void setWeightColumnName(String weightColumnName) {
        this.weightColumnName = weightColumnName;
    }

    public String getTargetColumnName() {
        return targetColumnName;
    }

    public void setTargetColumnName(String targetColumnName) {
        this.targetColumnName = targetColumnName;
    }

    public List<String> getPosTags() {
        return posTags;
    }

    public void setPosTags(List<String> posTags) {
        this.posTags = posTags;
    }

    public List<String> getNegTags() {
        return negTags;
    }

    public void setNegTags(List<String> negTags) {
        this.negTags = negTags;
    }

    @Override
    public RawSourceData clone() {
        RawSourceData copy = null;
        try {
            copy = (RawSourceData) super.clone();
        } catch (CloneNotSupportedException e) {
            // This should never happen
            throw new InternalError(e.toString());
        }

        copy.setSource(source);
        copy.setDataPath(dataPath);
        copy.setDataDelimiter(dataDelimiter);
        copy.setHeaderPath(headerPath);
        copy.setHeaderDelimiter(headerDelimiter);
        copy.setFilterExpressions(filterExpressions);

        copy.setTargetColumnName(targetColumnName);
        copy.setPosTags(new ArrayList<String>(posTags));
        copy.setNegTags(new ArrayList<String>(negTags));
        copy.setMissingOrInvalidValues(missingOrInvalidValues);
        return copy;
    }

    /**
     * @return the missingOrInvalidValues
     */
    public List<String> getMissingOrInvalidValues() {
        return missingOrInvalidValues;
    }

    /**
     * @param missingOrInvalidValues
     *            the missingOrInvalidValues to set
     */
    public void setMissingOrInvalidValues(List<String> missingOrInvalidValues) {
        this.missingOrInvalidValues = missingOrInvalidValues;
    }

    /**
     * @return the autoType
     */
    @JsonIgnore
    public Boolean getAutoType() {
        return autoType;
    }

    /**
     * @param autoType
     *            the autoType to set
     */
    @JsonProperty
    public void setAutoType(Boolean autoType) {
        this.autoType = autoType;
    }
}
