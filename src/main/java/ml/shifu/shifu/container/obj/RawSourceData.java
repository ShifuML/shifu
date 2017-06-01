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

import java.util.ArrayList;
import java.util.List;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.annotation.JsonDeserialize;
import com.google.common.collect.Lists;
import ml.shifu.shifu.util.CommonUtils;

/**
 * SourceData part for ModelConfig.json
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public class RawSourceData implements Cloneable {

    /**
     * If data is from local or hdfs, S3 is not supported so far.
     * 
     * @author Zhang David (pengzhang@paypal.com)
     */
    @JsonDeserialize(using = SouceTypeDeserializer.class)
    public static enum SourceType {
        LOCAL, HDFS, S3
    }

    /**
     * If source from local or hdfs
     */
    private SourceType source = SourceType.LOCAL;

    /**
     * Data path, from local or hdfs. Folder and file are all supported. Recursive folder is also supported. CSV format
     * file is supported and if csv, header no need to set.
     */
    private String dataPath;

    /**
     * Validation data path which is used in train step for validation data. Such data should have the same schema like
     * {@link #dataPath}. If {@link #validationDataPath} is not empty, specified validation data is enabled and all
     * other sampling parameters have no effect. If empty (by default), such feature is not enabled.
     */
    private String validationDataPath;

    /**
     * How to split data and validation data.
     */
    private String dataDelimiter = "|";

    /**
     * Header path for schema, if null, first line of data will be checked and read to get schema. That's why csv format
     * file works well in Shifu.
     */
    private String headerPath;

    /**
     * How to split header content.
     * 
     */
    private String headerDelimiter = "|";

    /**
     * Filter expression on data path and validation data path, this is helpful to filter some data not in original
     * data. Example like 'columna > 10'
     */
    private String filterExpressions = "";

    /**
     * Weight column, should be one of columns
     */
    private String weightColumnName = "";

    /**
     * Target column, should be one of columns
     */
    private String targetColumnName;

    /**
     * Positive tag list: Example like ["0", "1"];
     */
    private List<String> posTags;
    
    /**
     * Negative tag list: Example like ["2", "3"]
     */
    private List<String> negTags;

    /**
     * Missing or invalid values.
     */
    private List<String> missingOrInvalidValues = Lists.asList("", new String[] { "?" });
    // private List<String> missingOrInvalidValues = Lists.asList("", new String[] { "*", "#", "?", "null", "none" });

    /**
     * Auto type column feature, if eanabled by tree, shifu will set categorical or numerical feature automatically.
     * Since severl false-positive setting categorical features, such feature is disabled
     */
    private Boolean autoType = Boolean.FALSE;

    /**
     * If number ratio over autoTypeThreshold/100, column will be set to numeric when {@link #autoType} is true.
     */
    private Integer autoTypeThreshold = 0;

    /**
     * Meta column configuration file
     */
    private String metaColumnNameFile;

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

    public String getValidationDataPath() {
        return validationDataPath;
    }

    public void setValidationDataPath(String validationDataPath) {
        this.validationDataPath = validationDataPath;
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
        this.posTags = trimTags(posTags);
    }

    public List<String> getNegTags() {
        return negTags;
    }

    public void setNegTags(List<String> negTags) {
        this.negTags = trimTags(negTags);
    }

    private List<String> trimTags(List<String> tags) {
        if(tags != null) {
            List<String> trimmedTags = new ArrayList<String>();
            for(String tag: tags) {
                trimmedTags.add(CommonUtils.trimTag(tag));
            }
            return trimmedTags;
        } else {
            return null;
        }
    }

    public String getMetaColumnNameFile() {
        return metaColumnNameFile;
    }

    public void setMetaColumnNameFile(String metaColumnNameFile) {
        this.metaColumnNameFile = metaColumnNameFile;
    }

    @Override
    public RawSourceData clone() {
        RawSourceData copy = new RawSourceData();

        copy.setSource(source);
        copy.setDataPath(dataPath);
        copy.setDataDelimiter(dataDelimiter);
        copy.setHeaderPath(headerPath);
        copy.setHeaderDelimiter(headerDelimiter);
        copy.setFilterExpressions(filterExpressions);
        copy.setWeightColumnName(weightColumnName);

        copy.setTargetColumnName(targetColumnName);
        copy.setPosTags(new ArrayList<String>(posTags));
        copy.setNegTags(new ArrayList<String>(negTags));
        copy.setMissingOrInvalidValues(missingOrInvalidValues);
        copy.setMetaColumnNameFile(metaColumnNameFile);

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
