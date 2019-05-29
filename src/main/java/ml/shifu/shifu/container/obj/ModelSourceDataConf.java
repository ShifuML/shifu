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
import com.google.common.collect.Lists;

import java.util.ArrayList;
import java.util.List;

/**
 * ModelSourceDataConf class
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public class ModelSourceDataConf extends RawSourceData {

    private String categoricalColumnNameFile;

    /**
     * Validation data path which is used in train step for validation data. Such data should have the same schema like
     * {@link #dataPath}. If {@link #validationDataPath} is not empty, specified validation data is enabled and all
     * other sampling parameters have no effect. If empty (by default), such feature is not enabled.
     */
    private String validationDataPath;

    /**
     * Filter expression on validation data path, this is helpful to filter some data not in original
     * data. Example like 'column_a > 10'
     */
    private String validationFilterExpressions = "";

    /**
     * Missing or invalid values.
     */
    private List<String> missingOrInvalidValues = Lists.asList("", new String[] { "?" });
    // private List<String> missingOrInvalidValues = Lists.asList("", new String[] { "*", "#", "?", "null", "none" });

    private String hybridColumnNameFile;

    private String segExpressionFile;
    
    private String categoricalHashSeedConfFile;

    public String getCategoricalHashSeedConfFile() {
		return categoricalHashSeedConfFile;
	}

	public void setCategoricalHashSeedConfFile(String categoricalHashSeedConfFile) {
		this.categoricalHashSeedConfFile = categoricalHashSeedConfFile;
	}

	public String getCategoricalColumnNameFile() {
        return categoricalColumnNameFile;
    }

    public void setCategoricalColumnNameFile(String categoricalColumnNameFile) {
        this.categoricalColumnNameFile = categoricalColumnNameFile;
    }

    public String getValidationDataPath() {
        return validationDataPath;
    }

    public void setValidationDataPath(String validationDataPath) {
        this.validationDataPath = validationDataPath;
    }

    public String getValidationFilterExpressions() {
        return validationFilterExpressions;
    }

    public void setValidationFilterExpressions(String validationFilterExpressions) {
        this.validationFilterExpressions = validationFilterExpressions;
    }

    public List<String> getMissingOrInvalidValues() {
        return missingOrInvalidValues;
    }

    public void setMissingOrInvalidValues(List<String> missingOrInvalidValues) {
        this.missingOrInvalidValues = missingOrInvalidValues;
    }

    @JsonIgnoreProperties(ignoreUnknown = true)
    public RawSourceData cloneRawSourceData() {
        return super.clone();
    }

    /**
     * @return the hybridColumnNameFile
     */
    @JsonIgnore
    public String getHybridColumnNameFile() {
        return hybridColumnNameFile;
    }

    /**
     * @param hybridColumnNameFile
     *            the hybridColumnNameFile to set
     */
    @JsonProperty
    public void setHybridColumnNameFile(String hybridColumnNameFile) {
        this.hybridColumnNameFile = hybridColumnNameFile;
    }

    @Override
    public ModelSourceDataConf clone() {
        ModelSourceDataConf other = new ModelSourceDataConf();

        other.setSource(this.getSource());
        other.setDataPath(this.getDataPath());
        other.setDataDelimiter(this.getDataDelimiter());
        other.setHeaderPath(this.getHeaderPath());
        other.setHeaderDelimiter(this.getHeaderDelimiter());
        other.setFilterExpressions(this.getFilterExpressions());
        other.setWeightColumnName(this.getWeightColumnName());

        other.setTargetColumnName(this.getTargetColumnName());
        other.setPosTags(new ArrayList<String>(this.getPosTags()));
        other.setNegTags(new ArrayList<String>(this.getNegTags()));

        other.setValidationDataPath(this.validationDataPath);
        other.setValidationFilterExpressions(this.validationFilterExpressions);
        other.setMissingOrInvalidValues(this.getMissingOrInvalidValues());

        other.setCategoricalColumnNameFile(categoricalColumnNameFile);
        other.setHybridColumnNameFile(this.hybridColumnNameFile);
        other.setSegExpressionFile(this.segExpressionFile);
        other.setMetaColumnNameFile(this.getMetaColumnNameFile());

        return other;
    }

    /**
     * @return the segExpressionFile
     */
    @JsonIgnore
    public String getSegExpressionFile() {
        return segExpressionFile;
    }

    /**
     * @param segExpressionFile
     *            the segExpressionFile to set
     */
    @JsonProperty
    public void setSegExpressionFile(String segExpressionFile) {
        this.segExpressionFile = segExpressionFile;
    }

}
