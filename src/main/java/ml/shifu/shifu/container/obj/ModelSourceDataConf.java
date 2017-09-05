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

import java.util.ArrayList;

/**
 * ModelSourceDataConf class
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public class ModelSourceDataConf extends RawSourceData {

    private String categoricalColumnNameFile;

    private String hybridColumnNameFile;

    private String segExpressionFile;

    public String getCategoricalColumnNameFile() {
        return categoricalColumnNameFile;
    }

    public void setCategoricalColumnNameFile(String categoricalColumnNameFile) {
        this.categoricalColumnNameFile = categoricalColumnNameFile;
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
