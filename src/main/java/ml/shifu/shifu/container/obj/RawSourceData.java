/**
 * Copyright [2012-2014] eBay Software Foundation
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

import com.fasterxml.jackson.databind.annotation.JsonDeserialize;

import java.util.ArrayList;
import java.util.List;

/**
 * SourceData class
 */
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

        return copy;
    }
}
