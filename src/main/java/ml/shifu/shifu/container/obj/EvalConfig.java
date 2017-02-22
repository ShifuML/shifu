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

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.fs.PathFinder;
import ml.shifu.shifu.util.CommonUtils;
import ml.shifu.shifu.util.Constants;

import org.apache.commons.lang.StringUtils;
import org.apache.hadoop.fs.Path;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

/**
 * EvalConfig class
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public class EvalConfig {

    private String name;
    private RawSourceData dataSet;

    private Integer performanceBucketNum = 10;
    private String performanceScoreSelector = "mean";
    private String scoreMetaColumnNameFile;
    private Map<String, String> customPaths;

    /**
     * For typical 0-1 binary regression, this is set to be true, while for other regression, better to set it to false
     * as normal regression.
     */
    private Boolean gbtConvertToProb = Boolean.TRUE;

    /**
     * Cache meta columns to a list to avoid reading this file for several times
     */
    @JsonIgnore
    private volatile List<String> metaColumns = null;

    /**
     * Cache raw score meta columns to avoid reading file several times
     */
    @JsonIgnore
    private volatile List<String> scoreMetaColumns = null;

    public EvalConfig() {
        customPaths = new HashMap<String, String>(1);
        /**
         * Since most user won't use this function,
         * hidden the custom paths for creating new model.
         */
        /*
         * customPaths.put(Constants.KEY_MODELS_PATH, null);
         * customPaths.put(Constants.KEY_SCORE_PATH, null);
         * customPaths.put(Constants.KEY_CONFUSION_MATRIX_PATH, null);
         * customPaths.put(Constants.KEY_PERFORMANCE_PATH, null);
         */
    }

    /**
     * @return the models_path
     */
    @JsonIgnore
    public String getModelsPath() {
        return ((customPaths == null) ? null : customPaths.get(Constants.KEY_MODELS_PATH));
    }

    /**
     * @return the score_path
     */
    @JsonIgnore
    public String getScorePath() {
        return ((customPaths == null) ? null : customPaths.get(Constants.KEY_SCORE_PATH));
    }

    /**
     * @return the performance_path
     */
    @JsonIgnore
    public String getPerformancePath() {
        return ((customPaths == null) ? null : customPaths.get(Constants.KEY_PERFORMANCE_PATH));
    }

    /**
     * @return the confusionMatrixPath
     */
    @JsonIgnore
    public String getConfusionMatrixPath() {
        return ((customPaths == null) ? null : customPaths.get(Constants.KEY_CONFUSION_MATRIX_PATH));
    }

    @JsonIgnore
    public List<String> getScoreMetaColumns(ModelConfig modelConfig) throws IOException {
        if(scoreMetaColumns == null) {
            synchronized(this) {
                if(scoreMetaColumns == null) {
                    PathFinder pathFinder = new PathFinder(modelConfig);
                    if(StringUtils.isNotBlank(scoreMetaColumnNameFile) && SourceType.HDFS.equals(dataSet.getSource())) {
                        String path = scoreMetaColumnNameFile;
                        File file = new File(scoreMetaColumnNameFile);
                        path = new Path(pathFinder.getEvalSetPath(this), file.getName()).toString();
                        scoreMetaColumns = CommonUtils.readConfFileIntoList(path, dataSet.getSource(),
                                dataSet.getHeaderDelimiter());
                    }
                }
            }
        }
        return scoreMetaColumns;
    }

    @JsonIgnore
    public List<String> getAllMetaColumns(ModelConfig modelConfig) throws IOException {
        if(metaColumns == null) {
            synchronized(this) {
                if(metaColumns == null) {
                    PathFinder pathFinder = new PathFinder(modelConfig);
                    List<String> scoreMetaColumns = null;
                    if(StringUtils.isNotBlank(scoreMetaColumnNameFile) && SourceType.HDFS.equals(dataSet.getSource())) {
                        String path = scoreMetaColumnNameFile;
                        File file = new File(scoreMetaColumnNameFile);
                        path = new Path(pathFinder.getEvalSetPath(this), file.getName()).toString();
                        scoreMetaColumns = CommonUtils.readConfFileIntoList(path, dataSet.getSource(),
                                dataSet.getHeaderDelimiter());
                        metaColumns = scoreMetaColumns;
                    }

                    if(StringUtils.isNotBlank(dataSet.getMetaColumnNameFile())) {
                        String rawMetaPath = dataSet.getMetaColumnNameFile();
                        if(StringUtils.isNotBlank(rawMetaPath) && SourceType.HDFS.equals(dataSet.getSource())) {
                            File file = new File(rawMetaPath);
                            rawMetaPath = new Path(pathFinder.getEvalSetPath(this), file.getName()).toString();
                        }
                        List<String> rawMetaColumns = CommonUtils.readConfFileIntoList(rawMetaPath,
                                dataSet.getSource(), dataSet.getHeaderDelimiter());
                        if(metaColumns != null) {
                            for(String column: rawMetaColumns) {
                                if(!metaColumns.contains(column)) {
                                    metaColumns.add(column);
                                }
                            }
                        } else {
                            metaColumns = rawMetaColumns;
                        }
                    }
                }
            }
        }
        return metaColumns;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public synchronized RawSourceData getDataSet() {
        return dataSet;
    }

    public synchronized void setDataSet(RawSourceData dataSet) {
        this.dataSet = dataSet;
    }

    public Integer getPerformanceBucketNum() {
        return performanceBucketNum;
    }

    public void setPerformanceBucketNum(Integer performanceBucketNum) {
        this.performanceBucketNum = performanceBucketNum;
    }

    public synchronized String getScoreMetaColumnNameFile() {
        return scoreMetaColumnNameFile;
    }

    public synchronized void setScoreMetaColumnNameFile(String scoreMetaColumnNameFile) {
        this.scoreMetaColumnNameFile = scoreMetaColumnNameFile;
    }

    public String getPerformanceScoreSelector() {
        return performanceScoreSelector;
    }

    public void setPerformanceScoreSelector(String performanceScoreSelector) {
        this.performanceScoreSelector = performanceScoreSelector;
    }

    public Map<String, String> getCustomPaths() {
        return customPaths;
    }

    public void setCustomPaths(Map<String, String> customPaths) {
        this.customPaths = customPaths;
    }

    /**
     * @return the gbtConvertToProb
     */
    public Boolean getGbtConvertToProb() {
        return gbtConvertToProb;
    }

    /**
     * @param gbtConvertToProb
     *            the gbtConvertToProb to set
     */
    public void setGbtConvertToProb(Boolean gbtConvertToProb) {
        this.gbtConvertToProb = gbtConvertToProb;
    }

    @Override
    public EvalConfig clone() {
        EvalConfig other = new EvalConfig();
        other.setCustomPaths(new HashMap<String, String>(customPaths));
        other.setDataSet(dataSet.clone());
        other.setGbtConvertToProb(gbtConvertToProb);
        other.setName(name);
        other.setPerformanceBucketNum(performanceBucketNum);
        other.setPerformanceScoreSelector(performanceScoreSelector);
        other.setScoreMetaColumnNameFile(scoreMetaColumnNameFile);
        return other;
    }
}
