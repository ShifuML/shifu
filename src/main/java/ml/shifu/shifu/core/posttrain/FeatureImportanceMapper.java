/*
 * Copyright [2013-2015] PayPal Software Foundation
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
package ml.shifu.shifu.core.posttrain;

import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.core.DataPurifier;
import ml.shifu.shifu.util.CommonUtils;
import ml.shifu.shifu.util.Constants;

/**
 * {@link FeatureImportanceMapper} is to compute the most important variables in one model.
 * 
 * <p>
 * Per each record, get the top 3 biggest variables in one bin. Then sent to reducer for further statistics.
 * 
 * @author Zhang David (pengzhang@paypal.com)
 */
public class FeatureImportanceMapper extends Mapper<LongWritable, Text, IntWritable, DoubleWritable> {

    private final static Logger LOG = LoggerFactory.getLogger(FeatureImportanceMapper.class);

    /**
     * Model Config read from HDFS
     */
    private ModelConfig modelConfig;

    /**
     * To filter records by customized expressions
     */
    private DataPurifier dataPurifier;

    /**
     * Output key cache to avoid new operation.
     */
    private IntWritable outputKey;

    /**
     * Tag column index
     */
    private int tagColumnNum = -1;

    /**
     * Column Config list read from HDFS
     */
    private List<ColumnConfig> columnConfigList;

    // cache tags in set for search
    private Set<String> tags;

    private String[] headers;

    /**
     * Prevent too many new objects for output key.
     */
    private DoubleWritable outputValue;

    private Map<Integer, Double> variableStatsMap;

    private void loadConfigFiles(final Context context) {
        try {
            SourceType sourceType = SourceType.valueOf(context.getConfiguration().get(
                    Constants.SHIFU_MODELSET_SOURCE_TYPE, SourceType.HDFS.toString()));
            this.modelConfig = CommonUtils.loadModelConfig(
                    context.getConfiguration().get(Constants.SHIFU_MODEL_CONFIG), sourceType);
            this.columnConfigList = CommonUtils.loadColumnConfigList(
                    context.getConfiguration().get(Constants.SHIFU_COLUMN_CONFIG), sourceType);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    protected void setup(Context context) throws IOException, InterruptedException {
        loadConfigFiles(context);

        loadTagWeightNum();

        this.dataPurifier = new DataPurifier(this.modelConfig);

        this.outputKey = new IntWritable();
        this.outputValue = new DoubleWritable();

        this.tags = new HashSet<String>(modelConfig.getFlattenTags());

        this.headers = CommonUtils.getFinalHeaders(modelConfig);

        this.initFeatureStats();
    }

    /**
     * Load tag weight index field.
     */
    private void loadTagWeightNum() {
        for(ColumnConfig config: this.columnConfigList) {
            if(config.isTarget()) {
                this.tagColumnNum = config.getColumnNum();
                break;
            }
        }

        if(this.tagColumnNum == -1) {
            throw new RuntimeException("No valid target column.");
        }
    }

    private void initFeatureStats() {
        this.variableStatsMap = new HashMap<Integer, Double>();
        for(ColumnConfig config: this.columnConfigList) {
            if(!config.isMeta() && !config.isTarget() && config.isFinalSelect()) {
                this.variableStatsMap.put(config.getColumnNum(), 0d);
            }
        }
    }

    public static class FeatureScore {

        public FeatureScore(int columnNum, int binAvgScore) {
            super();
            this.columnNum = columnNum;
            this.binAvgScore = binAvgScore;
        }

        private int columnNum;
        private int binAvgScore;
    }

    @Override
    protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
        String valueStr = value.toString();
        // StringUtils.isBlank is not used here to avoid import new jar
        if(valueStr == null || valueStr.length() == 0 || valueStr.trim().length() == 0) {
            LOG.warn("Empty input.");
            return;
        }

        if(!this.dataPurifier.isFilter(valueStr)) {
            return;
        }

        String[] units = CommonUtils.split(valueStr, this.modelConfig.getDataSetDelimiter());
        // tagColumnNum should be in units array, if not IndexOutofBoundException
        String tag = CommonUtils.trimTag(units[this.tagColumnNum]);

        if(!this.tags.contains(tag)) {
            if(System.currentTimeMillis() % 20 == 0) {
                LOG.warn("Data with invalid tag is ignored in post train, invalid tag: {}.", tag);
            }
            context.getCounter(Constants.SHIFU_GROUP_COUNTER, "INVALID_TAG").increment(1L);
            return;
        }

        List<FeatureScore> featureScores = new ArrayList<FeatureImportanceMapper.FeatureScore>();
        for(int i = 0; i < headers.length; i++) {
            ColumnConfig config = this.columnConfigList.get(i);
            if(!config.isMeta() && !config.isTarget() && config.isFinalSelect()) {
                int binNum = CommonUtils.getBinNum(config, units[i]);
                List<Integer> binAvgScores = config.getBinAvgScore();
                int binScore = 0;
                if(binNum == -1) {
                    binScore = binAvgScores.get(binAvgScores.size() - 1);
                } else {
                    binScore = binAvgScores.get(binNum);
                }
                featureScores.add(new FeatureScore(config.getColumnNum(), binScore));
            }
        }
        Collections.sort(featureScores, new Comparator<FeatureScore>() {
            @Override
            public int compare(FeatureScore fs1, FeatureScore fs2) {
                if(fs1.binAvgScore < fs2.binAvgScore) {
                    return 1;
                }
                if(fs1.binAvgScore > fs2.binAvgScore) {
                    return -1;
                }

                return 0;
            }
        });

        int size = featureScores.size() >= 3 ? 3 : featureScores.size();
        for(int i = 0; i < size; i++) {
            FeatureScore featureScore = featureScores.get(i);
            Double currValue = this.variableStatsMap.get(featureScore.columnNum);
            currValue += size - i;
            this.variableStatsMap.put(featureScore.columnNum, currValue);
        }
    }

    @Override
    protected void cleanup(Context context) throws IOException, InterruptedException {
        for(Entry<Integer, Double> entry: this.variableStatsMap.entrySet()) {
            this.outputKey.set(entry.getKey());
            this.outputValue.set(entry.getValue());
            context.write(this.outputKey, this.outputValue);
        }
    }
}
