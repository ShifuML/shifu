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

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

import ml.shifu.shifu.container.CaseScoreResult;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.core.DataPurifier;
import ml.shifu.shifu.core.ModelRunner;
import ml.shifu.shifu.core.posttrain.FeatureStatsWritable.BinStats;
import ml.shifu.shifu.util.CommonUtils;
import ml.shifu.shifu.util.Constants;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.TaskInputOutputContext;
import org.apache.hadoop.mapreduce.lib.output.MultipleOutputs;
import org.encog.ml.BasicML;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * TODO
 * 
 * @author Zhang David (pengzhang@paypal.com)
 */
public class PostTrainMapper extends Mapper<LongWritable, Text, IntWritable, FeatureStatsWritable> {

    private final static Logger LOG = LoggerFactory.getLogger(PostTrainMapper.class);

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
     * TODO using approximate method to estimate real frequent items and store into this map
     */
    // private Map<Integer, CountAndFrequentItems> variableCountMap;

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

    private ModelRunner modelRunner;

    private MultipleOutputs<Text, Text> mos;

    /**
     * Prevent too many new objects for output key.
     */
    private Text outputValue;

    private Map<Integer, List<BinStats>> variableStatsMap;

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

    @SuppressWarnings({ "rawtypes", "unchecked" })
    @Override
    protected void setup(Context context) throws IOException, InterruptedException {
        loadConfigFiles(context);

        this.dataPurifier = new DataPurifier(this.modelConfig);

        this.outputKey = new IntWritable();
        this.outputValue = new Text();

        this.variableStatsMap = new HashMap<Integer, List<BinStats>>();

        this.tags = new HashSet<String>(modelConfig.getFlattenTags());
        SourceType sourceType = this.modelConfig.getDataSet().getSource();

        List<BasicML> models = CommonUtils.loadBasicModels(modelConfig, columnConfigList, null, sourceType);
        this.headers = CommonUtils.getHeaders(this.modelConfig.getDataSet().getHeaderPath(), this.modelConfig
                .getDataSet().getDataDelimiter(), sourceType);
        this.modelRunner = new ModelRunner(modelConfig, columnConfigList, this.headers,
                modelConfig.getDataSetDelimiter(), models);

        this.mos = new MultipleOutputs<Text, Text>((TaskInputOutputContext) context);
    }

    @Override
    protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
        String valueStr = value.toString();
        // StringUtils.isBlank is not used here to avoid import new jar
        if(valueStr == null || valueStr.length() == 0 || valueStr.trim().length() == 0) {
            LOG.warn("Empty input.");
            return;
        }

        if(!this.dataPurifier.isFilterOut(valueStr)) {
            return;
        }

        String[] units = CommonUtils.split(valueStr, this.modelConfig.getDataSetDelimiter());
        // tagColumnNum should be in units array, if not IndexOutofBoundException
        String tag = units[this.tagColumnNum];

        if(!this.tags.contains(tag)) {
            if(System.currentTimeMillis() % 20 == 0) {
                LOG.warn("Data with invalid tag is ignored in post train, invalid tag: {}.", tag);
            }
            context.getCounter(Constants.SHIFU_GROUP_COUNTER, "INVALID_TAG").increment(1L);
            return;
        }

        Map<String, String> rawDataMap = buildRawDataMap(units);
        CaseScoreResult csr = this.modelRunner.compute(rawDataMap);
        StringBuilder sb = new StringBuilder(500);

        sb.append(csr.getAvgScore()).append(Constants.DEFAULT_DELIMITER).append(csr.getMaxScore())
                .append(Constants.DEFAULT_DELIMITER).append(csr.getMinScore()).append(Constants.DEFAULT_DELIMITER);
        for(Integer score: csr.getScores()) {
            sb.append(score).append(Constants.DEFAULT_DELIMITER);
        }
        List<String> metaList = modelConfig.getMetaColumnNames();
        for(String meta: metaList) {
            sb.append(rawDataMap.get(meta)).append(Constants.DEFAULT_DELIMITER);
        }
        sb.deleteCharAt(sb.length() - 1);

        this.outputValue.set(sb.toString());
        // TODO score ???
        this.mos.write("score", NullWritable.get(), this.outputValue);

        for(int i = 0; i < headers.length; i++) {
            ColumnConfig config = this.columnConfigList.get(i);
            if(!config.isMeta() && !config.isTarget() && config.isFinalSelect()) {
                int binNum = -1; // TODO
                List<BinStats> feaureStatistics = this.variableStatsMap.get(config.getColumnNum());
                if(feaureStatistics == null) {
                    if(config.isNumerical()) {
                        feaureStatistics = new ArrayList<BinStats>(config.getBinBoundary().size());
                    }
                    if(config.isCategorical()) {
                        feaureStatistics = new ArrayList<BinStats>(config.getBinCategory().size());
                    }
                }

                BinStats bs = feaureStatistics.get(binNum);
                if(bs == null) {
                    bs = new BinStats(csr.getAvgScore(), 1L);
                } else {
                    bs.setBinSum(csr.getAvgScore() + bs.getBinSum());
                    bs.setBinCnt(1L + bs.getBinCnt());
                }
            }
        }
    }

    private Map<String, String> buildRawDataMap(String[] units) {
        Map<String, String> rawDataMap = new HashMap<String, String>(headers.length, 1f);
        for(int i = 0; i < headers.length; i++) {
            if(units[i] == null) {
                rawDataMap.put(headers[i], "");
            } else {
                rawDataMap.put(headers[i], units[i].toString());
            }
        }
        return rawDataMap;
    }

    @Override
    protected void cleanup(Context context) throws IOException, InterruptedException {
        for(Entry<Integer, List<BinStats>> entry: this.variableStatsMap.entrySet()) {
            this.outputKey.set(entry.getKey());
            context.write(this.outputKey, new FeatureStatsWritable(entry.getValue()));
        }
    }
}
