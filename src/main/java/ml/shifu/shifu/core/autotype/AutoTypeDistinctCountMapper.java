/*
 * Copyright [2012-2015] PayPal Software Foundation
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
package ml.shifu.shifu.core.autotype;

import java.io.IOException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.core.DataPurifier;
import ml.shifu.shifu.util.CommonUtils;
import ml.shifu.shifu.util.Constants;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.clearspring.analytics.stream.cardinality.HyperLogLogPlus;

/**
 * {@link AutoTypeDistinctCountMapper} is a mapper to get {@link HyperLogLogPlus} statistics per split. Such statistics
 * will be merged in our reducer.
 */
public class AutoTypeDistinctCountMapper extends Mapper<LongWritable, Text, IntWritable, CountAndFrequentItemsWritable> {

    private final static Logger LOG = LoggerFactory.getLogger(AutoTypeDistinctCountMapper.class);

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
     * Using approximate method to estimate real frequent items and store into this map
     */
    private Map<Integer, CountAndFrequentItems> variableCountMap;

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

    /**
     * Missing or invalid values
     */
    private Set<String> missingOrInvalidValues;

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

        this.dataPurifier = new DataPurifier(this.modelConfig);

        loadTagWeightNum();

        this.variableCountMap = new HashMap<Integer, CountAndFrequentItems>();

        this.outputKey = new IntWritable();

        this.tags = new HashSet<String>(modelConfig.getFlattenTags());

        this.missingOrInvalidValues = new HashSet<String>(this.modelConfig.getDataSet().getMissingOrInvalidValues());
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

    @Override
    protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
        String valueStr = value.toString();
        // StringUtils.isBlank is not used here to avoid import new jar
        if(valueStr == null || valueStr.length() == 0 || valueStr.trim().length() == 0) {
            LOG.warn("Empty input.");
            return;
        }

        context.getCounter(Constants.SHIFU_GROUP_COUNTER, "TOTAL_VALID_COUNT").increment(1L);

        if(!this.dataPurifier.isFilter(valueStr)) {
            context.getCounter(Constants.SHIFU_GROUP_COUNTER, "FILTER_OUT_COUNT").increment(1L);
            return;
        }

        String[] units = CommonUtils.split(valueStr, this.modelConfig.getDataSetDelimiter());
        // tagColumnNum should be in units array, if not IndexOutofBoundException
        String tag = CommonUtils.trimTag(units[this.tagColumnNum]);

        if(!this.tags.contains(tag)) {
            if(System.currentTimeMillis() % 50 == 0L) {
                LOG.warn("Data with invalid tag is ignored in distinct count computing, invalid tag: {}.", tag);
            }
            context.getCounter(Constants.SHIFU_GROUP_COUNTER, "INVALID_TAG").increment(1L);
            return;
        }

        int i = 0;
        for(String unit: units) {
            CountAndFrequentItems countAndFrequentItems = this.variableCountMap.get(i);
            if(countAndFrequentItems == null) {
                countAndFrequentItems = new CountAndFrequentItems();
                this.variableCountMap.put(i, countAndFrequentItems);
            }
            countAndFrequentItems.offer(this.missingOrInvalidValues, unit);
            i++;
        }
    }

    /**
     * Write column info to reducer for merging.
     */
    @Override
    protected void cleanup(Context context) throws IOException, InterruptedException {
        for(Map.Entry<Integer, CountAndFrequentItems> entry: this.variableCountMap.entrySet()) {
            this.outputKey.set(entry.getKey());
            byte[] bytes = entry.getValue().hyper.getBytes();
            Set<String> frequentItems = entry.getValue().frequentItems;
            context.write(this.outputKey, new CountAndFrequentItemsWritable(entry.getValue().count,
                    entry.getValue().invalidCount, entry.getValue().validNumCount, bytes, frequentItems));
        }
    }

    public static class CountAndFrequentItems {

        private final HyperLogLogPlus hyper = new HyperLogLogPlus(8);;

        private final Set<String> frequentItems = new HashSet<String>();

        private long count;

        private long invalidCount;

        private long validNumCount;

        public void offer(Set<String> missingorInvalidValues, String unit) {
            count += 1;

            if(unit == null || missingorInvalidValues.contains(unit.toLowerCase())) {
                invalidCount += 1;
                return;
            }

            hyper.offer(unit);

            try {
                Double.parseDouble(unit);
                validNumCount += 1;
            } catch (NumberFormatException e) {
                // ignore as only do stats on validNumCount
            }

            if(frequentItems.size() <= CountAndFrequentItemsWritable.FREQUET_ITEM_MAX_SIZE
                    && !frequentItems.contains(unit)) {
                frequentItems.add(unit);
            }
        }

        /**
         * @return the hyper
         */
        public HyperLogLogPlus getHyper() {
            return hyper;
        }

        /**
         * @return the frequentItems
         */
        public Set<String> getFrequentItems() {
            return frequentItems;
        }

        /**
         * @return the count
         */
        public long getCount() {
            return count;
        }

        /**
         * @return the invalidCount
         */
        public long getInvalidCount() {
            return invalidCount;
        }

        /**
         * @return the validNumCount
         */
        public long getValidNumCount() {
            return validNumCount;
        }

        /*
         * (non-Javadoc)
         * 
         * @see java.lang.Object#toString()
         */
        @Override
        public String toString() {
            return "CountAndFrequentItems [hyper=" + hyper + ", frequentItems=" + frequentItems + ", count=" + count
                    + ", invalidCount=" + invalidCount + ", validNumCount=" + validNumCount + "]";
        }

    }
}
