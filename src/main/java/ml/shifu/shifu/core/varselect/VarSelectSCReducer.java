package ml.shifu.shifu.core.varselect;

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.RawSourceData;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.core.dtrain.CommonConstants;
import ml.shifu.shifu.core.dtrain.DTrainUtils;
import ml.shifu.shifu.util.CommonUtils;
import ml.shifu.shifu.util.Constants;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Copyright [2013-2018] PayPal Software Foundation
 * <p>
 * Licensed under the Apache License, Version 2.0 (the "License")
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * <p>
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 **/

public class VarSelectSCReducer extends Reducer<LongWritable, ColumnScore, LongWritable, Text> {

    private final static Logger LOG = LoggerFactory.getLogger(VarSelectSCReducer.class);

    /**
     * Model Config read from HDFS
     */
    private ModelConfig modelConfig;

    /**
     * Column Config list read from HDFS
     */
    private List<ColumnConfig> columnConfigList;

    /**
     * Wrapper by sensitivity by target(ST) or sensitivity(SE).
     */
    private String filterBy;

    /**
     * Explicit set number of variables to be selected,this overwrites filterOutRatio
     */
    private int filterNum;

    /**
     * To set as a ratio instead an absolute number, each time it is
     * a ratio. For example, 100 variables, using ratio 0.05, first time select 95 variables, next as candidates are
     * decreasing, next time it is still 0.05, but only 4 variables are removed.
     */
    private float filterOutRatio;

    private OpMetric opMetric = OpMetric.ACTION_RATE;
    private double opUnit;

    /**
     * Load all configurations for modelConfig and columnConfigList from source type.
     */
    private void loadConfigFiles(final Context context) {
        try {
            RawSourceData.SourceType sourceType = SourceType.valueOf(
                    context.getConfiguration().get(Constants.SHIFU_MODELSET_SOURCE_TYPE, SourceType.HDFS.toString()));
            this.modelConfig = CommonUtils
                    .loadModelConfig(context.getConfiguration().get(Constants.SHIFU_MODEL_CONFIG), sourceType);
            this.columnConfigList = CommonUtils
                    .loadColumnConfigList(context.getConfiguration().get(Constants.SHIFU_COLUMN_CONFIG), sourceType);

            Object obj = null;
            if (modelConfig.getVarSelect().getParams() != null) {
                obj = modelConfig.getVarSelect().getParams().get(CommonConstants.OP_METRIC);
            }
            if(obj != null) {
                try {
                    this.opMetric = OpMetric.valueOf(obj.toString());
                } catch (Exception e) {
                    this.opMetric = OpMetric.ACTION_RATE;
                    // use default
                }
            }
            this.opUnit = DTrainUtils.getDouble(modelConfig.getVarSelect().getParams(),
                    CommonConstants.OP_UNIT, 0.01d);

            LOG.info("opMetric = {}, opUinit = {}", this.opMetric, this.opUnit);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Do initialization like ModelConfig and ColumnConfig loading.
     */
    @Override
    protected void setup(Context context) {
        loadConfigFiles(context);

        this.filterOutRatio = context.getConfiguration().getFloat(Constants.SHIFU_VARSELECT_FILTEROUT_RATIO, Constants.SHIFU_DEFAULT_VARSELECT_FILTEROUT_RATIO);
        this.filterNum = context.getConfiguration().getInt(Constants.SHIFU_VARSELECT_FILTER_NUM, Constants.SHIFU_DEFAULT_VARSELECT_FILTER_NUM);
        this.filterBy = context.getConfiguration()
                .get(Constants.SHIFU_VARSELECT_FILTEROUT_TYPE, Constants.FILTER_BY_SE);
        LOG.info("FilterBy is {}, filterOutRatio is {}, filterNum is {}", filterBy, filterOutRatio, filterNum);
    }

    @Override
    protected void reduce(LongWritable key, Iterable<ColumnScore> values, Context context)
            throws IOException, InterruptedException {
        List<ColumnScore> columnScores = new ArrayList<>();
        for(ColumnScore columnScore : values) {
            columnScores.add(columnScore.clone());
        }

        LOG.info("The column id is {}, and there are {} scores.", key.get(), columnScores.size());
        for (int i = 0; i < 100; i ++ ) {
            LOG.info(columnScores.get(i).toString());
        }

        VarSelPerfGenerator generator = new VarSelPerfGenerator(columnScores, this.opMetric, this.opUnit);
        Text output = new Text(Double.toString(generator.getSensitivityPerf()));
        context.write(key, output);
    }

}