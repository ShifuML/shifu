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
package ml.shifu.shifu.udf;

import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.Map;

import ml.shifu.shifu.container.CaseScoreResult;
import ml.shifu.shifu.container.obj.EvalConfig;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.core.ModelRunner;
import ml.shifu.shifu.fs.ShifuFileUtils;
import ml.shifu.shifu.util.CommonUtils;
import ml.shifu.shifu.util.Constants;

import org.apache.commons.collections.CollectionUtils;
import org.apache.commons.collections.MapUtils;
import org.apache.commons.lang.StringUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.pig.data.DataType;
import org.apache.pig.data.Tuple;
import org.apache.pig.data.TupleFactory;
import org.apache.pig.impl.logicalLayer.schema.Schema;
import org.apache.pig.impl.logicalLayer.schema.Schema.FieldSchema;
import org.apache.pig.impl.util.UDFContext;
import org.apache.pig.tools.pigstats.PigStatusReporter;
import org.encog.ml.BasicML;

/**
 * Calculate the score for each evaluation data
 */
public class EvalScoreUDF extends AbstractTrainerUDF<Tuple> {

    private static final String SCHEMA_PREFIX = "shifu::";

    private EvalConfig evalConfig;
    private ModelRunner modelRunner;
    private String[] headers;

    private int modelCnt;

    private int maxScore = Integer.MIN_VALUE;

    private int minScore = Integer.MAX_VALUE;

    /**
     * A simple weight exception validation: if over 5000 throw exceptions
     */
    private int weightExceptions;

    public EvalScoreUDF(String source, String pathModelConfig, String pathColumnConfig, String evalSetName)
            throws IOException {
        super(source, pathModelConfig, pathColumnConfig);

        evalConfig = modelConfig.getEvalConfigByName(evalSetName);

        if(evalConfig.getModelsPath() != null) {
            // renew columnConfig
            this.columnConfigList = ShifuFileUtils.searchColumnConfig(evalConfig, columnConfigList);
        }

        // create model runner
        if(StringUtils.isNotBlank(evalConfig.getDataSet().getHeaderPath())) {
            this.headers = CommonUtils.getHeaders(evalConfig.getDataSet().getHeaderPath(), evalConfig.getDataSet()
                    .getHeaderDelimiter(), evalConfig.getDataSet().getSource());
        } else {
            String delimiter = StringUtils.isBlank(evalConfig.getDataSet().getHeaderDelimiter()) ? evalConfig
                    .getDataSet().getDataDelimiter() : evalConfig.getDataSet().getHeaderDelimiter();
            String[] fields = CommonUtils.takeFirstLine(evalConfig.getDataSet().getDataPath(), delimiter, evalConfig
                    .getDataSet().getSource());
            if(StringUtils.join(fields, "").contains(modelConfig.getTargetColumnName())) {
                this.headers = new String[fields.length];
                for(int i = 0; i < fields.length; i++) {
                    this.headers[i] = CommonUtils.getRelativePigHeaderColumnName(fields[i]);
                }
                log.warn("No header path is provided, we will try to read first line and detect schema.");
                log.warn("Schema in ColumnConfig.json are named as first line of data set path.");
            } else {
                log.warn("No header path is provided, we will try to read first line and detect schema.");
                log.warn("Schema in ColumnConfig.json are named as  index 0, 1, 2, 3 ...");
                log.warn("Please make sure weight column and tag column are also taking index as name.");
                this.headers = new String[fields.length];
                for(int i = 0; i < fields.length; i++) {
                    this.headers[i] = i + "";
                }
            }
        }

        // move model runner construction in exec to avoid OOM error in client side if model is too big like RF
        this.modelCnt = CommonUtils.getBasicModelsCnt(modelConfig, this.columnConfigList, evalConfig, evalConfig
                .getDataSet().getSource());
    }

    public Tuple exec(Tuple input) throws IOException {
        if(modelRunner == null) {
            // here to initialize modelRunner, this is moved from constructor to here to avoid OOM in client side.
            // UDF in pig client will be initialized to get some metadata issues
            List<BasicML> models = CommonUtils.loadBasicModels(modelConfig, this.columnConfigList, evalConfig,
                    evalConfig.getDataSet().getSource(), evalConfig.getGbtConvertToProb());
            modelRunner = new ModelRunner(modelConfig, columnConfigList, this.headers, evalConfig.getDataSet()
                    .getDataDelimiter(), models);
            modelCnt = models.size();
        }
        Map<String, String> rawDataMap = CommonUtils.convertDataIntoMap(input, this.headers);
        if(MapUtils.isEmpty(rawDataMap)) {
            return null;
        }

        String tag = rawDataMap.get(modelConfig.getTargetColumnName(evalConfig));

        // filter invalid tag record out
        // disable the tag check, since there is no bad tag in eval data set
        // and user just want to score the data, but don't run performance evaluation
        /*
         * if(!tagSet.contains(tag)) {
         * if(System.currentTimeMillis() % 100 == 0) {
         * log.warn("Invalid tag: " + tag);
         * }
         * if(isPigEnabled(Constants.SHIFU_GROUP_COUNTER, "INVALID_TAG")) {
         * PigStatusReporter.getInstance().getCounter(Constants.SHIFU_GROUP_COUNTER, Constants.COUNTER_RECORDS)
         * .increment(1);
         * }
         * return null;
         * }
         */

        long startTime = System.currentTimeMillis();
        CaseScoreResult cs = modelRunner.compute(rawDataMap);
        long runInterval = System.currentTimeMillis() - startTime;

        if(cs == null) {
            if(System.currentTimeMillis() % 50 == 0) {
                log.warn("Get null result, for input: " + input.toDelimitedString("|"));
            }
            return null;
        }

        Tuple tuple = TupleFactory.getInstance().newTuple();
        tuple.append(StringUtils.trimToEmpty(tag));

        String weight = null;
        if(StringUtils.isNotBlank(evalConfig.getDataSet().getWeightColumnName())) {
            weight = rawDataMap.get(evalConfig.getDataSet().getWeightColumnName());
        } else {
            weight = "1.0";
        }

        incrementTagCounters(tag, weight, runInterval);

        tuple.append(weight);

        if(modelConfig.isRegression()) {
            tuple.append(cs.getAvgScore());
            tuple.append(cs.getMaxScore());
            tuple.append(cs.getMinScore());
            tuple.append(cs.getMedianScore());

            for(Integer score: cs.getScores()) {
                tuple.append(score);
            }

            // get maxScore and minScore for such mapper or reducer
            if(cs.getMedianScore() > maxScore) {
                maxScore = cs.getMedianScore();
            }

            if(cs.getMedianScore() < minScore) {
                minScore = cs.getMedianScore();
            }
        } else {
            for(int i = 0; i < cs.getScores().size(); i++) {
                tuple.append(cs.getScores().get(i));
            }
        }

        // append meta data
        List<String> metaColumns = evalConfig.getScoreMetaColumns(modelConfig);
        if(CollectionUtils.isNotEmpty(metaColumns)) {
            for(String meta: metaColumns) {
                tuple.append(rawDataMap.get(meta));
            }
        }

        return tuple;
    }

    @Override
    public void finish() {
        this.modelRunner.close();

        if(modelConfig.isClassification()) {
            return;
        }

        // only for regression
        BufferedWriter writer = null;
        Configuration jobConf = UDFContext.getUDFContext().getJobConf();
        String scoreOutput = jobConf.get(Constants.SHIFU_EVAL_MAXMIN_SCORE_OUTPUT);

        log.debug("shifu.eval.maxmin.score.output is {}, job id is {}, task id is {}, attempt id is {}" + scoreOutput
                + " " + jobConf.get("mapreduce.job.id") + " " + jobConf.get("mapreduce.task.id") + " "
                + jobConf.get("mapreduce.task.partition") + " " + jobConf.get("mapreduce.task.attempt.id"));

        try {
            FileSystem fileSystem = FileSystem.get(jobConf);
            fileSystem.mkdirs(new Path(scoreOutput));
            String taskMaxMinScoreFile = scoreOutput + File.separator + "part-"
                    + jobConf.get("mapreduce.task.attempt.id");
            writer = ShifuFileUtils.getWriter(taskMaxMinScoreFile, SourceType.HDFS);
            writer.write(maxScore + "," + minScore);
        } catch (IOException e) {
            log.error("error in finish", e);
        } finally {
            if(writer != null) {
                try {
                    writer.close();
                } catch (IOException ignore) {
                }
            }
        }
    }

    @SuppressWarnings("deprecation")
    private void incrementTagCounters(String tag, String weight, long runModelInterval) {
        if(tag == null || weight == null) {
            log.warn("tag is empty " + tag + " or weight is empty " + weight);
            return;
        }
        // TODO default weight here = 1 ? or throw exceptions
        double dWeight = 1.0;
        if(StringUtils.isNotBlank(weight)) {
            try {
                dWeight = Double.parseDouble(weight);
            } catch (Exception e) {
                if(isPigEnabled(Constants.SHIFU_GROUP_COUNTER, "weight_exceptions")) {
                    PigStatusReporter.getInstance().getCounter(Constants.SHIFU_GROUP_COUNTER, "weight_exceptions")
                            .increment(1);
                }
                weightExceptions += 1;
                if(weightExceptions > 5000) {
                    throw new IllegalStateException(
                            "Please check weight column in eval, exceptional weight count is over 5000");
                }
            }
        }
        long weightLong = (long) (dWeight * Constants.EVAL_COUNTER_WEIGHT_SCALE);

        // update model run time for stats
        if(isPigEnabled(Constants.SHIFU_GROUP_COUNTER, Constants.TOTAL_MODEL_RUNTIME)) {
            PigStatusReporter.getInstance().getCounter(Constants.SHIFU_GROUP_COUNTER, Constants.TOTAL_MODEL_RUNTIME)
                    .increment(runModelInterval);
        }

        if(isPigEnabled(Constants.SHIFU_GROUP_COUNTER, Constants.COUNTER_RECORDS)) {
            PigStatusReporter.getInstance().getCounter(Constants.SHIFU_GROUP_COUNTER, Constants.COUNTER_RECORDS)
                    .increment(1);
        }

        if(posTagSet.contains(tag)) {
            if(isPigEnabled(Constants.SHIFU_GROUP_COUNTER, Constants.COUNTER_POSTAGS)) {
                PigStatusReporter.getInstance().getCounter(Constants.SHIFU_GROUP_COUNTER, Constants.COUNTER_POSTAGS)
                        .increment(1);
            }
            if(isPigEnabled(Constants.SHIFU_GROUP_COUNTER, Constants.COUNTER_WPOSTAGS)) {
                PigStatusReporter.getInstance().getCounter(Constants.SHIFU_GROUP_COUNTER, Constants.COUNTER_WPOSTAGS)
                        .increment(weightLong);
            }
        }

        if(negTagSet.contains(tag)) {
            if(isPigEnabled(Constants.SHIFU_GROUP_COUNTER, Constants.COUNTER_NEGTAGS)) {
                PigStatusReporter.getInstance().getCounter(Constants.SHIFU_GROUP_COUNTER, Constants.COUNTER_NEGTAGS)
                        .increment(1);
            }
            if(isPigEnabled(Constants.SHIFU_GROUP_COUNTER, Constants.COUNTER_WNEGTAGS)) {
                PigStatusReporter.getInstance().getCounter(Constants.SHIFU_GROUP_COUNTER, Constants.COUNTER_WNEGTAGS)
                        .increment(weightLong);
            }
        }
    }

    /**
     * output the schema for evaluation score
     */
    public Schema outputSchema(Schema input) {
        try {
            Schema tupleSchema = new Schema();
            tupleSchema.add(new FieldSchema(SCHEMA_PREFIX + modelConfig.getTargetColumnName(evalConfig),
                    DataType.CHARARRAY));

            String weightName = StringUtils.isBlank(evalConfig.getDataSet().getWeightColumnName()) ? "weight"
                    : evalConfig.getDataSet().getWeightColumnName();
            tupleSchema.add(new FieldSchema(SCHEMA_PREFIX + weightName, DataType.CHARARRAY));

            if(modelConfig.isRegression()) {
                tupleSchema.add(new FieldSchema(SCHEMA_PREFIX + "mean", DataType.INTEGER));
                tupleSchema.add(new FieldSchema(SCHEMA_PREFIX + "max", DataType.INTEGER));
                tupleSchema.add(new FieldSchema(SCHEMA_PREFIX + "min", DataType.INTEGER));
                tupleSchema.add(new FieldSchema(SCHEMA_PREFIX + "median", DataType.INTEGER));
                for(int i = 0; i < modelCnt; i++) {
                    tupleSchema.add(new FieldSchema(SCHEMA_PREFIX + "model" + i, DataType.INTEGER));
                }
            } else {
                for(int i = 0; i < modelCnt; i++) {
                    for(int j = 0; j < modelConfig.getTags().size(); j++) {
                        tupleSchema.add(new FieldSchema(SCHEMA_PREFIX + "model_" + i + "_tag_" + j, DataType.INTEGER));
                    }
                }
            }

            List<String> metaColumns = evalConfig.getScoreMetaColumns(modelConfig);
            if(CollectionUtils.isNotEmpty(metaColumns)) {
                for(String columnName: metaColumns) {
                    tupleSchema.add(new FieldSchema(columnName, DataType.CHARARRAY));
                }
            }

            return new Schema(new Schema.FieldSchema("EvalScore", tupleSchema, DataType.TUPLE));
        } catch (IOException e) {
            log.error("Error in outputSchema", e);
            return null;
        }
    }

}
