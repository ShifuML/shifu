/*
 * Copyright [2013-2017] PayPal Software Foundation
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
import ml.shifu.shifu.container.obj.EvalConfig;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.fs.ShifuFileUtils;
import ml.shifu.shifu.util.CommonUtils;
import ml.shifu.shifu.util.Constants;

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

/**
 * To project only useful columns used in eval sorting. Meta, target, weight and score columns should be included.
 */
public class ColumnProjector extends AbstractTrainerUDF<Tuple> {

    private EvalConfig evalConfig;

    private String scoreMetaColumn;

    private String[] headers;

    private int targetColumnIndex = -1;

    private int weightColumnIndex = -1;

    private int scoreMetaColumnIndex = -1;

    private double maxScore = Double.MIN_VALUE;

    private double minScore = Double.MAX_VALUE;

    /**
     * A simple weight exception validation: if over 5000 throw exceptions
     */
    private int weightExceptions;

    public ColumnProjector(String source, String pathModelConfig, String pathColumnConfig) throws IOException {
        super(source, pathModelConfig, pathColumnConfig);
    }

    public ColumnProjector(String source, String pathModelConfig, String pathColumnConfig, String evalSetName,
            String columnName) throws IOException {
        super(source, pathModelConfig, pathColumnConfig);
        this.evalConfig = modelConfig.getEvalConfigByName(evalSetName);
        this.scoreMetaColumn = columnName;

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

        for(int i = 0; i < this.headers.length; i++) {
            if(this.headers[i].equals(evalConfig.getDataSet().getTargetColumnName())) {
                this.targetColumnIndex = i;
            }
            if(this.headers[i].equals(this.scoreMetaColumn)) {
                this.scoreMetaColumnIndex = i;
            }
            if(StringUtils.isNotBlank(evalConfig.getDataSet().getWeightColumnName())
                    && this.headers[i].equals(evalConfig.getDataSet().getWeightColumnName())) {
                this.weightColumnIndex = i;
            }
        }
    }

    @SuppressWarnings("deprecation")
    @Override
    public Tuple exec(Tuple input) throws IOException {
        Tuple tuple = TupleFactory.getInstance().newTuple(3);
        String tag = input.get(targetColumnIndex).toString();
        tuple.set(0, tag);
        double score = 0;
        try {
            score = Double.parseDouble(input.get(scoreMetaColumnIndex).toString());
        } catch (Exception e) {
            if(isPigEnabled(Constants.SHIFU_GROUP_COUNTER, "BAD_META_SCORE")) {
                PigStatusReporter.getInstance().getCounter(Constants.SHIFU_GROUP_COUNTER, "BAD_META_SCORE")
                        .increment(1);
            }
        }
        if(score > maxScore) {
            maxScore = score;
        }
        if(score < minScore) {
            minScore = score;
        }

        tuple.set(1, score);
        String weight = "1";
        if(weightColumnIndex != -1) {
            weight = input.get(weightColumnIndex).toString();
        }
        tuple.set(2, weight);

        incrementTagCounters(tag, weight);
        return tuple;
    }

    @SuppressWarnings("deprecation")
    private void incrementTagCounters(String tag, String weight) {
        if(tag == null || weight == null) {
            log.warn("tag is empty " + tag + " or weight is empty " + weight);
            return;
        }
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

    @Override
    public void finish() {
        if(modelConfig.isClassification()) {
            return;
        }

        // only for regression, in some cases like gbdt, it's regression score is not in [0,1], to do eval performance,
        // max and min score should be collected to set bounds.
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

    @Override
    public Schema outputSchema(Schema input) {
        try {
            Schema tupleSchema = new Schema();
            tupleSchema.add(new FieldSchema("target", DataType.CHARARRAY));
            tupleSchema.add(new FieldSchema(scoreMetaColumn, DataType.DOUBLE));
            tupleSchema.add(new FieldSchema("weight", DataType.CHARARRAY));

            return new Schema(new Schema.FieldSchema("score", tupleSchema, DataType.TUPLE));
        } catch (IOException e) {
            log.error("Error in outputSchema", e);
            return null;
        }
    }

}
