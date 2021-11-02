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

import ml.shifu.shifu.column.NSColumn;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.fs.PathFinder;
import ml.shifu.shifu.fs.ShifuFileUtils;
import ml.shifu.shifu.util.*;
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

import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;

/**
 * To project only useful columns used in eval sorting. Meta, target, weight and score columns should be included.
 */
public class ColumnProjector extends AbstractEvalUDF<Tuple> {

    private String[] headers;
    private String scoreMetaColumn;

    private int targetColumnIndex = -1;
    private int weightColumnIndex = -1;
    private int scoreMetaColumnIndex = -1;

    private double maxScore = Double.MIN_VALUE;
    private double minScore = Double.MAX_VALUE;

    /**
     * A simple weight exception validation: if over 5000 throw exceptions
     */
    private int weightExceptions;

    public ColumnProjector(String source, String pathModelConfig, String pathColumnConfig, String evalSetName,
            String columnName) throws IOException {
        super(source, pathModelConfig, pathColumnConfig, evalSetName);
        this.scoreMetaColumn = columnName;

        PathFinder pathFinder = new PathFinder(this.modelConfig);
        String scoreHeaderPath = pathFinder.getEvalScoreHeaderPath(this.evalConfig);
        String delimiter = getUdfProperty(Constants.SHIFU_OUTPUT_DATA_DELIMITER, Constants.DEFAULT_DELIMITER);
        delimiter = Base64Utils.base64DecodeIfEncodedInput(delimiter);
        Environment.setProperty(Constants.SHIFU_NAMESPACE_STRICT_MODE, Boolean.TRUE.toString());
        this.headers = CommonUtils.getHeaders(scoreHeaderPath, delimiter, evalConfig.getDataSet().getSource());

        NSColumn target = new NSColumn(modelConfig.getTargetColumnName(evalConfig, modelConfig.getTargetColumnName()));
        NSColumn scoreMeta = new NSColumn(this.scoreMetaColumn);
        NSColumn weight = new NSColumn("shifu::"
                + (StringUtils.isBlank(evalConfig.getDataSet().getWeightColumnName()) ? "weight"
                    : evalConfig.getDataSet().getWeightColumnName()));

        for(int i = 0; i < this.headers.length; i++) {
            NSColumn nsColumn = new NSColumn(this.headers[i]);
            if (this.targetColumnIndex < 0 && nsColumn.equals(target)) {
                this.targetColumnIndex = i;
            } else if (this.weightColumnIndex < 0 && nsColumn.equals(weight)) {
                this.weightColumnIndex = i;
            } else if (this.scoreMetaColumnIndex < 0 && nsColumn.equals(scoreMeta)) {
                this.scoreMetaColumnIndex = i;
            }
        }
    }

    @SuppressWarnings("deprecation")
    @Override
    public Tuple exec(Tuple input) throws IOException {
        Tuple tuple = TupleFactory.getInstance().newTuple(3);
        Object tval = input.get(targetColumnIndex);
        String tag = CommonUtils.trimTag((tval == null) ? "" : tval.toString());
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
            tupleSchema.add(new FieldSchema(modelConfig.getTargetColumnName(evalConfig, "target"),
                    DataType.CHARARRAY));
            tupleSchema.add(new FieldSchema(scoreMetaColumn, DataType.DOUBLE));
            tupleSchema.add(new FieldSchema(StringUtils.isBlank(evalConfig.getDataSet().getWeightColumnName())
                    ? "weight" : evalConfig.getDataSet().getWeightColumnName(), DataType.CHARARRAY));

            return new Schema(new Schema.FieldSchema("score", tupleSchema, DataType.TUPLE));
        } catch (IOException e) {
            log.error("Error in outputSchema", e);
            return null;
        }
    }

}
